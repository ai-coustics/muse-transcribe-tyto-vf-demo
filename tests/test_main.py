import asyncio
import io
import time
import wave

import numpy as np
from fastapi.testclient import TestClient

import app.main as main


def _wav() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    return target.getvalue()


def test_status_accepts_model_api_key(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("AIC_SDK_LICENSE", "test-license")

    response = TestClient(main.app).get("/api/status")

    assert response.json() == {
        "meta": True,
        "live": True,
        "ai_coustics": True,
        "credentials": "API key",
    }


def test_status_reports_missing_model_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AIC_SDK_LICENSE", "test-license")

    body = TestClient(main.app).get("/api/status").json()

    assert body["meta"] is False
    assert body["live"] is False
    assert body["credentials"] == "Not configured"


def test_service_worker_is_neutralized():
    response = TestClient(main.app).get("/sw.js")

    assert response.status_code == 200
    assert "unregister" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_invalid_api_key_error_is_actionable():
    error = RuntimeError("Client error '401 Unauthorized' for url 'https://api.meta.ai/v1/asr/transcribe'")

    result = main._public_muse_error(error)

    assert "not a valid Meta Model API key" in result
    assert "dev.meta.ai" in result


def test_missing_endpoint_error_points_at_the_base_url():
    error = RuntimeError("Client error '404 Not Found' for url 'https://api.meta.ai/asr/transcribe'")

    assert "MUSE_API_BASE" in main._public_muse_error(error)


def test_rejected_session_setup_is_actionable():
    # what the realtime handshake raises when the model name is not accepted
    error = RuntimeError("server rejected WebSocket connection: HTTP 400")

    result = main._public_muse_error(error)

    assert "MUSE_MODEL" in result
    assert "16 or 24 kHz" in result


def test_rate_limit_error_is_actionable():
    error = RuntimeError("Client error '429 Too Many Requests' for url 'https://api.meta.ai/v1/asr/transcribe'")

    assert "rate limit or quota" in main._public_muse_error(error)


def test_permission_denied_error_is_actionable():
    error = RuntimeError("Client error '403 Forbidden' for url 'https://api.meta.ai/v1/asr/transcribe'")

    assert "Muse Voice Transcribe" in main._public_muse_error(error)


def test_unsupported_sample_rate_message_reaches_the_client_verbatim():
    error = RuntimeError("Muse realtime accepts 16 kHz or 24 kHz audio, but the browser captured 44100 Hz")

    assert main._public_muse_error(error) == str(error)


def test_unknown_error_falls_back():
    assert "server log" in main._public_muse_error(RuntimeError("boom"))


def test_compare_runs_both_paths(monkeypatch):
    calls = []

    def fake_enhance(audio, level):
        return audio, {
            "model": "quail-vf-2.2-l-16khz",
            "enhancement_level": level,
            "audio_delay_ms": 30,
            "processing_ms": 4,
        }

    def fake_transcribe(audio, *options):
        calls.append(len(audio.samples))
        return {"text": "hello", "segments": [], "words": [], "elapsed_ms": 5}

    monkeypatch.setattr(main, "enhance_with_quail", fake_enhance)
    monkeypatch.setattr(main, "transcribe_with_muse", fake_transcribe)

    response = TestClient(main.app).post(
        "/api/compare",
        files={"audio_file": ("sample.wav", _wav(), "audio/wav")},
        data={"enhancement_level": "0.8"},
    )

    assert response.status_code == 200
    assert sorted(calls) == [8_000, 8_000]
    assert response.json()["raw"]["text"] == "hello"
    assert response.json()["quail"]["enhancement_level"] == 0.8


def test_comparison_audio_is_served_from_memory_and_expires(monkeypatch):
    def fake_enhance(audio, level):
        return audio, {"model": "quail", "enhancement_level": level, "audio_delay_ms": 30, "processing_ms": 4}

    monkeypatch.setattr(main, "enhance_with_quail", fake_enhance)
    monkeypatch.setattr(main, "transcribe_with_muse",
                        lambda audio, *a: {"text": "hi", "segments": [], "words": [], "elapsed_ms": 5})
    main.rate_limiter = main.RateLimiter(per_ip=0, daily=0)
    main.audio_store = main.AudioStore()
    client = TestClient(main.app)

    job_id = client.post("/api/compare", files={"audio_file": ("s.wav", _wav(), "audio/wav")}).json()["job_id"]

    for name in ("original.wav", "quail.wav"):
        served = client.get(f"/audio/{job_id}/{name}")
        assert served.status_code == 200
        assert served.headers["content-type"] == "audio/wav"

    # the uploaded source is never retrievable, and nothing survives expiry
    assert client.get(f"/audio/{job_id}/source.wav").status_code == 404
    main.audio_store = main.AudioStore(ttl=0)
    assert client.get(f"/audio/{job_id}/original.wav").status_code == 404


def test_compare_refuses_over_the_per_ip_limit(monkeypatch):
    monkeypatch.setattr(main, "enhance_with_quail",
                        lambda audio, level: (audio, {"model": "q", "enhancement_level": level,
                                                      "audio_delay_ms": 1, "processing_ms": 1}))
    monkeypatch.setattr(main, "transcribe_with_muse",
                        lambda audio, *a: {"text": "hi", "segments": [], "words": [], "elapsed_ms": 1})
    main.rate_limiter = main.RateLimiter(per_ip=1, window=600, daily=0)
    client = TestClient(main.app)
    files = {"audio_file": ("s.wav", _wav(), "audio/wav")}

    assert client.post("/api/compare", files=files).status_code == 200
    second = client.post("/api/compare", files={"audio_file": ("s.wav", _wav(), "audio/wav")})
    assert second.status_code == 429
    assert "Too many comparisons" in second.json()["detail"]


def test_daily_budget_is_enforced_across_addresses():
    main.rate_limiter = main.RateLimiter(per_ip=0, daily=1)
    assert main.rate_limiter.check("1.1.1.1") is None
    assert "daily limit" in main.rate_limiter.check("2.2.2.2")


def test_client_ip_prefers_the_proxy_header():
    assert main.client_ip({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, "127.0.0.1") == "9.9.9.9"
    assert main.client_ip({}, "127.0.0.1") == "127.0.0.1"


class FakeMuseSession:
    """Records what a paced sender delivers, with a handshake in the past."""

    def __init__(self, started_ago=0.0):
        self.started = time.monotonic() - started_ago
        self.chunks = []

    async def send_audio(self, chunk):
        self.chunks.append(chunk)


def _forward(session, items, chunk_size=8, tick=0.02, timeout=2.0):
    async def run():
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)
        await asyncio.wait_for(
            main._forward_audio(session, queue, chunk_size, tick), timeout=timeout
        )

    asyncio.run(run())


def test_forward_audio_sends_queued_chunks_in_order():
    session = FakeMuseSession()

    _forward(session, [b"a" * 8, b"b" * 8, main.END_OF_AUDIO])

    assert session.chunks[:2] == [b"a" * 8, b"b" * 8]


def test_forward_audio_stops_at_the_end_marker():
    session = FakeMuseSession()

    _forward(session, [b"a" * 8, main.END_OF_AUDIO, b"never"])

    assert b"never" not in session.chunks


def test_forward_audio_pays_off_the_handshake_gap_immediately():
    # Muse measures ingress from its handshake, so a sender that starts 200 ms
    # late owes ten 20 ms chunks and must deliver them at once. Captured audio
    # is spent first and the rest of the debt is covered with silence.
    session = FakeMuseSession(started_ago=0.2)

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait(b"a" * 8)
        task = asyncio.create_task(main._forward_audio(session, queue, 8, 0.02))
        await asyncio.sleep(0)             # one pass of the catch-up loop
        task.cancel()

    asyncio.run(run())

    assert len(session.chunks) >= 10
    assert session.chunks[0] == b"a" * 8
    assert set(session.chunks[1:]) == {bytes(8)}


def test_forward_audio_keeps_feeding_silence_while_the_browser_is_quiet():
    session = FakeMuseSession()

    async def run():
        queue = asyncio.Queue()
        task = asyncio.create_task(main._forward_audio(session, queue, 8, 0.01))
        await asyncio.sleep(0.12)          # nothing is ever queued
        task.cancel()

    asyncio.run(run())

    # roughly one chunk per 10 ms tick, all of it silence
    assert len(session.chunks) >= 8
    assert set(session.chunks) == {bytes(8)}

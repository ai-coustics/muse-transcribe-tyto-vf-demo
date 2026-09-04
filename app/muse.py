"""Meta Muse Voice Transcribe client.

Every Muse-specific wire detail lives in this module. Two surfaces, per
https://dev.meta.ai/docs/speech-to-text:

    POST {base}/asr/transcribe?sessionId=...   multipart: JSON `request` + WAV `audio`
    wss  {base}/asr/realtime?sessionId=...     JSON handshake, then raw PCM frames

Note this is *not* an OpenAI-compatible surface, unlike the Muse Spark text
models on the same host: the realtime socket authenticates in its first JSON
frame rather than an Authorization header, audio rides as bare binary frames,
and the field names are camelCase.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from app.audio import Audio, encode_wav


DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-voice-transcribe-1.0"
REQUEST_TIMEOUT_SECONDS = 180

# The realtime socket takes raw signed 16-bit little-endian mono PCM at exactly
# one of these rates, so the browser is asked to capture at PREFERRED_LIVE_RATE.
LIVE_ENCODINGS = {16_000: "PCM_16KHZ", 24_000: "PCM_24KHZ"}
PREFERRED_LIVE_RATE = 24_000

# The model consumes audio in 80 ms chunks, so outgoing audio is batched to that
# cadence rather than sent as one frame per capture block. Muse hangs up with
# "Ingress below real-time" on a session it judges to be underfed, and a steady
# stream of chunks at the rate it expects is what keeps it satisfied.
CHUNK_MS = 80


def chunk_bytes(sample_rate: int) -> int:
    """Bytes of 16-bit mono PCM in one 80 ms chunk at `sample_rate`."""
    return int(sample_rate * CHUNK_MS / 1000) * 2


def base_url() -> str:
    return os.environ.get("MUSE_API_BASE", DEFAULT_BASE_URL).rstrip("/")


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not configured")
    return key


def model() -> str:
    return os.environ.get("MUSE_MODEL", DEFAULT_MODEL)


def _session_id() -> str:
    return uuid.uuid4().hex


def _mode(diarization: bool) -> str:
    # ENDPOINTING still segments the audio into turns; DIARIZATION adds the
    # speaker label. PUSH_TO_TALK is for a single utterance, so neither path
    # here uses it.
    return "DIARIZATION" if diarization else "ENDPOINTING"


def _settings(diarization: bool, language_bias: list[str], keywords: list[str]) -> dict[str, Any]:
    settings: dict[str, Any] = {"model": model(), "mode": _mode(diarization)}
    if language_bias:
        settings["languageBias"] = language_bias
    if keywords:
        settings["keywords"] = keywords
    return settings


def parse_transcription(payload: dict[str, Any], elapsed_ms: int) -> dict[str, Any]:
    """Normalise a /asr/transcribe response into the shape the UI renders."""
    segments: list[dict[str, Any]] = []
    for turn in payload.get("turns") or []:
        text = str(turn.get("transcript") or "")
        if not text.strip():
            continue
        segments.append(
            {
                "speaker": str(turn.get("speaker") or ""),
                "text": text,
                "start_ms": turn.get("startMs"),
                "end_ms": turn.get("endMs"),
            }
        )

    whole = payload.get("transcript")
    text = str(whole).strip() if whole else " ".join(s["text"].strip() for s in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "elapsed_ms": elapsed_ms,
        "audio_duration_ms": payload.get("audioDurationMs"),
    }


def transcribe_with_muse(
    audio: Audio,
    language_bias: list[str],
    keywords: list[str],
    diarization: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    settings = {**_settings(diarization, language_bias, keywords), "audioEncoding": "WAV"}

    response = httpx.post(
        f"{base_url()}/asr/transcribe",
        params={"sessionId": _session_id()},
        headers={"Authorization": f"Bearer {api_key()}"},
        files={
            "request": (None, json.dumps(settings), "application/json"),
            "audio": ("audio.wav", encode_wav(audio), "audio/wav"),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_transcription(response.json(), round((time.perf_counter() - started) * 1000))


def _ws_base() -> str:
    http_base = base_url()
    scheme, _, rest = http_base.partition("://")
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}"


class MuseLiveSession:
    """One realtime transcription stream.

    `partialMode` is CUMULATIVE, so each partial already carries the whole
    utterance and the UI can replace rather than append. A `speaker` event
    labels the span behind it and arrives once per turn, just before the turn
    is finalised, so the latest label is the one that applies.
    """

    def __init__(self, socket: Any, session_id: str):
        self.socket = socket
        self.session_id = session_id
        #: Muse measures ingress from the handshake, so callers pace against this.
        self.started = time.monotonic()
        self.speaker = ""
        self._last_final = ""

    async def send_audio(self, pcm16: bytes) -> None:
        await self.socket.send(pcm16)

    async def end_audio(self) -> None:
        await self.socket.send(json.dumps({"type": "endStream"}))

    def _final(self, text: str) -> dict[str, Any] | None:
        # DIARIZATION finalises through speechComplete and PUSH_TO_TALK through
        # transcript.final; guard in case a deployment sends both.
        if not text.strip() or text.strip() == self._last_final:
            return None
        self._last_final = text.strip()
        return {"final": True, "text": text.strip(), "speaker": self.speaker}

    async def transcripts(self) -> AsyncIterator[dict[str, Any]]:
        async for raw in self.socket:
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            kind = str(event.get("type", ""))
            if kind == "error":
                raise RuntimeError(str(event.get("message") or event))
            if kind == "speaker":
                self.speaker = str(event.get("label") or "")
            elif kind == "speechComplete":
                update = self._final(str(event.get("transcript") or ""))
                if update:
                    yield update
            elif kind == "transcript":
                text = str(event.get("transcript") or "")
                if event.get("final"):
                    update = self._final(text)
                    if update:
                        yield update
                elif text.strip():
                    yield {"final": False, "text": text.strip(), "speaker": self.speaker}


@asynccontextmanager
async def open_live_session(
    sample_rate: int,
    language_bias: list[str],
    keywords: list[str],
    diarization: bool = True,
):
    import websockets

    encoding = LIVE_ENCODINGS.get(sample_rate)
    if encoding is None:
        supported = " or ".join(f"{rate // 1000} kHz" for rate in sorted(LIVE_ENCODINGS))
        raise RuntimeError(
            f"Muse realtime accepts {supported} audio, but the browser captured {sample_rate} Hz"
        )

    handshake = {
        "authorization": {"accessToken": f"Bearer {api_key()}"},
        "audioEncoding": encoding,
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": False,
        **_settings(diarization, language_bias, keywords),
    }

    url = f"{_ws_base()}/asr/realtime?sessionId={_session_id()}"
    async with websockets.connect(url, max_size=None) as socket:
        await socket.send(json.dumps(handshake))
        ack = json.loads(await socket.recv())
        if not isinstance(ack, dict) or not ack.get("sessionId"):
            raise RuntimeError(f"Muse realtime handshake was rejected: {str(ack)[:200]}")
        yield MuseLiveSession(socket, str(ack["sessionId"]))

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from app.audio import AudioError, decode_wav, encode_wav
from app.limits import AudioStore, RateLimiter, client_ip
from app.muse import CHUNK_MS, chunk_bytes, open_live_session, transcribe_with_muse
from app.services import (
    LiveQuailProcessor,
    LiveTytoAnalyzer,
    analyze_with_tyto,
    enhance_with_quail,
)


load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
comparison_slots = asyncio.Semaphore(2)
# Room for ~5 s of 80 ms chunks per path, so a brief network stall is absorbed
# and drained afterwards instead of stalling the browser reader.
LIVE_QUEUE_CHUNKS = 64

# Uploaded audio is never written to disk; it lives here only long enough for
# the browser to play the A/B comparison back.
audio_store = AudioStore()
rate_limiter = RateLimiter()

app = FastAPI(title="Muse Voice Transcribe × Quail transcription lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/audio/{job_id}/{name}")
def comparison_audio(job_id: str, name: str):
    data = audio_store.get(job_id, name)
    if data is None:
        raise HTTPException(404, "That audio has expired")
    return Response(data, media_type="audio/wav")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/sw.js")
def neutral_service_worker():
    return Response(
        "self.addEventListener('install',()=>self.skipWaiting());"
        "self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-store"},
    )


@app.get("/api/status")
def status():
    model_api_key = bool(os.environ.get("MODEL_API_KEY"))
    return {
        "meta": model_api_key,
        "live": model_api_key,
        "ai_coustics": bool(os.environ.get("AIC_SDK_LICENSE")),
        "credentials": "API key" if model_api_key else "Not configured",
    }


def _csv(value: str, limit: int) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()][:limit]


def _public_muse_error(exc: Exception) -> str:
    message = str(exc)
    if "kHz audio" in message or "not configured" in message:
        return message  # already phrased for the client
    if "401" in message or "Unauthorized" in message:
        return (
            "MODEL_API_KEY is not a valid Meta Model API key. "
            "Create one in the Model API dashboard at https://dev.meta.ai."
        )
    if "429" in message or "rate_limit" in message:
        return "Meta Model API rate limit or quota exceeded. Wait a moment and retry."
    if "403" in message or "Forbidden" in message:
        return "Meta denied access. Check that the API key can use Muse Voice Transcribe."
    if "404" in message or "Not Found" in message:
        return (
            "The Muse transcription endpoint was not found. "
            "Check MUSE_API_BASE; it should point at the /v1 root."
        )
    if "400" in message or "Bad Request" in message:
        return (
            "Muse rejected the session setup. Check MUSE_MODEL and that the audio "
            "is 16 or 24 kHz mono."
        )
    return "Live transcription failed. Check the server log for details."


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


async def _receive_live_transcript(
    session: Any, websocket: WebSocket, path: str, send_lock: asyncio.Lock
) -> None:
    async for update in session.transcripts():
        async with send_lock:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "path": path,
                    "final": update["final"],
                    "text": update["text"],
                    "speaker": update.get("speaker", ""),
                }
            )


def _reraise_first_failure(*groups: list[asyncio.Task]) -> None:
    """Surface a task's exception promptly instead of at the end of the session."""
    for group in groups:
        for task in group:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()


#: queued instead of a chunk to tell a sender the stream is finished
END_OF_AUDIO = object()


async def _forward_audio(
    session: Any, queue: asyncio.Queue, chunk_size: int, tick_seconds: float
) -> None:
    """Feed one path's audio to Muse on the clock Muse itself is measuring.

    Muse closes a session whose ingress falls behind real time, measured from
    its own handshake. Two sessions open one after the other and neither can
    send until the browser delivers, so a session is already in debt by the time
    audio starts flowing, and a shortfall can never be repaid with audio that
    has not been captured yet.

    So the schedule is anchored on `session.started` and each pass sends
    everything owed since then, substituting silence when the browser has not
    delivered in time. That covers the handshake gap and any later stall, and
    drains a backlog promptly so a bursty producer catches up.

    Sending is decoupled from receiving for the same reason: a stall on one
    socket must not hold up the other path or stop us reading from the browser.
    """
    silence = bytes(chunk_size)
    sent = 0

    while True:
        due = int((time.monotonic() - session.started) / tick_seconds) + 1
        while sent < due:
            try:
                chunk = queue.get_nowait()
            except asyncio.QueueEmpty:
                chunk = silence
            if chunk is END_OF_AUDIO:
                return
            await session.send_audio(chunk)
            sent += 1

        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=tick_seconds)
        except TimeoutError:
            continue
        if chunk is END_OF_AUDIO:
            return
        await session.send_audio(chunk)
        sent += 1


async def _publish_live_tyto(
    tyto: LiveTytoAnalyzer,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
            break
        except TimeoutError:
            pass
        if not tyto.ready:
            continue
        try:
            result = await asyncio.to_thread(tyto.analyze)
            if result:
                async with send_lock:
                    await websocket.send_json({"type": "tyto", **result})
        except Exception as exc:
            async with send_lock:
                await websocket.send_json(
                    {"type": "tyto_warning", "message": f"Tyto analysis paused: {exc}"}
                )


async def _run_tyto_only(
    websocket: WebSocket,
    tyto: LiveTytoAnalyzer,
    block_size: int,
    muse_error: Exception,
) -> None:
    send_lock = asyncio.Lock()
    stop = asyncio.Event()
    task = asyncio.create_task(_publish_live_tyto(tyto, websocket, send_lock, stop))
    try:
        await websocket.send_json(
            {"type": "warning", "message": _public_muse_error(muse_error)}
        )
        await websocket.send_json(
            {
                "type": "status",
                "status": "ready",
                "text": "Tyto listening. First score after 5 seconds",
                "transcription_available": False,
            }
        )
        while True:
            incoming = await websocket.receive()
            if incoming["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if incoming.get("text"):
                command = json.loads(incoming["text"])
                if command.get("action") == "stop":
                    break
                continue
            audio_bytes = incoming.get("bytes")
            if not audio_bytes:
                continue
            samples = np.frombuffer(audio_bytes, dtype="<f4")
            if len(samples) == block_size:
                tyto.buffer(samples)
        await websocket.send_json({"type": "complete", "mode": "tyto"})
    finally:
        stop.set()
        await asyncio.gather(task, return_exceptions=True)


@app.websocket("/ws/live")
async def live_compare(websocket: WebSocket):
    await websocket.accept()
    refused = rate_limiter.check(
        client_ip(websocket.headers, websocket.client.host if websocket.client else None)
    )
    if refused:
        await websocket.send_json({"type": "error", "message": refused})
        await websocket.close()
        return
    processor: LiveQuailProcessor | None = None
    tyto: LiveTytoAnalyzer | None = None
    tyto_task: asyncio.Task | None = None
    tyto_stop = asyncio.Event()
    receivers: list[asyncio.Task] = []
    senders: list[asyncio.Task] = []
    try:
        options = await websocket.receive_json()
        sample_rate = int(options.get("sample_rate", 48_000))
        block_size = int(options.get("block_size", round(sample_rate * 0.015)))
        level = float(options.get("enhancement_level", 0.5))
        use_tyto = bool(options.get("tyto", True))
        if not 8_000 <= sample_rate <= 192_000 or not 1 <= block_size <= 8192:
            raise ValueError("Unsupported microphone audio format")
        if not 0 <= level <= 1:
            raise ValueError("Enhancement level must be between 0 and 1")

        loading = "Loading Quail and Tyto" if use_tyto else "Loading Quail"
        await websocket.send_json({"type": "status", "status": "loading", "text": loading})
        async with comparison_slots:
            processor = await asyncio.to_thread(
                LiveQuailProcessor, sample_rate, block_size, level
            )
            if use_tyto:
                tyto = await asyncio.to_thread(LiveTytoAnalyzer, sample_rate, block_size)
            language_bias = _csv(str(options.get("language_bias", "")), 20)
            keywords = _csv(str(options.get("keywords", "")), 1_000)
            try:
                async with (
                    open_live_session(sample_rate, language_bias, keywords) as raw_session,
                    open_live_session(sample_rate, language_bias, keywords) as enhanced_session,
                ):
                    send_lock = asyncio.Lock()
                    receivers = [
                        asyncio.create_task(
                            _receive_live_transcript(raw_session, websocket, "raw", send_lock)
                        ),
                        asyncio.create_task(
                            _receive_live_transcript(
                                enhanced_session, websocket, "enhanced", send_lock
                            )
                        ),
                    ]
                    queues = {
                        "raw": asyncio.Queue(maxsize=LIVE_QUEUE_CHUNKS),
                        "enhanced": asyncio.Queue(maxsize=LIVE_QUEUE_CHUNKS),
                    }
                    chunk = chunk_bytes(sample_rate)
                    tick = CHUNK_MS / 1000
                    senders = [
                        asyncio.create_task(
                            _forward_audio(raw_session, queues["raw"], chunk, tick)
                        ),
                        asyncio.create_task(
                            _forward_audio(enhanced_session, queues["enhanced"], chunk, tick)
                        ),
                    ]
                    if tyto is not None:
                        tyto_task = asyncio.create_task(
                            _publish_live_tyto(tyto, websocket, send_lock, tyto_stop)
                        )
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": "ready",
                            "text": "Listening",
                            "quail_delay_ms": processor.delay_ms,
                            "transcription_available": True,
                        }
                    )
                    pending = np.zeros(0, dtype=np.float32)
                    outgoing = {"raw": bytearray(), "enhanced": bytearray()}
                    while True:
                        _reraise_first_failure(receivers, senders)
                        incoming = await websocket.receive()
                        if incoming["type"] == "websocket.disconnect":
                            raise WebSocketDisconnect()
                        if incoming.get("text"):
                            command = json.loads(incoming["text"])
                            if command.get("action") == "stop":
                                break
                            continue
                        audio_bytes = incoming.get("bytes")
                        if not audio_bytes:
                            continue
                        # Re-block rather than dropping odd-sized frames: Quail
                        # needs its exact block size, but discarded audio reads
                        # as ingress below real time and Muse hangs up.
                        pending = np.concatenate(
                            (pending, np.frombuffer(audio_bytes, dtype="<f4"))
                        )
                        while len(pending) >= block_size:
                            block, pending = pending[:block_size], pending[block_size:]
                            if tyto is not None:
                                tyto.buffer(block)
                            outgoing["raw"] += _pcm16(block)
                            outgoing["enhanced"] += _pcm16(processor.process(block))
                        for path, buffered in outgoing.items():
                            while len(buffered) >= chunk:
                                await queues[path].put(bytes(buffered[:chunk]))
                                del buffered[:chunk]
                    for path, buffered in outgoing.items():
                        if buffered:
                            await queues[path].put(bytes(buffered))
                        await queues[path].put(END_OF_AUDIO)
                    await asyncio.gather(*senders)
                    await asyncio.gather(raw_session.end_audio(), enhanced_session.end_audio())
                    tyto_stop.set()
                    if tyto_task is not None:
                        await asyncio.gather(tyto_task, return_exceptions=True)
                    done, pending = await asyncio.wait(receivers, timeout=5)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*done, *pending, return_exceptions=True)
                    await websocket.send_json({"type": "complete"})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                if tyto is None or tyto_task is not None:
                    raise
                logger.warning("Muse unavailable, continuing with Tyto", exc_info=exc)
                await _run_tyto_only(websocket, tyto, block_size, exc)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("live transcription failed")
        try:
            await websocket.send_json({"type": "error", "message": _public_muse_error(exc)})
        except Exception:
            pass
    finally:
        tyto_stop.set()
        if tyto_task is not None and not tyto_task.done():
            tyto_task.cancel()
            await asyncio.gather(tyto_task, return_exceptions=True)
        for task in (*receivers, *senders):
            if not task.done():
                task.cancel()
        await asyncio.gather(*receivers, *senders, return_exceptions=True)
        if processor is not None:
            try:
                await asyncio.to_thread(processor.close)
            except Exception:
                pass
        if tyto is not None:
            try:
                await asyncio.to_thread(tyto.close)
            except Exception:
                pass


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "WAV file is too large (150 MB maximum)")
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/compare")
async def compare(
    request: Request,
    audio_file: UploadFile = File(...),
    enhancement_level: float = Form(0.5),
    language_bias: str = Form(""),
    keywords: str = Form(""),
    diarization: bool = Form(False),
    tyto: bool = Form(False),
):
    refused = rate_limiter.check(client_ip(request.headers, request.client.host if request.client else None))
    if refused:
        raise HTTPException(429, refused)
    if not 0 <= enhancement_level <= 1:
        raise HTTPException(400, "Enhancement level must be between 0 and 1")
    try:
        uploaded_bytes = await _read_limited(audio_file)
        raw_audio = decode_wav(uploaded_bytes)
    except AudioError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    raw_wav = encode_wav(raw_audio)
    del uploaded_bytes

    try:
        async with comparison_slots, asyncio.timeout(180):
            enhanced_audio, quail = await asyncio.to_thread(
                enhance_with_quail, raw_audio, enhancement_level
            )
            enhanced_wav = encode_wav(enhanced_audio)

            args = (
                _csv(language_bias, 20),
                _csv(keywords, 1_000),
                diarization,
            )
            raw_task = asyncio.to_thread(transcribe_with_muse, raw_audio, *args)
            enhanced_task = asyncio.to_thread(transcribe_with_muse, enhanced_audio, *args)
            if tyto:
                raw_result, enhanced_result, tyto_result = await asyncio.gather(
                    raw_task, enhanced_task, asyncio.to_thread(analyze_with_tyto, raw_audio)
                )
            else:
                raw_result, enhanced_result = await asyncio.gather(raw_task, enhanced_task)
                tyto_result = None
    except Exception as exc:
        logger.exception("comparison failed for job %s", job_id)
        if "not configured" in str(exc):
            public_error = str(exc)
        elif isinstance(exc, TimeoutError):
            public_error = "Comparison timed out after 3 minutes"
        else:
            public_error = _public_muse_error(exc)
        raise HTTPException(502, public_error) from exc

    audio_store.put(job_id, {"original.wav": raw_wav, "quail.wav": enhanced_wav})
    result = {
        "job_id": job_id,
        "duration_seconds": round(raw_audio.duration_seconds, 2),
        "raw": {**raw_result, "audio_url": f"/audio/{job_id}/original.wav"},
        "enhanced": {**enhanced_result, "audio_url": f"/audio/{job_id}/quail.wav"},
        "quail": quail,
        "tyto": tyto_result,
    }
    return result

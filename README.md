# Muse Voice Transcribe × Quail transcription lab

Compare speech through two paths:

1. original audio → Muse Voice Transcribe
2. original audio → Quail Voice Focus 2.2 → Muse Voice Transcribe

Live mode streams microphone PCM to two Muse Voice Transcribe realtime sessions while Quail Voice Focus processes the second path locally. Both transcript panels update as you speak, and diarized turns are labelled as the speaker changes. Optional Tyto Audio Insight scores the original microphone signal after a five-second warm-up and refreshes once per second. File mode accepts a WAV upload, runs both Muse requests concurrently, and adds playback, optional WER scoring, diarization, timestamps, and Tyto analysis.

## Setup

Prerequisites: Python 3.11 to 3.13, a Meta Model API key with access to Muse Voice Transcribe, and an ai-coustics SDK license.

```bash
cp .env.example .env
# Edit .env with MODEL_API_KEY and AIC_SDK_LICENSE.
# Create the Model API key at https://dev.meta.ai
set -a; source .env; set +a
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

## The Muse integration

Muse Voice Transcribe has its own API on `api.meta.ai`, separate from the
OpenAI-compatible surface the Muse Spark text models use. Every wire detail is
confined to [app/muse.py](app/muse.py):

| Path | Surface |
| --- | --- |
| File compare | `POST /v1/asr/transcribe?sessionId=…`, `multipart/form-data` with a JSON `request` part and a WAV `audio` part |
| `/ws/live` | `wss://api.meta.ai/v1/asr/realtime?sessionId=…`, JSON handshake frame, then raw PCM binary frames |

Three things differ from an OpenAI-style API and are easy to trip over:

- **The realtime socket authenticates in its first JSON frame**, not an
  `Authorization` header: `{"authorization": {"accessToken": "Bearer …"}}`.
  The server replies with `{"sessionId": …}` before it will accept audio.
- **Live audio must be 16 or 24 kHz** signed 16-bit little-endian mono PCM, sent
  as bare binary frames. The browser is therefore pinned to 24 kHz with
  `new AudioContext({ sampleRate: 24000 })`; a microphone's native 48 kHz is
  rejected. Closing the input is `{"type": "endStream"}`, which leaves the
  socket open for the final result.
- **`mode` replaces per-feature flags.** `DIARIZATION` adds speaker labels,
  `ENDPOINTING` just segments turns, and `PUSH_TO_TALK` handles a single
  utterance. This app uses `DIARIZATION` when the box is ticked and
  `ENDPOINTING` otherwise. `partialMode` is `CUMULATIVE`, so each partial
  carries the whole utterance and the UI replaces rather than appends.

Biasing rides along as `languageBias` (language *names*, e.g. `English, German`,
not BCP-47 codes) and `keywords`.

There are no word-level timestamps. The response carries turn-level `startMs`,
`endMs` and `speaker` per turn, which is what the "Turn timestamps" box shows.

### Keeping the live sessions alive

Muse closes a realtime session with `1008 Ingress below real-time` when it
judges the session underfed, and it measures that from **its own handshake**.
Two sessions open one after the other, and neither can send anything until the
browser delivers its first block, so a session is in debt before audio starts
flowing. That debt cannot be repaid later: audio which has not been captured yet
cannot be sent early.

So each path has a sender task ([`_forward_audio`](app/main.py)) anchored on its
own `session.started`, which delivers everything owed since the handshake and
substitutes silence whenever the browser has not produced a chunk in time. This
covers the startup gap and any later stall — a throttled tab or a GC pause —
and drains a backlog promptly so a bursty producer catches up rather than
accumulating a deficit.

Sending is decoupled from receiving for the same reason. Audio is also re-blocked
rather than discarded when a frame does not match Quail's block size, since
dropped audio reads as ingress below real time just the same.

Nothing is hardcoded, so the same app can point at a self-hosted deployment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_API_KEY` | — | Meta Model API key, required |
| `MUSE_API_BASE` | `https://api.meta.ai/v1` | Base URL; the live path derives `wss://` from it |
| `MUSE_MODEL` | `muse-voice-transcribe-1.0` | Model id for both paths |

If the server reports `401`, recreate the key in the Model API dashboard. A `400`
on the realtime handshake usually means `MUSE_MODEL` is wrong — note the model id
carries its version, `muse-voice-transcribe-1.0`, and the bare
`muse-voice-transcribe` is rejected.

The first enhanced run downloads `quail-vf-2.2-l-16khz` (~20 MB). Enabling Tyto downloads `tyto-1.1-l-16khz` (~13 MB). Uploaded audio is never written to disk. It is held in memory only long enough for the browser to play the A/B comparison back (10 minutes by default), then dropped.

## Notes

- Quail Voice Focus is intended for a single primary speaker. Its first few seconds are a warm-up period, so longer samples are more representative. Muse itself handles 20+ speakers, so the unenhanced path is the fairer one for heavily overlapped audio.
- Both paths use the same normalized 16 kHz mono control signal. The untouched uploaded WAV is retained alongside the run for auditability but is not sent directly to Muse; this keeps Quail as the only treatment variable.
- This app caps recordings at 15 minutes. Muse supports far longer audio — over an hour natively — so raise `MAX_DURATION_SECONDS` in [app/audio.py](app/audio.py) if you want longer runs, keeping the request timeout in mind.
- Endpointing is part of the model rather than a separate VAD stage, so both paths run in `ENDPOINTING` mode (or `DIARIZATION`) and Muse decides the turn boundaries. Nothing is trimmed before it is sent, which keeps the two paths comparable.

## Privacy

Uploaded audio is never written to disk. `/api/compare` holds the decoded original and
the Quail-enhanced result in memory only so the browser can play the A/B comparison
back, and drops them after `AUDIO_TTL_SECONDS` (10 minutes by default). The bytes the
client uploaded are never re-served. Live mode persists nothing at all.

Audio is still sent to Meta for transcription — twice per file comparison, and as a
stream in live mode. That is inherent to comparing Muse output.

## Limits

Modal has no built-in per-IP rate limiting, so it lives in `app/limits.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_IP` | 5 | Comparisons per address per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 600 | Sliding window length |
| `RATE_LIMIT_DAILY` | 200 | Global daily budget across all callers |
| `AUDIO_TTL_SECONDS` | 600 | How long playback audio stays in memory |

Set any of them to `0` to disable it. Both `/api/compare` and `/ws/live` are limited —
a live session opens two Muse realtime streams, so it is not cheaper than a file run.

## Deploying to Modal

```bash
modal secret create muse-demo-secrets MODEL_API_KEY=... AIC_SDK_LICENSE=... -e aic-demos
modal deploy modal_app.py -e aic-demos
```

`modal_app.py` pins `max_containers=1`. The rate limiter keeps per-process state, so it
is only accurate while one container serves every request. If you raise that, move the
limiter to a `modal.Dict` first or the limits become per-container.

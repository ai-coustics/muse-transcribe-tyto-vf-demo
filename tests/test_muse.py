import asyncio
import json

from app.muse import MuseLiveSession, parse_transcription


def test_parse_transcription_maps_diarized_turns():
    payload = {
        "sessionId": "9f1c",
        "transcript": "How is the weather? It is raining.",
        "audioDurationMs": 8240,
        "turns": [
            {"turnId": 1, "startMs": 1520, "endMs": 4640, "transcript": "How is the weather?", "speaker": "A"},
            {"turnId": 2, "startMs": 4900, "endMs": 6100, "transcript": "It is raining.", "speaker": "B"},
        ],
    }

    parsed = parse_transcription(payload, 321)

    assert parsed["text"] == "How is the weather? It is raining."
    assert parsed["segments"][1] == {
        "speaker": "B",
        "text": "It is raining.",
        "start_ms": 4900,
        "end_ms": 6100,
    }
    assert parsed["audio_duration_ms"] == 8240
    assert parsed["elapsed_ms"] == 321


def test_parse_transcription_joins_turns_when_no_whole_transcript():
    parsed = parse_transcription(
        {"turns": [{"transcript": "one "}, {"transcript": "two"}, {"transcript": "  "}]}, 5
    )

    assert parsed["text"] == "one two"
    assert len(parsed["segments"]) == 2


def test_parse_transcription_handles_an_empty_result():
    parsed = parse_transcription({"transcript": "", "turns": []}, 5)

    assert parsed["text"] == ""
    assert parsed["segments"] == []


class FakeSocket:
    """Stands in for the realtime WebSocket."""

    def __init__(self, events):
        self.events = events
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    def __aiter__(self):
        return self._replay()

    async def _replay(self):
        for event in self.events:
            yield json.dumps(event)


def _drain(session):
    async def run():
        return [update async for update in session.transcripts()]

    return asyncio.run(run())


def test_live_session_labels_the_final_turn_with_its_speaker():
    # DIARIZATION finalises through speechComplete, and the speaker event
    # arrives first, labelling the span behind it.
    session = MuseLiveSession(FakeSocket([
        {"type": "speechStart"},
        {"type": "transcript", "transcript": "How is the", "final": False},
        {"type": "transcript", "transcript": "How is the weather?", "final": False},
        {"type": "speaker", "label": "A", "audioProcessedMs": 2480},
        {"type": "speechComplete", "turnId": 1, "transcript": "How is the weather?"},
    ]), "session-1")

    updates = _drain(session)

    assert [(u["final"], u["text"]) for u in updates] == [
        (False, "How is the"),
        (False, "How is the weather?"),
        (True, "How is the weather?"),
    ]
    assert updates[-1]["speaker"] == "A"
    assert updates[0]["speaker"] == ""


def test_live_session_does_not_emit_a_final_twice():
    session = MuseLiveSession(FakeSocket([
        {"type": "transcript", "transcript": "Done.", "final": True},
        {"type": "speechComplete", "turnId": 1, "transcript": "Done."},
    ]), "session-1")

    assert [u["text"] for u in _drain(session)] == ["Done."]


def test_live_session_raises_on_an_error_event():
    session = MuseLiveSession(FakeSocket([{"type": "error", "message": "bad audio"}]), "s")

    try:
        _drain(session)
    except RuntimeError as exc:
        assert "bad audio" in str(exc)
    else:
        raise AssertionError("expected the error event to raise")


def test_live_session_sends_audio_as_binary_and_ends_with_a_control_frame():
    socket = FakeSocket([])
    session = MuseLiveSession(socket, "s")

    async def run():
        await session.send_audio(b"\x00\x01")
        await session.end_audio()

    asyncio.run(run())

    assert socket.sent[0] == b"\x00\x01"
    assert json.loads(socket.sent[1]) == {"type": "endStream"}

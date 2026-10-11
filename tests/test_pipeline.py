"""End-to-end behaviour of Video with deterministic doubles (energy VAD + ToneASR)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import TONE_EVENTS
from saccade import (
    AudioStreamNotFoundError,
    IndexLockedError,
    NotIndexedError,
    ProgressEvent,
    Stage,
    UnsupportedFormatError,
    VadConfig,
)
from support import ToneASR, place, silence, tone, write_media

TEXTS = {
    440: "The deployment failed because of authentication.",
    660: "Lunch was good today.",
    880: "We fixed the authentication configuration with managed identity.",
    1100: "Now the deployment works.",
}


def test_speaker_clustering_is_bounded_and_resumable(monkeypatch) -> None:
    from saccade.config import DiarizationConfig
    from saccade.diarization import SpeakerDiarizer

    config = DiarizationConfig(model="stand-in.onnx", max_speakers=2)
    vectors = iter([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    sizes = []

    def embed(self, audio):
        sizes.append(len(audio))
        return np.array(next(vectors), dtype=np.float32)

    monkeypatch.setattr(SpeakerDiarizer, "_embed", embed)
    diarizer = SpeakerDiarizer(config)
    audio = np.ones(16000 * 10, dtype=np.float32)
    assert diarizer.label(audio) == "SPEAKER_1"
    assert diarizer.label(audio) == "SPEAKER_2"
    restored = SpeakerDiarizer(config)
    restored.restore(json.loads(json.dumps(diarizer.snapshot())))
    assert restored.label(audio) == "SPEAKER_1"
    assert restored.label(audio) == "SPEAKER_2"
    assert len(restored.snapshot()["centroids"]) == 2
    assert sizes == [48000] * 4
    assert restored.label(np.zeros(100, dtype=np.float32)) is None


def test_speaker_clustering_rejects_invalid_embeddings(monkeypatch) -> None:
    from saccade import TranscriptionError
    from saccade.config import DiarizationConfig
    from saccade.diarization import SpeakerDiarizer

    diarizer = SpeakerDiarizer(DiarizationConfig(model="stand-in.onnx"))
    monkeypatch.setattr(SpeakerDiarizer, "_embed", lambda self, audio: np.array([np.nan]))
    with pytest.raises(TranscriptionError, match="finite"):
        diarizer.label(np.ones(16000, dtype=np.float32))


def test_speaker_storage_migrates_and_saves_state(tmp_path: Path) -> None:
    from saccade.index.database import Database
    from saccade.models.segment import TranscriptSegment

    path = tmp_path / "index.db"
    db = Database(path)
    with db.session() as conn:
        conn.execute("ALTER TABLE transcript_segments DROP COLUMN speaker")
        conn.execute("UPDATE metadata SET value = '2' WHERE key = 'schema_version'")
    db.close()
    db = Database(path)
    run = db.create_run(kind="transcript", config_key="speakers", config={}, library_version="test")
    state = {"centroids": [[1.0, 0.0]], "counts": [1]}
    segment = TranscriptSegment("seg_00000", 0, 1, "hello", speaker="SPEAKER_1")
    with db.session() as conn:
        db.commit_batch(
            conn,
            run.id,
            segments=[(0, segment)],
            regions=[],
            chunks=[],
            processed_until=1,
            speech_seconds=1,
            elapsed_seconds=0,
            speaker_state=state,
        )
    assert db.segments(run.id)[0][1].speaker == "SPEAKER_1"
    assert db.speaker_state(run.id) == state
    db.rename_speaker(run.id, "SPEAKER_1", "Ana")
    assert db.segments(run.id)[0][1].speaker == "Ana"
    assert db.speaker_state(run.id) == state
    db.delete_run(run.id)
    assert db.speaker_state(run.id) is None
    assert db.speaker_names(run.id) == {}
    db.close()


def expected_segments() -> list[tuple[str, float, float]]:
    return [(f"tone {f}.", s, s + d) for s, f, d in TONE_EVENTS]


def test_speaker_labels_resume_search_context_and_exports(
    make_video, tmp_path: Path, monkeypatch
) -> None:
    from saccade import DiarizationConfig
    from saccade.diarization import SpeakerDiarizer

    path = write_media(
        tmp_path / "speakers.wav",
        place(
            65,
            [
                (1, tone(440, 2)),
                (30, tone(880, 2)),
                (60, tone(440, 2)),
            ],
        ),
    )
    model = tmp_path / "speaker.onnx"
    model.write_bytes(b"stand-in")
    config = DiarizationConfig(model=str(model))

    def embed(self, audio):
        spectrum = np.abs(np.fft.rfft(audio))
        frequency = np.argmax(spectrum) * 16000 / len(audio)
        return np.array([1, 0] if frequency < 600 else [0, 1], dtype=np.float32)

    monkeypatch.setattr(SpeakerDiarizer, "_embed", embed)
    video = make_video(path, ToneASR(), diarization=config)
    iterator = video.transcribe()
    assert next(iterator).speaker == "SPEAKER_1"
    iterator.close()
    assert len(video.transcript()) == 1
    video.rename_speaker("SPEAKER_1", "Ana")
    resumed_backend = ToneASR()
    resumed = make_video(path, resumed_backend, diarization=config)
    segments = list(resumed.transcribe())
    assert resumed_backend.calls > 0
    assert [segment.speaker for segment in segments] == ["Ana", "SPEAKER_2", "Ana"]
    assert [segment.start for segment in segments] == pytest.approx([1, 30, 60], abs=0.05)
    assert resumed.search("tone", speaker="SPEAKER_2")[0].segment_ids == ("seg_00001",)
    assert resumed.search("tone", speaker="Ana", expand=90)[0].segment_ids == (
        "seg_00000",
        "seg_00002",
    )
    assert resumed.search("tone", speaker="missing") == []
    assert "| Ana]" in resumed.context("tone").text
    transcript = resumed.transcript()
    assert "Ana: tone 440." in transcript.to_srt()
    assert "<v Ana>tone 440.</v>" in transcript.to_vtt()
    assert json.loads(transcript.to_json())["segments"][0]["speaker"] == "Ana"
    from saccade.ask import build_messages
    from saccade.exceptions import ConfigError

    messages, _, _, mode = build_messages(
        resumed, "Who spoke?", evidence_tokens=48000, max_images=0, images=False
    )
    assert mode == "full" and "| Ana]" in str(messages)
    with pytest.raises(ValueError, match="already assigned"):
        resumed.rename_speaker("SPEAKER_2", "Ana")
    with pytest.raises(ConfigError, match="canonical"):
        resumed.rename_speaker("SPEAKER_1", "SPEAKER_2")
    with pytest.raises(ValueError, match="Unknown"):
        resumed.rename_speaker("SPEAKER_3", "Someone")
    backend = ToneASR()
    assert list(make_video(path, backend, diarization=config).transcribe()) == segments
    assert backend.calls == 0
    plain = make_video(path, ToneASR())
    assert (
        plain._cache_key(plain._backend_for())[0] != resumed._cache_key(resumed._backend_for())[0]
    )
    assert all(segment.speaker is None for segment in plain.transcribe())


def test_timestamps_are_restored_to_source_timeline(make_video, tone_video: Path) -> None:
    backend = ToneASR()
    segments = list(make_video(tone_video, backend).transcribe())
    assert [s.text for s in segments] == [t for t, _, _ in expected_segments()]
    for segment, (_, start, end) in zip(segments, expected_segments(), strict=True):
        assert segment.start == pytest.approx(start, abs=0.05)
        assert segment.end == pytest.approx(end, abs=0.05)
    assert [s.id for s in segments] == ["seg_00000", "seg_00001", "seg_00002", "seg_00003"]


def test_silence_is_never_sent_to_asr(make_video, tone_video: Path) -> None:
    backend = ToneASR()
    list(make_video(tone_video, backend).transcribe())
    speech = sum(d for _, _, d in TONE_EVENTS)
    assert backend.audio_seconds == pytest.approx(speech + len(TONE_EVENTS) * 0.4, abs=0.6)
    assert backend.audio_seconds < 20  # of a 90 s file


def test_second_run_reuses_cache_and_new_queries_do_not_transcribe(
    make_video, tone_video: Path
) -> None:
    first = ToneASR(texts=TEXTS)
    summary = make_video(tone_video, first).index()
    assert summary.status == "complete" and summary.segments == 4 and not summary.cached
    calls = first.calls

    second = ToneASR(texts=TEXTS)
    video = make_video(tone_video, second)
    again = video.index()
    assert again.cached and again.segments == 4
    video.search("deployment")
    video.context("Why did the deployment fail?")
    assert list(video.transcribe())  # served from SQLite
    assert second.calls == 0
    assert calls > 0


def test_cache_key_tracks_configuration(make_video, tone_video: Path, energy_vad) -> None:
    make_video(tone_video, ToneASR()).index()
    other_vad = ToneASR()
    make_video(tone_video, other_vad, vad=VadConfig(padding_ms=300)).index()
    assert other_vad.calls > 0, "VAD settings are part of the cache key"
    other_lang = ToneASR()
    make_video(tone_video, other_lang, language="pt").index()
    assert other_lang.calls > 0, "an explicit language is part of the cache key"
    info = make_video(tone_video, ToneASR()).info()
    assert len(info.runs) == 3


def test_changed_file_gets_a_new_index(make_video, tmp_path: Path) -> None:
    path = tmp_path / "edit.mp4"
    write_media(path, place(10.0, [(2.0, tone(440, 1.0))]))
    first = make_video(path, ToneASR())
    first.index()
    write_media(path, place(10.0, [(2.0, tone(660, 1.0))]))
    backend = ToneASR()
    video = make_video(path, backend)
    assert video.fingerprint != first.fingerprint
    assert [s.text for s in video.transcribe()] == ["tone 660."]
    assert backend.calls == 1


def test_interrupted_run_resumes_without_duplicates(make_video, tone_video: Path) -> None:
    backend = ToneASR()
    stream = make_video(tone_video, backend).transcribe()
    first = next(stream)
    stream.close()  # stop after the first segment
    partial = make_video(tone_video, ToneASR()).transcript()
    assert partial.status == "partial"
    assert 1 <= len(partial) < 4
    assert partial.indexed_until is not None and partial.indexed_until < 90

    resumed_backend = ToneASR()
    resumed = list(make_video(tone_video, resumed_backend).transcribe())
    assert resumed[0] == first
    assert [s.text for s in resumed] == [t for t, _, _ in expected_segments()]
    assert len({s.id for s in resumed}) == 4
    assert (
        resumed_backend.audio_seconds < backend.audio_seconds + 20
    )  # only the rest was transcribed
    assert make_video(tone_video, ToneASR()).transcript().status == "complete"


def test_search_works_while_indexing(make_video, tone_video: Path) -> None:
    video = make_video(tone_video, ToneASR(texts=TEXTS))
    seen = []
    for segment in video.transcribe():
        results = make_video(tone_video, ToneASR()).search("deployment")
        seen.append((segment.text, len(results)))
    assert seen[0][1] >= 1, "the first committed segment is immediately searchable"
    assert seen[-1][1] == 2


def test_background_index_job(make_video, tone_video: Path) -> None:
    video = make_video(tone_video, ToneASR(texts=TEXTS))
    job = video.index(background=True)
    summary = job.wait(timeout=60)
    assert job.done and job.error is None
    assert summary.segments == 4


def test_concurrent_writer_is_rejected(make_video, tone_video: Path) -> None:
    stream = make_video(tone_video, ToneASR()).transcribe()
    next(stream)
    try:
        with pytest.raises(IndexLockedError):
            list(make_video(tone_video, ToneASR()).transcribe())
    finally:
        stream.close()


def test_force_retranscribes(make_video, tone_video: Path) -> None:
    make_video(tone_video, ToneASR()).index()
    backend = ToneASR()
    summary = make_video(tone_video, backend).index(force=True)
    assert backend.calls > 0 and summary.segments == 4 and not summary.cached


def test_language_detection_and_explicit_language(make_video, tone_video: Path) -> None:
    summary = make_video(tone_video, ToneASR(language="pt")).index()
    assert summary.language == "pt"
    assert summary.language_probability == pytest.approx(0.99)
    explicit = make_video(tone_video, ToneASR(language="pt"), language="de").index()
    assert explicit.language == "de"


def test_workers_produce_identical_results(make_video, tone_video: Path, tmp_path: Path) -> None:
    one = [(s.text, s.start, s.end) for s in make_video(tone_video, ToneASR()).transcribe()]
    many = make_video(tone_video, ToneASR(), workers=3, cache_dir=tmp_path / "other-cache")
    assert [(s.text, s.start, s.end) for s in many.transcribe()] == one


def test_progress_events(make_video, tone_video: Path) -> None:
    events: list[ProgressEvent] = []
    make_video(tone_video, ToneASR()).index(progress=events.append)
    stages = [e.stage for e in events]
    assert stages[0] is Stage.INSPECT and stages[-1] is Stage.COMPLETE
    assert Stage.TRANSCRIBE in stages and Stage.DETECT_LANGUAGE in stages
    fractions = [e.fraction for e in events if e.stage is Stage.TRANSCRIBE]
    assert all(f is not None and 0 <= f <= 1 for f in fractions)
    assert fractions == sorted(fractions)
    cached: list[ProgressEvent] = []
    make_video(tone_video, ToneASR()).index(progress=cached.append)
    assert Stage.CACHED in [e.stage for e in cached]


# -- edge-case media --------------------------------------------------------------------------


def test_very_short_video(make_video, tmp_path: Path) -> None:
    path = write_media(tmp_path / "blip.mp4", place(0.6, [(0.1, tone(500, 0.3))]))
    segments = list(make_video(path, ToneASR()).transcribe())
    assert [s.text for s in segments] == ["tone 500."]
    assert 0.0 <= segments[0].start < segments[0].end <= 0.65


def test_only_silence(make_video, tmp_path: Path) -> None:
    path = write_media(tmp_path / "quiet.mkv", silence(65.0))
    backend = ToneASR()
    video = make_video(path, backend)
    summary = video.index()
    assert summary.status == "empty" and summary.segments == 0
    assert backend.calls == 0
    assert video.search("anything") == []
    assert "No transcript passages matched" in video.context("anything").text


def test_video_without_audio(make_video, tmp_path: Path) -> None:
    path = write_media(tmp_path / "no audio.mp4", None, duration=3.0)
    video = make_video(path, ToneASR())
    summary = video.index()
    assert summary.status == "empty"
    with pytest.raises(AudioStreamNotFoundError):
        list(video.transcribe())
    assert video.search("hello") == []
    assert "TRANSCRIPT: none" in video.context("hello").text


def test_corrupted_input(make_video, tmp_path: Path) -> None:
    path = tmp_path / "corrupt.mov"
    path.write_bytes(b"\x00\x01garbage" * 5000)
    with pytest.raises(UnsupportedFormatError, match=r"corrupt\.mov"):
        make_video(path, ToneASR()).index()


def test_unicode_filename_and_spaces(make_video, tmp_path: Path) -> None:
    folder = tmp_path / "Reuniões de equipe" / "會議 記錄"
    path = write_media(folder / "Ação — résumé.mkv", place(6.0, [(1.0, tone(440, 1.0))]))
    texts = {440: "Configuração da autenticação concluída. 認証の設定が完了しました。"}
    video = make_video(path, ToneASR(texts=texts))
    video.index()
    results = video.search("configuracao")
    assert results and results[0].text == texts[440]
    assert video.search("認証")[0].text == texts[440]
    data = json.loads(video.transcript().to_json())
    assert data["segments"][0]["text"] == texts[440]


def test_not_indexed(make_video, tone_video: Path) -> None:
    video = make_video(tone_video, ToneASR())
    with pytest.raises(NotIndexedError, match=r"video\.index\(\)"):
        video.search("anything")


# -- search & context ---------------------------------------------------------------------------


@pytest.fixture
def indexed(make_video, tone_video: Path):
    video = make_video(tone_video, ToneASR(texts=TEXTS))
    video.index()
    return video


def test_search_ranks_and_filters(indexed) -> None:
    results = indexed.search("authentication configuration")
    assert results[0].text == TEXTS[880]
    assert results[0].start == pytest.approx(75.0, abs=0.05)
    assert results[0].score == 1.0
    assert all(0 < r.score <= 1 for r in results)
    assert set(results[0].matched_terms) == {"authentication", "configuration"}

    late = indexed.search("deployment", start=60)
    assert [r.text for r in late] == [TEXTS[1100]]
    early = indexed.search("deployment", end=60)
    assert [r.text for r in early] == [TEXTS[440]]


def test_search_expand_adds_neighbours(indexed) -> None:
    plain = indexed.search("lunch")[0]
    wide = indexed.search("lunch", expand=10)[0]
    assert len(wide.segment_ids) > len(plain.segment_ids)
    assert wide.start < plain.start


def test_context_is_traceable_and_chronological(indexed) -> None:
    context = indexed.context("Why did the deployment fail?", before=0, after=0)
    text = context.text
    assert text.startswith("VIDEO: tones.mp4")
    assert "TRANSCRIPT: complete" in text
    assert context.found
    transcript = {s.id: s for s in indexed.transcript().segments}
    for evidence in (e for e in context.evidence if e.type == "transcript"):
        source = transcript[evidence.id]
        assert (evidence.start, evidence.end, evidence.text) == (
            source.start,
            source.end,
            source.text,
        )
        assert evidence.id in text
    starts = [p.start for p in context.segments]
    assert starts == sorted(starts)
    assert "[0:02.0–0:05.0 | seg_00000]" in text


def test_context_expansion_includes_surrounding_speech(indexed) -> None:
    narrow = indexed.context("lunch", before=0, after=0)
    wide = indexed.context("lunch", before=10, after=0)
    assert {e.id for e in narrow.evidence if e.type == "transcript"} == {"seg_00001"}
    assert {e.id for e in wide.evidence if e.type == "transcript"} == {"seg_00000", "seg_00001"}


def test_context_respects_token_budget(indexed) -> None:
    for budget in (64, 90, 200, 4000):
        context = indexed.context("deployment authentication", max_tokens=budget)
        assert context.metadata.estimated_tokens <= budget
    tiny = indexed.context("deployment authentication", max_tokens=90)
    assert len(tiny.segments) <= 1
    counted = indexed.context("deployment", token_counter=lambda s: len(s.split()), max_tokens=70)
    assert len(counted.text.split()) <= 70


def test_context_json_and_partial_status(make_video, tone_video: Path) -> None:
    stream = make_video(tone_video, ToneASR(texts=TEXTS)).transcribe()
    next(stream)
    stream.close()
    context = make_video(tone_video, ToneASR()).context("deployment")
    assert "TRANSCRIPT: partial" in context.text
    data = json.loads(context.to_json())
    assert data["metadata"]["transcript_status"] == "partial"


def test_transcript_exports(indexed) -> None:
    transcript = indexed.transcript()
    srt = transcript.to_srt()
    assert re.match(r"1\n00:00:0[12],\d{3} --> 00:00:0[45],\d{3}\nThe deployment failed", srt)
    assert re.match(r"WEBVTT\n\n00:00:0[12]\.\d{3} --> ", transcript.to_vtt())
    assert len(transcript.chunks) >= 1
    chunk_segments = [sid for c in indexed.chunks() for sid in c.segment_ids]
    assert chunk_segments == [s.id for s in transcript.segments]


# -- async ----------------------------------------------------------------------------------------


def test_async_api(make_video, tone_video: Path) -> None:
    async def main():
        video = make_video(tone_video, ToneASR(texts=TEXTS))
        texts = [s.text async for s in video.atranscribe()]
        summary = await video.aindex()
        hits = await video.asearch("authentication")
        context = await video.acontext("authentication", max_tokens=500)
        return texts, summary, hits, context

    texts, summary, hits, context = asyncio.run(main())
    assert texts == [TEXTS[f] for _, f, _ in TONE_EVENTS]
    assert summary.segments == 4
    assert summary.frames > 0  # aindex also extracts frames
    assert hits and context.found


def test_async_transcribe_does_not_block_event_loop(make_video, tone_video: Path) -> None:
    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        task = asyncio.create_task(ticker())
        video = make_video(tone_video, ToneASR())
        segments = [s async for s in video.atranscribe()]
        task.cancel()
        return segments, ticks

    segments, ticks = asyncio.run(main())
    assert len(segments) == 4 and ticks > 5


def test_async_cancellation_keeps_progress(make_video, tone_video: Path) -> None:
    async def main():
        video = make_video(tone_video, ToneASR())
        async for _ in video.atranscribe():
            break

    asyncio.run(main())
    transcript = make_video(tone_video, ToneASR()).transcript()
    assert transcript.status in ("partial", "complete")
    assert len(transcript) >= 1


def test_words_are_mapped_when_enabled(make_video, tmp_path: Path) -> None:
    from saccade import ASRConfig
    from saccade.asr.base import ASRSegment, ASRWord

    class WordASR(ToneASR):
        def transcribe(self, audio, *, language=None, word_timestamps=False, prompt=None):
            for s in super().transcribe(audio, language=language):
                mid = (s.start + s.end) / 2
                yield ASRSegment(
                    s.start,
                    s.end,
                    s.text,
                    -0.1,
                    0.0,
                    (ASRWord(s.start, mid, "a"), ASRWord(mid, s.end, "b")),
                )

    path = write_media(tmp_path / "w.mp4", place(40.0, [(30.0, tone(440, 2.0))]))
    video = make_video(path, WordASR(), asr=ASRConfig(word_timestamps=True))
    segment = next(iter(video.transcribe()))
    assert [w.text for w in segment.words] == ["a", "b"]
    assert segment.words[0].start == pytest.approx(30.0, abs=0.05)
    assert segment.words[1].end == pytest.approx(32.0, abs=0.05)
    assert video.transcript().segments[0].words == segment.words


def test_confidence_is_model_reported(make_video, tone_video: Path) -> None:
    segment = next(iter(make_video(tone_video, ToneASR()).transcribe()))
    assert segment.confidence == pytest.approx(float(np.exp(-0.1)), abs=1e-4)

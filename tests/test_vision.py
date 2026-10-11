"""Representative frames: selection, deduplication, timestamps, storage, context."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from saccade import VisualConfig
from saccade.media.frames import sample_frames
from saccade.vision.hashing import dhash, hamming
from support import ToneASR, pattern, place, slides, tone, write_media, write_video


def test_ocr_config_is_optional_and_validated() -> None:
    from saccade import ConfigError

    assert VisualConfig().ocr is None
    assert VisualConfig(ocr=False).ocr is False
    assert VisualConfig(ocr=True, ocr_score=0.9).ocr_score == 0.9
    for score in (-0.1, 1.1, float("nan")):
        with pytest.raises(ConfigError, match="ocr_score"):
            VisualConfig(ocr_score=score)
    with pytest.raises(ConfigError, match="ocr must"):
        VisualConfig(ocr="yes")


def frame_pts(path: Path) -> set[float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {round(float(f.time), 3) for f in container.decode(stream)}


def test_ocr_adapter_is_lazy_and_filters_confidence(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from saccade import ConfigError
    from saccade.vision.ocr import FrameOCR, ocr_identity

    assert ocr_identity(VisualConfig()) is None
    with pytest.raises(ConfigError, match=r"saccade-video\[ocr\]"):
        ocr_identity(VisualConfig(ocr=True))
    monkeypatch.setattr("saccade.vision.ocr.installed_version", lambda: "test-version")
    assert ocr_identity(VisualConfig()) == {
        "engine": "rapidocr-onnxruntime",
        "version": "test-version",
    }
    assert ocr_identity(VisualConfig(ocr=False)) is None
    assert ocr_identity(VisualConfig(strategy="off", ocr=True)) is None
    created = []

    class Engine:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def __call__(self, jpeg):
            assert jpeg == b"jpeg"
            return [[[], " Pricing table ", 0.99], [[], "noise", 0.2]], None

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=Engine))
    reader = FrameOCR(0.5)
    assert not created
    assert reader(b"jpeg") == "Pricing table"
    assert reader(b"jpeg") == "Pricing table"
    assert len(created) == 1
    reader._engine = lambda jpeg: (None, None)
    assert reader(b"jpeg") == ""


def test_slides_are_kept_once_and_returning_slides_are_deduplicated(
    make_video, tmp_path: Path
) -> None:
    path = write_video(
        tmp_path / "deck.mp4", slides([(0, 10, 1), (10, 20, 2), (20, 30, 1), (30, 40, 3)])
    )
    video = make_video(path, ToneASR())
    video.index(visual_strategy="screen")
    frames = video.frames()
    assert [f.reason for f in frames] == ["start", "change", "change"]
    times = [f.timestamp for f in frames]
    assert times[0] == pytest.approx(0.0, abs=0.01)
    assert 10.0 <= times[1] <= 12.5, "kept shortly after the B transition, once settled"
    assert 30.0 <= times[2] <= 32.5, "A at 20 s repeats slide 1 and is skipped; C is new"


def test_static_screen_yields_a_single_frame(make_video, tmp_path: Path) -> None:
    path = write_video(tmp_path / "static.mp4", slides([(0, 150, 7)], fps=5))
    for strategy in ("auto", "interval", "screen"):
        video = make_video(path, ToneASR(), visual=VisualConfig(strategy=strategy))
        video.index()
        assert len(video.frames()) == 1, strategy


def test_camera_footage_gets_sparse_periodic_frames(make_video, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = pattern(5).astype(np.int16)
    frames = []
    for i in range(150 * 5):
        drift = int(4 * np.sin(i / 50))  # a person shifting slightly in a static shot
        noisy = np.clip(np.roll(base, drift, axis=1) + rng.integers(-12, 12, base.shape), 0, 255)
        frames.append((i / 5, noisy.astype(np.uint8)))
    path = write_video(tmp_path / "camera.mp4", frames)
    video = make_video(path, ToneASR())
    video.index(visual_strategy="auto")
    kept = video.frames()
    assert 2 <= len(kept) <= 8
    assert any(f.reason == "interval" for f in kept)


def test_scene_strategy_only_hard_cuts(make_video, tmp_path: Path) -> None:
    path = write_video(tmp_path / "cuts.mp4", slides([(0, 8, 11), (8, 16, 12), (16, 24, 13)]))
    video = make_video(path, ToneASR())
    video.index(visual_strategy="scenes")
    times = [f.timestamp for f in video.frames()]
    assert len(times) == 3
    assert times[1] == pytest.approx(8.0, abs=1.0)
    assert times[2] == pytest.approx(16.0, abs=1.0)


def test_variable_frame_rate_timestamps_come_from_pts(make_video, tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    times, t = [], 0.0
    while t < 30:
        times.append(round(t, 3))
        t += float(rng.choice([0.033, 0.1, 0.5, 1.7]))  # very irregular frame spacing
    frames = [(ts, pattern(1 if ts < 12.345 else 2)) for ts in times]
    path = write_video(tmp_path / "vfr.mkv", frames)
    real = frame_pts(path)
    video = make_video(path, ToneASR())
    video.index(visual_strategy="scenes")
    kept = video.frames()
    assert all(round(f.timestamp, 3) in real for f in kept), "every timestamp is a real frame's PTS"
    change = [f for f in kept if f.reason == "change"]
    assert len(change) == 1
    assert 12.345 <= change[0].timestamp <= 12.345 + 4.0


def test_sampling_rate_and_hashes(tmp_path: Path) -> None:
    path = write_video(tmp_path / "s.mp4", slides([(0, 10, 1), (10, 20, 2)], fps=25))
    samples = list(sample_frames(path, sample_fps=2.0))
    assert 35 <= len(samples) <= 42
    first, last = samples[0], samples[-1]
    assert hamming(dhash(first.hash_input), dhash(samples[1].hash_input)) <= 4
    assert hamming(dhash(first.hash_input), dhash(last.hash_input)) > 40


def test_stored_frames_are_jpegs_and_data_urls(make_video, tmp_path: Path) -> None:
    big = [
        (t / 10, np.kron(pattern(4), np.ones((10, 10, 1), np.uint8))) for t in range(30)
    ]  # 1600x960
    path = write_video(tmp_path / "big.mp4", big)
    video = make_video(path, ToneASR(), visual=VisualConfig(max_width=640))
    video.index()
    frame = video.frames()[0]
    data = Path(frame.path).read_bytes()
    assert data[:2] == b"\xff\xd8"
    assert frame.width == 640 and frame.height == 384
    assert frame.data_url().startswith("data:image/jpeg;base64,/9j/")
    assert Path(frame.path).is_relative_to(video.index_dir)


def test_frames_are_cached_and_keyed_by_strategy(make_video, tmp_path: Path) -> None:
    path = write_video(tmp_path / "deck.mp4", slides([(0, 5, 1), (5, 10, 2)]))
    first = make_video(path, ToneASR()).index()
    assert first.frames >= 2 and not first.cached
    again = make_video(path, ToneASR()).index()
    assert again.cached and again.frames == first.frames
    other = make_video(path, ToneASR()).index(visual_strategy="interval")
    assert not other.cached
    off = make_video(path, ToneASR(), visual=VisualConfig(strategy="off"))
    off.index()
    assert off.frames() == []


def test_audio_only_files_have_no_frames(make_video, tmp_path: Path) -> None:
    path = write_media(tmp_path / "voice.wav", place(5.0, [(1.0, tone(440, 1.0))]))
    video = make_video(path, ToneASR())
    summary = video.index()
    assert summary.segments == 1 and summary.frames == 0
    assert video.frames() == []


def text_slide(text="PRICING TABLE"):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 240), "white")
    ImageDraw.Draw(image).text((40, 80), text, fill="black", font=ImageFont.load_default(size=48))
    return [(index / 5, np.asarray(image)) for index in range(20)]


def test_ocr_text_is_stored_searched_and_cached(make_video, tmp_path, monkeypatch) -> None:
    from saccade.index.fts import build_match

    monkeypatch.setattr("saccade.vision.ocr.installed_version", lambda: "test")
    calls = []

    def recognize(self, jpeg):
        assert jpeg.startswith(b"\xff\xd8")
        calls.append(jpeg)
        return "Pricing table: configuração"

    monkeypatch.setattr("saccade.vision.ocr.FrameOCR.__call__", recognize)
    path = write_video(tmp_path / "ocr.mp4", text_slide())
    video = make_video(path)
    video.index()
    assert len(calls) == len(video.frames()) == 1
    frame = video.frames()[0]
    assert frame.ocr_text == "Pricing table: configuração"
    run = video._frames_run()
    assert run is not None
    match = build_match("configuracao")
    assert match
    found = video._db().search_frames(run.id, match.expression, start=None, end=None, limit=5)
    assert len(found) == 1 and found[0].timestamp == frame.timestamp
    assert not video._db().search_frames(run.id, match.expression, start=2, end=None, limit=5)
    results = video.search("pricing table")
    assert len(results) == 1
    assert results[0].source == "frame"
    assert results[0].frame_ids == (frame.id,)
    assert results[0].segment_ids == ()
    assert results[0].start == results[0].end == frame.timestamp
    assert results[0].to_dict()["frame_ids"] == [frame.id]
    assert not video.search("pricing", start=2)
    assert not video.search('" OR * -- nonexistent')
    context = video.context("pricing table", max_tokens=500)
    assert context.found
    assert "Text on screen at" in context.text
    assert "Pricing table" in context.text
    assert context.metadata.estimated_tokens <= 500
    assert context.frames[0].id == frame.id
    assert context.evidence[0].text == frame.ocr_text
    assert context.evidence[0].type == "frame"
    assert not context.segments
    for budget in (64, 90, 200, 4000):
        limited = video.context("pricing table", max_tokens=budget)
        assert limited.metadata.estimated_tokens <= budget
    counted = video.context("pricing table", max_tokens=500, token_counter=len)
    assert counted.found and len(counted.text) <= 500
    from saccade.ask import build_messages

    messages, evidence, images, mode = build_messages(
        video, "What is on the slide?", evidence_tokens=1000, max_images=0, images=False
    )
    assert mode == "full" and images == []
    assert "Text on screen at" in messages[1]["content"]
    assert "Pricing table" in messages[1]["content"]
    assert evidence[0].type == "frame" and evidence[0].text == frame.ocr_text
    assert make_video(path).index().cached
    assert len(calls) == 1
    changed = make_video(path, visual=VisualConfig(ocr_score=0.9))
    assert changed._visual_key() != video._visual_key()
    assert changed.frames() == []
    disabled = make_video(path, visual=VisualConfig(ocr=False))
    disabled.index()
    assert disabled.frames()[0].ocr_text == ""
    assert len(calls) == 1
    video._db().delete_run(run.id)
    with video._db().session() as conn:
        assert conn.execute("SELECT count(*) FROM frame_fts").fetchone()[0] == 0


def test_ocr_search_combines_speech_and_frame_evidence(make_video, tmp_path, monkeypatch) -> None:
    from saccade.ask import build_messages

    monkeypatch.setattr("saccade.vision.ocr.installed_version", lambda: "test")
    monkeypatch.setattr("saccade.vision.ocr.FrameOCR.__call__", lambda self, jpeg: "Pricing table")
    audio = place(4.0, [(1.0, tone(440, 1.0))])
    path = write_video(tmp_path / "mixed.mp4", text_slide(), audio=audio)
    video = make_video(path, ToneASR(texts={440: "Pricing table approved. " * 300}))
    video.index()
    assert {hit.source for hit in video.search("pricing table")} == {"transcript", "frame"}
    assert len(video.search("pricing table", limit=1)) == 1
    messages, evidence, _, mode = build_messages(
        video, "pricing table", evidence_tokens=500, max_images=0, images=False
    )
    assert mode == "retrieved"
    assert "Text on screen at" in messages[1]["content"]
    assert any(item.type == "frame" and item.text == "Pricing table" for item in evidence)


@pytest.mark.integration
def test_real_ocr_finds_rendered_slide(make_video, tmp_path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    path = write_video(tmp_path / "real-ocr.mp4", text_slide())
    video = make_video(path, visual=VisualConfig(ocr=True))
    video.index()
    results = video.search("pricing table")
    assert results and results[0].source == "frame"
    assert "pricing" in results[0].text.lower() and "table" in results[0].text.lower()
    assert results[0].start in frame_pts(path)


def test_silent_screen_recording_still_gets_frames(make_video, tmp_path: Path) -> None:
    path = write_video(tmp_path / "silent deck.mp4", slides([(0, 6, 1), (6, 12, 2)]))
    summary = make_video(path, ToneASR()).index()
    assert summary.segments == 0
    assert summary.frames == 2


def test_context_includes_frames_near_passages(make_video, tmp_path: Path) -> None:
    audio = place(40.0, [(12.0, tone(880, 3.0))])
    path = write_video(
        tmp_path / "talk.mp4", slides([(0, 10, 1), (10, 25, 2), (25, 40, 3)]), audio=audio
    )
    video = make_video(path, ToneASR(texts={880: "Here is the architecture diagram."}))
    video.index(visual_strategy="screen")
    context = video.context("architecture diagram", before=0, after=0)
    assert context.frames, "a frame showing slide 2 accompanies the passage at 12 s"
    assert all(10.0 <= f.timestamp <= 15.5 for f in context.frames)
    assert "Relevant visual evidence" in context.text
    assert context.frames[0].id in context.text
    assert any(
        e.type == "frame" and e.timestamp == context.frames[0].timestamp for e in context.evidence
    )
    window = video.frames(start=20, end=40)
    assert window and all(20 <= f.timestamp <= 40 for f in window)

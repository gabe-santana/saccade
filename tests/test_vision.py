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


def frame_pts(path: Path) -> set[float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {round(float(f.time), 3) for f in container.decode(stream)}


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

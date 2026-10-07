from __future__ import annotations

import itertools
import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from saccade import (
    AudioStreamNotFoundError,
    MediaDecodeError,
    MediaNotFoundError,
    UnsupportedFormatError,
)
from saccade.media import decode_audio, fingerprint, probe
from support import SR, decode_all, place, tone, write_media


@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ".mov", ".webm", ".wav", ".mp3"])
def test_common_containers_probe_and_decode(tmp_path: Path, suffix: str) -> None:
    audio = place(6.0, [(2.0, tone(440, 1.0))])
    path = write_media(tmp_path / f"clip{suffix}", audio)
    info = probe(path)
    assert info.has_audio
    assert info.duration == pytest.approx(6.0, abs=0.2)
    decoded = decode_all(path)
    assert len(decoded) / SR == pytest.approx(6.0, abs=0.15)
    # The burst must be where it was placed (codec delay is compensated by timestamps).
    loud = np.flatnonzero(np.abs(decoded) > 0.2) / SR
    assert loud[0] == pytest.approx(2.0, abs=0.06)
    assert loud[-1] == pytest.approx(3.0, abs=0.06)


def test_stereo_44k_is_downmixed_and_resampled(tmp_path: Path) -> None:
    audio = place(4.0, [(1.0, tone(1000, 1.0))])
    path = write_media(tmp_path / "stereo.mkv", audio, stereo=True)
    info = probe(path)
    assert info.audio_streams[0].channels == 2
    blocks = list(decode_audio(path))
    assert all(
        b.sample_rate == SR and b.samples.dtype == np.float32 and b.samples.ndim == 1
        for b in blocks
    )


def test_blocks_are_contiguous_and_bounded(tmp_path: Path) -> None:
    path = write_media(tmp_path / "long.mp4", place(23.0, [(5.0, tone(300, 2.0))]))
    blocks = list(decode_audio(path, block_seconds=5.0))
    for a, b in itertools.pairwise(blocks):
        assert b.start == pytest.approx(a.end, abs=1e-3)
    assert all(b.duration < 6.0 for b in blocks)


def test_decode_from_offset_keeps_absolute_time(tmp_path: Path) -> None:
    path = write_media(tmp_path / "seek.mp4", place(30.0, [(20.0, tone(500, 1.0))]))
    blocks = list(decode_audio(path, start=15.0))
    assert blocks[0].start == pytest.approx(15.0, abs=0.03)
    samples = np.concatenate([b.samples for b in blocks])
    loud = np.flatnonzero(np.abs(samples) > 0.2) / SR + blocks[0].start
    assert loud[0] == pytest.approx(20.0, abs=0.06)


def test_late_starting_audio_is_aligned_to_container_timeline(tmp_path: Path) -> None:
    """Audio whose first packet is at 1.5 s must not be shifted to 0."""
    path = tmp_path / "late.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=SR)
        stream.layout = "mono"
        pcm = (place(3.0, [(1.0, tone(700, 0.5))]) * 32767).astype(np.int16)[None, :]
        for i in range(0, pcm.shape[1], 1600):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(pcm[:, i : i + 1600]), format="s16", layout="mono"
            )
            frame.sample_rate = SR
            frame.pts = i + int(1.5 * SR)
            frame.time_base = Fraction(1, SR)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    blocks = list(decode_audio(path))
    samples = np.concatenate([b.samples for b in blocks])
    loud = np.flatnonzero(np.abs(samples) > 0.2) / SR + blocks[0].start
    # Matroska normalises the first timestamp to 0; either way the burst is 1.0 s after audio start.
    assert loud[0] - blocks[0].start == pytest.approx(1.0, abs=0.05)


def test_video_without_audio(tmp_path: Path) -> None:
    path = write_media(tmp_path / "silent-film.mp4", None, duration=2.0)
    info = probe(path)
    assert not info.has_audio and info.has_video
    with pytest.raises(AudioStreamNotFoundError, match="no audio stream"):
        list(decode_audio(path))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MediaNotFoundError):
        probe(tmp_path / "nope.mp4")
    with pytest.raises(MediaNotFoundError):
        probe(tmp_path)


def test_garbage_file_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(np.random.default_rng(0).bytes(50_000))
    with pytest.raises(UnsupportedFormatError) as info:
        probe(path)
    assert "broken.mp4" in str(info.value)
    assert "FFmpeg reported" in str(info.value)
    assert isinstance(info.value, MediaDecodeError)


def test_truncated_file_decodes_what_exists(tmp_path: Path) -> None:
    path = write_media(tmp_path / "full.mkv", place(10.0, [(1.0, tone(400, 1.0))]))
    data = path.read_bytes()
    cut = tmp_path / "cut.mkv"
    cut.write_bytes(data[: len(data) // 2])
    decoded = decode_all(cut)
    assert 1.5 < len(decoded) / SR < 10.0


def test_unicode_and_spaces_in_paths(tmp_path: Path) -> None:
    folder = tmp_path / "pasta com espaços" / "日本語 フォルダ"
    path = write_media(folder / "reunião — ünïcödé ✓.mp4", place(3.0, [(1.0, tone(440, 1.0))]))
    assert probe(path).has_audio
    assert len(decode_all(path)) > 0


def test_fingerprint_is_content_based(tmp_path: Path) -> None:
    a = write_media(tmp_path / "a.mp4", place(5.0, [(1.0, tone(440, 1.0))]))
    b = write_media(tmp_path / "b.mp4", place(5.0, [(1.0, tone(880, 1.0))]))
    copy = tmp_path / "copy of a.mp4"
    shutil.copy(a, copy)
    fa = fingerprint(a, probe(a))
    assert fa == fingerprint(a, probe(a)), "stable across calls"
    assert fa == fingerprint(copy, probe(copy)), "a copy (new mtime, new name) is the same media"
    assert fa != fingerprint(b, probe(b))
    assert fingerprint(a, probe(a), "strict").startswith("f1-")


def test_sampled_fingerprint_reads_little_but_strict_sees_everything(tmp_path: Path) -> None:
    from saccade.media.probe import MediaInfo

    path = tmp_path / "big.mp4"
    data = bytearray(np.random.default_rng(3).bytes(8_000_000))
    path.write_bytes(data)
    info = MediaInfo(
        path=str(path),
        format="mp4",
        size_bytes=len(data),
        duration=1.0,
        start_time=0.0,
        bit_rate=None,
    )
    sampled, strict = fingerprint(path, info), fingerprint(path, info, "strict")

    data[2_000_000] ^= 0xFF  # one byte between sampled windows: only strict mode notices
    path.write_bytes(data)
    assert fingerprint(path, info) == sampled
    assert fingerprint(path, info, "strict") != strict

    data[:16] = b"\1" * 16  # edits to the head (container headers) are always sampled
    path.write_bytes(data)
    assert fingerprint(path, info) != sampled

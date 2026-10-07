"""Batched (GPU-style) transcription and device selection."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import TONE_EVENTS
from saccade import ConfigError, Video
from saccade.utils.gpu import cuda_device_count, resolve_device
from support import ToneASR


class BatchedToneASR(ToneASR):
    """ToneASR that also implements the batched interface, recording batch sizes."""

    device = "cuda"

    def __init__(self, batch_size: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batches: list[int] = []

    def transcribe_batch(self, audios, *, language=None, word_timestamps=False):
        self.batches.append(len(audios))
        return [list(self.transcribe(a, language=language)) for a in audios]


def test_batched_backend_gives_identical_timeline(
    make_video, tone_video: Path, tmp_path: Path
) -> None:
    plain = [(s.text, s.start, s.end) for s in make_video(tone_video, ToneASR()).transcribe()]
    backend = BatchedToneASR(batch_size=3)
    batched = make_video(tone_video, backend, cache_dir=tmp_path / "batched")
    assert [(s.text, s.start, s.end) for s in batched.transcribe()] == plain
    assert backend.batches and max(backend.batches) <= 3
    assert sum(backend.batches) >= len(TONE_EVENTS) // 2


def test_gpu_backend_extracts_frames_concurrently(make_video, tone_video: Path) -> None:
    summary = make_video(tone_video, BatchedToneASR()).index()
    assert summary.segments == 4 and summary.frames > 0


def test_device_resolution() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
    with pytest.raises(ConfigError, match="Unknown device"):
        resolve_device("tpu")
    if cuda_device_count() == 0:
        with pytest.raises(ConfigError, match="no CUDA-capable"):
            resolve_device("cuda")


def test_video_device_option(tmp_path: Path) -> None:
    assert Video(tmp_path / "x.mp4", device="auto").asr_config.device == "auto"
    assert Video(tmp_path / "x.mp4").asr_config.device == "cpu"


@pytest.mark.integration
@pytest.mark.skipif(cuda_device_count() == 0, reason="no NVIDIA GPU")
def test_real_gpu_batched_transcription(tmp_path: Path) -> None:
    from support import silence, tts, tts_available, write_media

    if not tts_available("en"):
        pytest.skip("no TTS voice")
    speech = tts("The deployment failed because the authentication secret expired.", "en")
    audio = np.concatenate([silence(1.0), speech, silence(40.0), speech, silence(1.0)])
    path = write_media(tmp_path / "gpu.mp4", audio)
    from saccade.utils.cache import default_cache_dir, models_dir

    os.environ.setdefault("SACCADE_MODELS_DIR", str(models_dir(default_cache_dir())))
    video = Video(path, cache_dir=tmp_path / "cache", device="cuda")
    summary = video.index()
    assert summary.segments >= 2 and summary.language == "en"
    late = [s for s in video.transcript().segments if s.start > 20]
    assert late and late[0].start == pytest.approx(1.0 + len(speech) / 16000 + 40.0, abs=0.6)

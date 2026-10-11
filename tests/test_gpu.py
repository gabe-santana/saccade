"""Batched (GPU-style) transcription and device selection."""

from __future__ import annotations

import os
import sys
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


def test_whispercpp_adapter_and_offline_resolution(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from saccade import ASRConfig, ModelNotFoundError
    from saccade.asr.whispercpp import WhisperCppBackend, resolve_model

    with pytest.raises(ModelNotFoundError, match="downloads are disabled"):
        resolve_model("small", tmp_path, allow_download=False)
    path = tmp_path / "ggml-small-q5_1.bin"
    path.write_bytes(b"test-model")
    assert resolve_model("small-q5_1", tmp_path, allow_download=False) == str(path.resolve())
    calls = []

    class Model:
        def __init__(self, model_path, **kwargs):
            calls.append((model_path, kwargs))

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(t0=25, t1=175, text="hello")]

        def auto_detect_language(self, audio, **kwargs):
            return (("pt", 0.9), {"pt": 0.9})

    monkeypatch.setitem(sys.modules, "pywhispercpp.model", SimpleNamespace(Model=Model))
    backend = WhisperCppBackend(ASRConfig(model=str(path)), models_dir=tmp_path, threads=2)
    assert not calls
    audio = np.zeros(32000, dtype=np.float32)
    assert backend.detect_language(audio).language == "pt"
    segment = next(backend.transcribe(audio, language="pt", prompt="previous"))
    assert (segment.start, segment.end, segment.text) == (0.25, 1.75, "hello")
    list(backend.transcribe(audio))
    assert calls[0][1]["context_params"] == {"use_gpu": False}
    assert calls[1]["initial_prompt"] == "previous"
    assert calls[2]["initial_prompt"] == ""
    assert calls[2]["no_context"] is True
    assert backend.identity["backend"] == "whispercpp"
    assert "q5_1" in backend.identity["quantization"]


def test_whispercpp_device_selection(monkeypatch) -> None:
    from saccade.asr.whispercpp import resolve_device as resolve_cpp_device

    monkeypatch.setattr("saccade.asr.whispercpp.platform.system", lambda: "Darwin")
    monkeypatch.setattr("saccade.asr.whispercpp.platform.machine", lambda: "arm64")
    assert resolve_cpp_device("auto") == "metal"
    assert resolve_cpp_device("cpu") == "cpu"
    monkeypatch.setattr("saccade.asr.whispercpp.platform.system", lambda: "Windows")
    assert resolve_cpp_device("auto") == "cpu"
    with pytest.raises(ConfigError):
        resolve_cpp_device("metal")


def test_whispercpp_uses_pipeline_timeline(
    tone_video: Path, tmp_path: Path, energy_vad, monkeypatch
) -> None:
    from types import SimpleNamespace

    from saccade import ASRConfig
    from saccade.asr.whispercpp import WhisperCppBackend

    tone_backend = ToneASR()

    def transcribe(audio, **kwargs):
        return [
            SimpleNamespace(t0=segment.start * 100, t1=segment.end * 100, text=segment.text)
            for segment in tone_backend.transcribe(audio)
        ]

    model = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(WhisperCppBackend, "model", property(lambda self: model))
    video = Video(
        tone_video,
        cache_dir=tmp_path / "cpp",
        vad_factory=energy_vad,
        asr=ASRConfig(backend="whispercpp", language="en"),
    )
    segments = list(video.transcribe())
    assert len(segments) == len(TONE_EVENTS)
    for segment, (start, frequency, duration) in zip(segments, TONE_EVENTS, strict=True):
        assert segment.text == f"tone {frequency}."
        assert segment.start == pytest.approx(start, abs=0.05)
        assert segment.end == pytest.approx(start + duration, abs=0.05)
    cpp_key = video._cache_key(video._backend_for())[0]
    default = Video(tone_video, asr=ASRConfig(language="en"))
    assert cpp_key != default._cache_key(default._backend_for())[0]
    calls = tone_backend.calls
    assert list(video.transcribe()) == segments
    assert tone_backend.calls == calls


@pytest.mark.integration
def test_real_whispercpp_cpu(tmp_path: Path) -> None:
    pytest.importorskip("pywhispercpp")
    from saccade import ASRConfig
    from saccade.utils.cache import default_cache_dir, models_dir
    from support import silence, tts, tts_available, write_media

    if not tts_available("en"):
        pytest.skip("no English TTS voice")
    speech = tts("The deployment failed because the authentication secret expired.", "en")
    path = write_media(tmp_path / "cpp.mp4", np.concatenate([silence(1.0), speech, silence(1.0)]))
    os.environ.setdefault("SACCADE_MODELS_DIR", str(models_dir(default_cache_dir())))
    video = Video(
        path, cache_dir=tmp_path / "cache", asr=ASRConfig(backend="whispercpp", model="tiny.en")
    )
    segments = list(video.transcribe())
    assert segments and "deployment" in " ".join(segment.text for segment in segments).lower()
    assert segments[0].start == pytest.approx(1.0, abs=0.8)
    assert video.transcript().language == "en"


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

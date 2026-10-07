"""Silero VAD on onnxruntime (no PyTorch), run as a stream.

Uses the Silero v6 ONNX model that ships inside faster-whisper, so no download is needed.
The model is recurrent: its LSTM state and the 64-sample context are carried between
calls, which makes block-by-block processing produce exactly the same probabilities as
processing the whole file at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from saccade.config import VadConfig
from saccade.exceptions import ModelNotFoundError
from saccade.media.audio import SAMPLE_RATE, AudioBlock
from saccade.vad.base import SpeechProbabilityModel, SpeechRegion
from saccade.vad.segmenter import SpeechSegmenter

_WINDOW = 512
_CONTEXT = 64
_MODEL_FILE = "silero_vad_v6.onnx"


def silero_model_path() -> Path:
    try:
        from faster_whisper.utils import get_assets_path
    except ImportError as exc:  # pragma: no cover - faster-whisper is a hard dependency
        raise ModelNotFoundError("Silero VAD requires the faster-whisper package.") from exc
    path = Path(get_assets_path()) / _MODEL_FILE
    if not path.exists():
        raise ModelNotFoundError(
            f"Silero VAD model not found at {path}. Upgrade faster-whisper to 1.2 or newer."
        )
    return path


class SileroModel:
    """Streaming wrapper around the Silero v6 ONNX graph (single-threaded, ~1 ms/s of audio)."""

    window_samples = _WINDOW

    def __init__(self, path: Path | None = None) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(path or silero_model_path()),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def __call__(self, windows: NDArray[np.float32]) -> NDArray[np.float32]:
        if not len(windows):
            return np.zeros(0, dtype=np.float32)
        previous = np.concatenate([self._context[None, :], windows[:-1, -_CONTEXT:]], axis=0)
        batch = np.concatenate([previous, windows], axis=1)
        probs, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = windows[-1, -_CONTEXT:].copy()
        return np.asarray(probs, dtype=np.float32).reshape(-1)


class SileroVAD:
    """Streaming :class:`~saccade.vad.base.VoiceActivityDetector` backed by Silero."""

    def __init__(
        self,
        config: VadConfig | None = None,
        *,
        model: SpeechProbabilityModel | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.config = config or VadConfig()
        self.sample_rate = sample_rate
        self._model: SpeechProbabilityModel = model if model is not None else SileroModel()
        self._model.reset()
        window = self._model.window_samples
        self._segmenter = SpeechSegmenter(
            self.config, sample_rate=sample_rate, window_samples=window
        )
        self._origin: float | None = None
        self._leftover = np.zeros(0, dtype=np.float32)
        self._total = 0

    def feed(self, block: AudioBlock) -> list[SpeechRegion]:
        if self._origin is None:
            self._origin = block.start
        self._total += len(block.samples)
        window = self._model.window_samples
        data = (
            np.concatenate([self._leftover, block.samples])
            if len(self._leftover)
            else block.samples
        )
        usable = len(data) - len(data) % window
        self._leftover = data[usable:].copy()
        if not usable:
            return []
        probs = self._model(data[:usable].reshape(-1, window))
        return self._regions(self._segmenter.push(probs))

    def flush(self) -> list[SpeechRegion]:
        spans: list[tuple[int, int]] = []
        if len(self._leftover):
            window = self._model.window_samples
            padded = np.zeros(window, dtype=np.float32)
            padded[: len(self._leftover)] = self._leftover
            spans += self._segmenter.push(self._model(padded[None, :]))
            self._leftover = np.zeros(0, dtype=np.float32)
        spans += self._segmenter.finish(self._total)
        return self._regions(
            [(s, min(e, self._total)) for s, e in spans if min(e, self._total) > s]
        )

    @property
    def retain_from(self) -> float:
        if self._origin is None:
            return 0.0
        return self._origin + self._segmenter.retain_from / self.sample_rate

    def _regions(self, spans: list[tuple[int, int]]) -> list[SpeechRegion]:
        origin = self._origin or 0.0
        sr = self.sample_rate
        return [
            SpeechRegion(round(origin + s / sr, 3), round(origin + e / sr, 3)) for s, e in spans
        ]

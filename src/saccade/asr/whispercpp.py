"""Optional whisper.cpp backend with CPU and Apple Metal inference."""

from __future__ import annotations

import platform
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from saccade.asr.base import ASRSegment, AudioSource, LanguageDetection
from saccade.config import ASRConfig
from saccade.exceptions import ConfigError, ModelNotFoundError, TranscriptionError
from saccade.progress import Reporter, Stage


def resolve_model(
    name: str, models_dir: Path, *, allow_download: bool, reporter: Reporter | None = None
) -> str:
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    for filename in (name, f"ggml-{name}.bin", f"{name}.bin"):
        candidate = models_dir / filename
        if candidate.is_file():
            return str(candidate.resolve())
    if not allow_download:
        raise ModelNotFoundError(
            f"whisper.cpp model {name!r} is not in {models_dir} and downloads are disabled. "
            "Pass a local GGML model file or transcribe once while online."
        )
    try:
        from pywhispercpp.constants import AVAILABLE_MODELS
        from pywhispercpp.utils import download_model
    except ImportError as exc:
        raise ConfigError(
            'Install the optional backend: pip install "saccade-video[whispercpp]".'
        ) from exc
    if name not in AVAILABLE_MODELS:
        raise ModelNotFoundError(
            f"Unknown whisper.cpp model {name!r}; use a model name or GGML file."
        )
    if reporter is not None:
        reporter(Stage.DOWNLOAD_MODEL, f"Downloading whisper.cpp model '{name}' (one-time)")
    try:
        path = Path(download_model(name, download_dir=str(models_dir)))
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path.resolve())
    except Exception as exc:
        raise ModelNotFoundError(f"Could not download whisper.cpp model {name!r}: {exc}") from exc


def resolve_device(device: str) -> str:
    apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )
    if device == "auto":
        return "metal" if apple_silicon else "cpu"
    if device == "cpu" or (device == "metal" and apple_silicon):
        return device
    raise ConfigError("whisper.cpp supports cpu, auto, or metal on Apple Silicon.")


class WhisperCppBackend:
    """Lazily loaded pywhispercpp adapter; the pipeline owns pack-to-video mapping."""

    def __init__(
        self,
        config: ASRConfig,
        *,
        models_dir: Path,
        threads: int,
        allow_download: bool = True,
        reporter: Reporter | None = None,
    ) -> None:
        if config.compute_type != "auto":
            raise ConfigError(
                "whisper.cpp quantization is selected by the model file; use compute_type='auto'."
            )
        if config.word_timestamps:
            raise ConfigError("whisper.cpp does not yet expose word timestamps through Saccade.")
        self.config = config
        self.models_dir = models_dir / "whispercpp"
        self.threads = threads
        self.allow_download = allow_download
        self.reporter = reporter
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "whispercpp",
            "model": self.config.model,
            "quantization": self.config.model.rsplit("-", 1)[-1]
            if "-q" in self.config.model
            else "model-file",
            "device": self.device,
            "beam_size": self.config.beam_size,
        }

    @property
    def model(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from pywhispercpp.model import Model
                except ImportError as exc:
                    raise ConfigError(
                        'Install the optional backend: pip install "saccade-video[whispercpp]".'
                    ) from exc
                path = resolve_model(
                    self.config.model,
                    self.models_dir,
                    allow_download=self.allow_download,
                    reporter=self.reporter,
                )
                if self.reporter is not None:
                    self.reporter(Stage.LOAD_MODEL, f"Loading whisper.cpp model on {self.device}")
                try:
                    self._model = Model(
                        path,
                        n_threads=self.threads,
                        context_params={"use_gpu": self.device == "metal"},
                        params_sampling_strategy=0 if self.config.beam_size == 1 else 1,
                        beam_search={"beam_size": self.config.beam_size, "patience": -1.0},
                        print_progress=False,
                        print_realtime=False,
                        print_timestamps=False,
                        redirect_whispercpp_logs_to=None,
                    )
                except Exception as exc:
                    raise ModelNotFoundError(
                        f"Could not load whisper.cpp model {path}: {exc}"
                    ) from exc
            return self._model

    def detect_language(self, audio: AudioSource) -> LanguageDetection:
        with self._lock:
            model = self.model
            if ".en" in Path(self.config.model).name:
                return LanguageDetection("en", 1.0)
            try:
                (language, probability), _ = model.auto_detect_language(
                    audio, n_threads=self.threads
                )
                return LanguageDetection(str(language), float(probability))
            except Exception as exc:
                raise TranscriptionError(f"whisper.cpp language detection failed: {exc}") from exc

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> Iterator[ASRSegment]:
        if word_timestamps:
            raise ConfigError("whisper.cpp does not yet expose word timestamps through Saccade.")
        with self._lock:
            model = self.model
            try:
                segments = model.transcribe(
                    audio,
                    language=language or "auto",
                    initial_prompt=prompt or "",
                    no_context=True,
                    translate=False,
                    no_timestamps=False,
                )
                results = [
                    ASRSegment(float(segment.t0) / 100, float(segment.t1) / 100, segment.text)
                    for segment in segments
                ]
            except Exception as exc:
                raise TranscriptionError(f"whisper.cpp transcription failed: {exc}") from exc
        yield from results

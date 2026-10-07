"""faster-whisper (CTranslate2) backend — the default, CPU-first ASR engine.

Model weights are resolved from Saccade's cache directory *without touching the
network*. Only if a model is missing locally and downloads are allowed is it fetched
once from the Hugging Face Hub; pass ``allow_download=False`` (or set
``SACCADE_OFFLINE=1``) to forbid that entirely.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from saccade.asr.base import ASRSegment, ASRWord, AudioSource, LanguageDetection
from saccade.config import ASRConfig
from saccade.exceptions import ModelNotFoundError, TranscriptionError
from saccade.progress import Reporter, Stage
from saccade.utils.logs import get_logger, log_event

logger = get_logger(__name__)


def resolve_model(
    name: str, models_dir: Path, *, allow_download: bool, report: Reporter | None = None
) -> str:
    """Return a local directory containing the CTranslate2 model ``name``."""
    candidate = Path(name).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())

    from faster_whisper.utils import available_models, download_model

    if name not in available_models() and "/" not in name:
        known = ", ".join(available_models())
        raise ModelNotFoundError(
            f"Unknown Whisper model {name!r}. Use one of: {known}; "
            "a Hugging Face repo id such as 'Systran/faster-whisper-small'; or a local model directory."
        )
    try:
        return str(download_model(name, local_files_only=True, cache_dir=str(models_dir)))
    except Exception as missing:
        if not allow_download:
            raise ModelNotFoundError(
                f"Whisper model {name!r} is not in {models_dir} and downloads are disabled. "
                f"Run `saccade models download {name}` once while online, or pass a local model path."
            ) from missing
    if report is not None:
        report(Stage.DOWNLOAD_MODEL, f"Downloading Whisper model '{name}' (one-time)")
    log_event(logger, logging.INFO, "model.download", model=name, target=str(models_dir))
    try:
        return str(download_model(name, cache_dir=str(models_dir)))
    except Exception as exc:
        raise ModelNotFoundError(
            f"Could not download Whisper model {name!r}: {exc}. "
            "Check your connection, or pass a local model directory as ASRConfig(model=...)."
        ) from exc


class FasterWhisperBackend:
    """:class:`~saccade.asr.base.ASRBackend` implementation on faster-whisper.

    The model is loaded lazily, on the first call that needs it, so cached videos never
    pay the model load cost.
    """

    def __init__(
        self,
        config: ASRConfig,
        *,
        models_dir: Path,
        threads: int,
        workers: int = 1,
        allow_download: bool = True,
        reporter: Reporter | None = None,
    ) -> None:
        self.config = config
        self.models_dir = models_dir
        self.threads = threads
        self.workers = workers
        self.allow_download = allow_download
        self.reporter = reporter
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "faster-whisper",
            "model": self.config.model,
            "compute_type": self.config.compute_type,
            "device": self.config.device,
            "beam_size": self.config.beam_size,
        }

    @property
    def model(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def _load(self) -> Any:
        from faster_whisper import WhisperModel

        path = resolve_model(
            self.config.model,
            self.models_dir,
            allow_download=self.allow_download,
            report=self.reporter,
        )
        if self.reporter is not None:
            self.reporter(
                Stage.LOAD_MODEL,
                f"Loading Whisper model '{self.config.model}' ({self.config.compute_type})",
            )
        began = time.perf_counter()
        try:
            model = WhisperModel(
                path,
                device=self.config.device,
                compute_type=self.config.compute_type,
                cpu_threads=self.threads,
                num_workers=self.workers,
            )
        except Exception as exc:
            raise ModelNotFoundError(f"Could not load Whisper model from {path}: {exc}") from exc
        log_event(
            logger,
            logging.INFO,
            "model.loaded",
            model=self.config.model,
            compute_type=self.config.compute_type,
            threads=self.threads,
            seconds=time.perf_counter() - began,
        )
        return model

    def detect_language(self, audio: AudioSource) -> LanguageDetection:
        model = self.model
        if not model.model.is_multilingual:
            return LanguageDetection(language="en", probability=1.0)
        try:
            language, probability, _ = model.detect_language(audio)
        except ModelNotFoundError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Language detection failed: {exc}") from exc
        return LanguageDetection(language=language, probability=float(probability))

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> Iterator[ASRSegment]:
        model = self.model
        try:
            segments, _info = model.transcribe(
                audio,
                language=language,
                task="transcribe",
                beam_size=self.config.beam_size,
                vad_filter=False,  # Saccade has already removed non-speech
                word_timestamps=word_timestamps,
                initial_prompt=prompt,
                condition_on_previous_text=True,
                log_progress=False,
            )
            for seg in segments:
                words = tuple(
                    ASRWord(start=w.start, end=w.end, text=w.word, probability=w.probability)
                    for w in (seg.words or ())
                )
                yield ASRSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                    avg_logprob=float(seg.avg_logprob),
                    no_speech_prob=float(seg.no_speech_prob),
                    words=words,
                )
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

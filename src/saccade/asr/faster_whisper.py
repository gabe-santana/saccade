"""faster-whisper (CTranslate2) backend — the default, CPU-first ASR engine.

Model weights are resolved from Saccade's cache directory *without touching the
network*. Only if a model is missing locally and downloads are allowed is it fetched
once from the Hugging Face Hub; pass ``allow_download=False`` (or set
``SACCADE_OFFLINE=1``) to forbid that entirely.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from saccade.asr.base import ASRSegment, ASRWord, AudioSource, LanguageDetection
from saccade.config import ASRConfig
from saccade.exceptions import ModelNotFoundError, TranscriptionError
from saccade.media.audio import SAMPLE_RATE
from saccade.progress import Reporter, Stage
from saccade.utils.gpu import missing_library_hint, resolve_device
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
        self.device = resolve_device(config.device)
        self.compute_type = resolve_compute_type(config.compute_type, self.device)
        # Batching only pays off on GPU; on CPU one chunk at a time with internal
        # threading is faster and keeps time-to-first-result low.
        self.batch_size = config.batch_size if self.device == "cuda" else 1
        self._model: Any = None
        self._batched: Any = None
        self._lock = threading.Lock()

    @property
    def identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "backend": "faster-whisper",
            "model": self.config.model,
            "compute_type": self.compute_type,
            "device": self.device,
            "beam_size": self.config.beam_size,
        }
        if self.batch_size > 1:
            identity["batched"] = True  # batched decoding segments text slightly differently
        return identity

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
                f"Loading Whisper model '{self.config.model}' ({self.compute_type} on {self.device})",
            )
        began = time.perf_counter()
        try:
            model = WhisperModel(
                path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.threads,
                num_workers=self.workers,
            )
        except Exception as exc:
            hint = missing_library_hint(exc) if self.device == "cuda" else None
            raise ModelNotFoundError(
                hint or f"Could not load Whisper model from {path}: {exc}"
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "model.loaded",
            model=self.config.model,
            compute_type=self.compute_type,
            device=self.device,
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
            raise _asr_error(exc, self.device) from exc

    def transcribe_batch(
        self,
        audios: list[AudioSource],
        *,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> list[list[ASRSegment]]:
        """Transcribe several chunks (each at most 30 s) in one batched GPU pass.

        Returns one list of segments per input chunk, with times relative to that chunk.
        """
        from faster_whisper import BatchedInferencePipeline

        model = self.model
        with self._lock:
            if self._batched is None:
                self._batched = BatchedInferencePipeline(model)
            pipeline = self._batched
        clips: list[dict[str, float]] = []
        offsets: list[float] = []
        position = 0
        for audio in audios:
            start = position / SAMPLE_RATE
            position += len(audio)
            clips.append({"start": start, "end": position / SAMPLE_RATE})
            offsets.append(start)
        results: list[list[ASRSegment]] = [[] for _ in audios]
        try:
            segments, _info = pipeline.transcribe(
                np.concatenate(audios),
                language=language,
                beam_size=self.config.beam_size,
                vad_filter=False,
                clip_timestamps=clips,
                batch_size=self.batch_size,
                without_timestamps=False,
                word_timestamps=word_timestamps,
                log_progress=False,
            )
            for seg in segments:
                index = max(0, bisect.bisect_right(offsets, float(seg.start) + 1e-3) - 1)
                offset = offsets[index]
                words = tuple(
                    ASRWord(
                        start=w.start - offset,
                        end=w.end - offset,
                        text=w.word,
                        probability=w.probability,
                    )
                    for w in (seg.words or ())
                )
                results[index].append(
                    ASRSegment(
                        start=float(seg.start) - offset,
                        end=float(seg.end) - offset,
                        text=seg.text,
                        avg_logprob=float(seg.avg_logprob),
                        no_speech_prob=float(seg.no_speech_prob),
                        words=words,
                    )
                )
        except Exception as exc:
            raise _asr_error(exc, self.device) from exc
        return results


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _asr_error(exc: Exception, device: str) -> TranscriptionError:
    hint = missing_library_hint(exc) if device == "cuda" else None
    return TranscriptionError(hint or f"Whisper transcription failed: {exc}")

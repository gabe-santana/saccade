"""Bounded, resumable online speaker clustering; optional ONNX inference."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saccade.config import DiarizationConfig
from saccade.exceptions import ConfigError, ModelNotFoundError, TranscriptionError
from saccade.media.audio import SAMPLE_RATE, Samples


class SpeakerDiarizer:
    """Assign one speaker per ASR segment, keeping only bounded online centroids."""

    def __init__(self, config: DiarizationConfig, *, threads: int = 1) -> None:
        self.config = config
        self.threads = threads
        self._extractor: Any = None
        self._identity: dict[str, Any] | None = None
        self._centroids: list[NDArray[np.float32]] = []
        self._counts: list[int] = []

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            path = Path(self.config.model).expanduser()
            try:
                with path.open("rb") as model:
                    digest = hashlib.file_digest(model, "sha256").hexdigest()
            except OSError as exc:
                raise ModelNotFoundError(
                    f"Cannot read speaker embedding model {path}: {exc}"
                ) from exc
            self._identity = {"version": 1, **asdict(self.config), "model_sha256": digest}
        return self._identity

    def _embed(self, audio: Samples) -> NDArray[np.float32] | None:
        if self._extractor is None:
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise ConfigError(
                    'Install speaker support: pip install "saccade-video[speakers]".'
                ) from exc
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(Path(self.config.model).expanduser()),
                    num_threads=self.threads,
                    provider="cpu",
                )
            )
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)

    def label(self, audio: Samples) -> str | None:
        """Label the central bounded window, or leave short/empty speech unassigned."""
        if len(audio) < self.config.min_seconds * SAMPLE_RATE:
            return None
        size = min(len(audio), round(self.config.window_s * SAMPLE_RATE))
        start = (len(audio) - size) // 2
        try:
            vector = self._embed(audio[start : start + size])
        except (ConfigError, ModelNotFoundError):
            raise
        except Exception as exc:
            raise TranscriptionError(f"Speaker embedding failed: {exc}") from exc
        if vector is None:
            return None
        unit = self._unit(vector)
        if self._centroids and unit.shape != self._centroids[0].shape:
            raise TranscriptionError("Speaker embedding dimensions changed within a run.")
        scores = [float(np.dot(centroid, unit)) for centroid in self._centroids]
        best = int(np.argmax(scores)) if scores else 0
        if not scores or (
            scores[best] < self.config.threshold and len(scores) < self.config.max_speakers
        ):
            best = len(self._centroids)
            self._centroids.append(unit)
            self._counts.append(1)
        else:
            weight = min(self._counts[best], 1000)
            mean = self._centroids[best] * weight + unit
            if float(np.linalg.norm(mean)) > 0:
                self._centroids[best] = self._unit(mean)
            self._counts[best] = min(weight + 1, 1000)
        return f"SPEAKER_{best + 1}"

    @staticmethod
    def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
        norm = float(np.linalg.norm(vector))
        if (
            vector.ndim != 1
            or not vector.size
            or not np.isfinite(vector).all()
            or not 0 < norm < float("inf")
        ):
            raise TranscriptionError("Speaker embedding must be a finite, nonzero vector.")
        return np.asarray(vector / norm, dtype=np.float32)

    def snapshot(self) -> dict[str, Any]:
        """Return bounded JSON state for the pack's database transaction."""
        return {
            "centroids": [vector.tolist() for vector in self._centroids],
            "counts": self._counts.copy(),
        }

    def restore(self, state: dict[str, Any] | None) -> None:
        """Restore a committed state, or reset for a new run."""
        self._centroids = []
        self._counts = []
        if state is None:
            return
        vectors = state["centroids"]
        counts = state["counts"]
        if len(vectors) != len(counts) or len(vectors) > self.config.max_speakers:
            raise TranscriptionError("Invalid saved speaker state.")
        self._centroids = [self._unit(np.asarray(vector, dtype=np.float32)) for vector in vectors]
        if self._centroids and any(
            vector.shape != self._centroids[0].shape for vector in self._centroids
        ):
            raise TranscriptionError("Invalid saved speaker dimensions.")
        if any(not isinstance(count, int) or not 1 <= count <= 1000 for count in counts):
            raise TranscriptionError("Invalid saved speaker counts.")
        self._counts = list(counts)

"""Optional ONNX embeddings and bounded-memory cosine retrieval."""

from __future__ import annotations

import heapq
import importlib.util
import threading
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from saccade.exceptions import ConfigError
from saccade.index.chunking import join_texts
from saccade.index.database import Database, EmbeddingChunk
from saccade.models.result import SearchResult

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Encoder(Protocol):
    @property
    def key(self) -> str: ...

    def encode(self, texts: Sequence[str], *, query: bool = False) -> NDArray[np.float32]: ...


def available() -> bool:
    return importlib.util.find_spec("fastembed") is not None


class FastEmbedEncoder:
    def __init__(self, model: str, cache_dir: Path, *, offline: bool, threads: int) -> None:
        if not available():
            raise ConfigError('Semantic search requires pip install "saccade-video[embeddings]".')
        self.key = f"embedding-v1:{model}:fastembed-{version('fastembed')}"
        self._model_name = model
        self._cache_dir = cache_dir
        self._offline = offline
        self._threads = threads
        self._model: Any = None
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str], *, query: bool = False) -> NDArray[np.float32]:
        from fastembed import TextEmbedding

        with self._lock:
            if self._model is None:
                try:
                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=str(self._cache_dir),
                        local_files_only=self._offline,
                        threads=self._threads,
                    )
                except (ValueError, OSError) as exc:
                    raise ConfigError(f"Cannot load embedding model: {exc}") from exc
            embed = self._model.query_embed if query else self._model.passage_embed
            return np.asarray(list(embed(list(texts))), dtype=np.float32)


def _unit_vectors(vectors: NDArray[np.float32], count: int) -> NDArray[np.float32]:
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] == 0:
        raise ConfigError("Embedding backend returned an invalid vector shape.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ConfigError("Embedding backend returned a non-finite or zero vector.")
    return vectors / norms


def search_semantic(
    db: Database,
    run_id: int,
    query: str,
    encoder: Encoder,
    *,
    limit: int = 5,
    start: float | None = None,
    end: float | None = None,
    expand: float = 0.0,
) -> list[SearchResult]:
    if limit < 1 or expand < 0:
        raise ConfigError("limit must be at least 1 and expand must be >= 0.")
    if not query.strip():
        return []
    query_vector = _unit_vectors(encoder.encode([query], query=True), 1)[0]
    best: list[tuple[float, int, EmbeddingChunk]] = []
    for batch in db.embedding_chunks(run_id, encoder.key, start=start, end=end):
        missing = [chunk for chunk in batch if chunk.vector is None]
        fresh: dict[int, bytes] = {}
        if missing:
            vectors = _unit_vectors(encoder.encode([chunk.text for chunk in missing]), len(missing))
            if vectors.shape[1] != query_vector.size:
                raise ConfigError("Query and passage embedding dimensions differ.")
            fresh = {
                chunk.id: vector.astype("<f2").tobytes()
                for chunk, vector in zip(missing, vectors, strict=True)
            }
            db.store_embeddings(encoder.key, list(fresh.items()))
        stored = [chunk.vector if chunk.vector is not None else fresh[chunk.id] for chunk in batch]
        if any(len(vector) != query_vector.size * 2 for vector in stored):
            raise ConfigError("Cached embedding dimensions differ from the query model.")
        matrix = _unit_vectors(
            np.asarray([np.frombuffer(vector, dtype="<f2") for vector in stored], dtype=np.float32),
            len(batch),
        )
        scores = matrix @ query_vector
        for chunk, score in zip(batch, scores, strict=True):
            candidate = (float(score), -chunk.id, chunk)
            if len(best) < limit:
                heapq.heappush(best, candidate)
            elif candidate[:2] > best[0][:2]:
                heapq.heapreplace(best, candidate)

    results = []
    ordered = sorted(best, key=lambda item: item[:2], reverse=True)
    top = max((ordered[0][0] + 1) / 2, 0.0001) if ordered else 1.0
    for score, _, chunk in ordered:
        rows = db.segments_between(run_id, chunk.first_segment, chunk.last_segment)
        if not rows:
            continue
        if expand:
            rows = db.segments(
                run_id,
                start=rows[0][1].start - expand,
                end=max(seg.end for _, seg in rows) + expand,
            )
        segments = [segment for _, segment in rows]
        results.append(
            SearchResult(
                start=segments[0].start,
                end=max(segment.end for segment in segments),
                text=join_texts(segment.text for segment in segments),
                score=min(1.0, max(0.0001, (score + 1) / 2 / top)),
                segment_ids=tuple(segment.id for segment in segments),
            )
        )
    return results

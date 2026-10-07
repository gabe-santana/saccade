"""RAG-ready context assembled for a query."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from saccade.models.frame import Frame
from saccade.models.transcript import TranscriptChunk, TranscriptStatus


@dataclass(frozen=True, slots=True)
class Evidence:
    """A pointer from returned context back to source data in the video.

    Transcript evidence has ``start``/``end``; frame evidence has ``timestamp``.
    """

    id: str
    type: Literal["transcript", "frame"]
    start: float | None = None
    end: float | None = None
    timestamp: float | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True, slots=True)
class ContextMetadata:
    video: str
    duration: float | None
    language: str | None
    transcript_status: TranscriptStatus
    indexed_until: float | None
    max_tokens: int
    estimated_tokens: int
    passages: int
    hits: int
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VideoContext:
    """Evidence retrieved for ``query``, ready to paste into any LLM prompt.

    ``text`` contains only retrieved source material with timestamps; Saccade never
    generates claims of its own. ``segments`` are the passages included (each traceable
    through ``segment_ids``) and ``evidence`` lists every source segment used.
    """

    query: str
    text: str
    segments: list[TranscriptChunk]
    frames: list[Frame]
    evidence: list[Evidence]
    metadata: ContextMetadata
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "frames": [f.to_dict() for f in self.frames],
            "evidence": [e.to_dict() for e in self.evidence],
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def __str__(self) -> str:
        return self.text

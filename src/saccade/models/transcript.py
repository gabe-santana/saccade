"""Whole transcripts and retrieval chunks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from saccade.models.segment import TranscriptSegment
from saccade.utils.time import format_span, format_srt_time, format_vtt_time

TranscriptStatus = Literal["complete", "partial", "running", "failed", "empty"]


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """A run of consecutive segments (typically 20–60 s) used as a retrieval unit.

    ``segment_ids`` lists the source segments, so every chunk stays traceable to the video.
    """

    id: str
    start: float
    end: float
    text: str
    segment_ids: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segment_ids"] = list(self.segment_ids)
        return data

    def __str__(self) -> str:
        return f"[{format_span(self.start, self.end)} | {self.id}] {self.text}"


@dataclass(frozen=True, slots=True)
class Transcript:
    """The stored transcript of a video.

    ``status`` is ``"partial"`` (or ``"running"``) while indexing is still in progress;
    ``indexed_until`` then says how far into the video the transcript reaches.
    """

    segments: tuple[TranscriptSegment, ...]
    language: str | None
    status: TranscriptStatus
    indexed_until: float | None = None
    duration: float | None = None
    model: str | None = None
    language_probability: float | None = None
    chunks: tuple[TranscriptChunk, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[TranscriptSegment]:
        return iter(self.segments)

    def to_srt(self) -> str:
        blocks = [
            f"{i}\n{format_srt_time(s.start)} --> {format_srt_time(s.end)}\n{s.text}\n"
            for i, s in enumerate(self.segments, start=1)
        ]
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        cues = [
            f"{format_vtt_time(s.start)} --> {format_vtt_time(s.end)}\n{s.text}\n"
            for s in self.segments
        ]
        return "WEBVTT\n\n" + "\n".join(cues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "status": self.status,
            "indexed_until": self.indexed_until,
            "duration": self.duration,
            "model": self.model,
            "segments": [s.to_dict() for s in self.segments],
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def __str__(self) -> str:
        return "\n".join(f"[{format_span(s.start, s.end)}] {s.text}" for s in self.segments)

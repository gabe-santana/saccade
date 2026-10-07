"""Transcript segments and words."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from saccade.utils.time import format_span


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """A word with its own timing. Only produced when ``ASRConfig.word_timestamps`` is on."""

    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One ASR segment on the source video's timeline.

    Attributes:
        id: Stable evidence identifier, e.g. ``seg_00042``.
        start: Start time in seconds on the media timeline.
        end: End time in seconds on the media timeline.
        text: Recognised text.
        language: Language the segment was decoded as (ISO-639-1).
        confidence: ``exp(avg_logprob)`` — the geometric-mean token probability reported
            by Whisper. Useful for spotting doubtful passages; it is *not* a calibrated
            probability that the text is correct.
        no_speech_prob: Whisper's probability that the window contained no speech.
        words: Word timings, empty unless word timestamps were requested.
    """

    id: str
    start: float
    end: float
    text: str
    language: str | None = None
    confidence: float | None = None
    no_speech_prob: float | None = None
    words: tuple[TranscriptWord, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not self.words:
            data.pop("words")
        return data

    def __str__(self) -> str:
        return f"[{format_span(self.start, self.end)}] {self.text}"


def segment_id(index: int) -> str:
    return f"seg_{index:05d}"


def chunk_id(index: int) -> str:
    return f"chunk_{index:05d}"

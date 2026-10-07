"""Typed progress events.

Progress is reported as *media position* (how far into the video processing has got)
rather than an invented percentage. ``fraction`` is only available when the container
reports a duration, and it measures timeline coverage, not remaining wall-clock time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from saccade.utils.time import format_clock


class Stage(StrEnum):
    INSPECT = "inspect"
    CACHED = "cached"
    RESUME = "resume"
    DOWNLOAD_MODEL = "download_model"
    LOAD_MODEL = "load_model"
    DETECT_LANGUAGE = "detect_language"
    DETECT_SPEECH = "detect_speech"
    TRANSCRIBE = "transcribe"
    INDEX = "index"
    FRAMES = "frames"
    EXPLORE = "explore"
    COMPLETE = "complete"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: Stage
    message: str
    position: float | None = None
    duration: float | None = None
    start: float | None = None
    end: float | None = None

    @property
    def fraction(self) -> float | None:
        if self.position is None or not self.duration:
            return None
        return max(0.0, min(1.0, self.position / self.duration))

    def __str__(self) -> str:
        fraction = self.fraction
        if fraction is None or self.stage is Stage.COMPLETE:
            return self.message
        return f"{self.message} ({fraction:.0%} of timeline)"


ProgressCallback = Callable[[ProgressEvent], None]


def span_message(verb: str, start: float, end: float) -> str:
    return f"{verb} {format_clock(start)}–{format_clock(end)}"


class Reporter:
    """Null-safe wrapper so pipeline code can always call ``report``."""

    def __init__(self, callback: ProgressCallback | None, duration: float | None = None) -> None:
        self._callback = callback
        self.duration = duration

    def __call__(
        self,
        stage: Stage,
        message: str,
        *,
        position: float | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        if self._callback is None:
            return
        self._callback(
            ProgressEvent(
                stage, message, position=position, duration=self.duration, start=start, end=end
            )
        )

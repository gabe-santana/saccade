"""Detector used when VAD is disabled: the whole timeline, cut into fixed windows."""

from __future__ import annotations

from saccade.media.audio import AudioBlock
from saccade.vad.base import SpeechRegion

_MIN_TAIL_S = 0.2


class FixedWindowVAD:
    """Treats all audio as speech, emitting consecutive ``window_s``-second regions."""

    def __init__(self, window_s: float = 28.0) -> None:
        self.window_s = window_s
        self._next_start: float | None = None
        self._position = 0.0

    def feed(self, block: AudioBlock) -> list[SpeechRegion]:
        if self._next_start is None:
            self._next_start = block.start
        self._position = block.end
        regions = []
        while self._position - self._next_start >= self.window_s:
            end = self._next_start + self.window_s
            regions.append(SpeechRegion(round(self._next_start, 3), round(end, 3)))
            self._next_start = end
        return regions

    def flush(self) -> list[SpeechRegion]:
        if self._next_start is None or self._position - self._next_start < _MIN_TAIL_S:
            return []
        region = SpeechRegion(round(self._next_start, 3), round(self._position, 3))
        self._next_start = self._position
        return [region]

    @property
    def retain_from(self) -> float:
        return self._next_start or 0.0

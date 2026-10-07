"""VAD interfaces.

A :class:`VoiceActivityDetector` consumes contiguous audio blocks and emits speech
regions on the absolute media timeline as soon as they are final, which is what lets
transcription start long before the whole file has been decoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from saccade.media.audio import AudioBlock


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """A span of the media timeline (seconds) that contains speech, including padding."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Streaming speech detector.

    ``feed`` is called with consecutive blocks; it returns regions that can no longer
    change. ``flush`` is called once at end of stream. ``retain_from`` tells the caller
    the earliest media time any *future* region could start at, so older audio can be freed.
    """

    def feed(self, block: AudioBlock) -> list[SpeechRegion]: ...

    def flush(self) -> list[SpeechRegion]: ...

    @property
    def retain_from(self) -> float: ...


class SpeechProbabilityModel(Protocol):
    """Frame-level speech classifier used by :class:`~saccade.vad.silero.SileroVAD`.

    Called with consecutive windows of shape ``(n, window_samples)``; returns ``n``
    probabilities. Implementations may keep recurrent state between calls.
    """

    window_samples: int

    def __call__(self, windows: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def reset(self) -> None: ...

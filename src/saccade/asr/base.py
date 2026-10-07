"""ASR backend interface.

Backends receive 16 kHz mono float32 audio and return segments with times *relative to
that audio*. The pipeline is responsible for mapping them back onto the video timeline,
so backends never need to know about VAD or packing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

AudioSource = NDArray[np.float32]
"""16 kHz mono float32 PCM in [-1, 1]."""


@dataclass(frozen=True, slots=True)
class ASRWord:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class ASRSegment:
    """A recognised segment with times relative to the audio that was passed in."""

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: tuple[ASRWord, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    language: str
    probability: float


@runtime_checkable
class ASRBackend(Protocol):
    """Anything that can transcribe audio. Implement this to plug in another engine."""

    @property
    def identity(self) -> dict[str, Any]:
        """Settings that change the output; part of the transcript cache key."""
        ...

    def detect_language(self, audio: AudioSource) -> LanguageDetection: ...

    def transcribe(
        self,
        audio: AudioSource,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> Iterator[ASRSegment]: ...

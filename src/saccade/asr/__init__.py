"""Automatic speech recognition backends."""

from saccade.asr.base import ASRBackend, ASRSegment, ASRWord, AudioSource, LanguageDetection
from saccade.asr.faster_whisper import FasterWhisperBackend, resolve_model

__all__ = [
    "ASRBackend",
    "ASRSegment",
    "ASRWord",
    "AudioSource",
    "FasterWhisperBackend",
    "LanguageDetection",
    "resolve_model",
]

"""Voice activity detection."""

from __future__ import annotations

from saccade.config import VadConfig
from saccade.vad.base import SpeechProbabilityModel, SpeechRegion, VoiceActivityDetector
from saccade.vad.passthrough import FixedWindowVAD
from saccade.vad.silero import SileroModel, SileroVAD


def create_vad(config: VadConfig) -> VoiceActivityDetector:
    """The detector implied by ``config`` (Silero, or fixed windows when disabled)."""
    if not config.enabled:
        return FixedWindowVAD(window_s=config.max_region_s)
    return SileroVAD(config)


__all__ = [
    "FixedWindowVAD",
    "SileroModel",
    "SileroVAD",
    "SpeechProbabilityModel",
    "SpeechRegion",
    "VoiceActivityDetector",
    "create_vad",
]

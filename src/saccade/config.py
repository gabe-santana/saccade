"""Typed configuration objects and performance profiles.

All configuration is immutable. Use :func:`dataclasses.replace` to derive variants::

    from dataclasses import replace
    asr = replace(ASRConfig(), model="medium", beam_size=5)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, get_args

from saccade.exceptions import ConfigError

Profile = Literal["fast", "balanced", "accurate"]
FingerprintMode = Literal["sampled", "strict"]
VisualStrategy = Literal["auto", "scenes", "interval", "screen", "off"]

PROFILES: dict[str, dict[str, Any]] = {
    "fast": {"model": "base", "compute_type": "int8", "beam_size": 1},
    "balanced": {"model": "small", "compute_type": "int8", "beam_size": 1},
    "accurate": {"model": "medium", "compute_type": "int8", "beam_size": 5},
}
"""Model presets used by ``profile=``. They only touch model, quantization and beam size."""


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """Speech recognition settings.

    Attributes:
        model: A faster-whisper model name (``"small"``, ``"large-v3"``, ...) or a path to a
            local CTranslate2 model directory. Multilingual models are the default; English-only
            ``*.en`` models work but cannot transcribe other languages.
        compute_type: CTranslate2 quantization. ``int8`` is the fastest on CPU.
        device: ``"cpu"`` (default) or ``"cuda"``/``"auto"`` if CTranslate2 was built with GPU support.
        beam_size: 1 is greedy decoding (fastest); 5 is Whisper's accuracy default.
        language: ISO-639-1 code such as ``"pt"``, or ``None`` to detect it automatically.
        word_timestamps: Also store per-word timings. Off by default: it costs extra decoding time.
        carry_context: Prompt each 30-second window with the end of the previous one. Improves
            continuity of names and punctuation; ignored when ``workers > 1``.
    """

    model: str = "small"
    compute_type: str = "int8"
    device: str = "cpu"
    beam_size: int = 1
    language: str | None = None
    word_timestamps: bool = False
    carry_context: bool = True

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("ASRConfig.model must be a model name or a path.")
        if self.beam_size < 1:
            raise ConfigError(f"ASRConfig.beam_size must be >= 1, got {self.beam_size}.")
        if self.language is not None:
            lang = self.language.strip().lower()
            object.__setattr__(self, "language", None if lang in ("", "auto") else lang)

    def with_profile(self, profile: Profile | str) -> ASRConfig:
        """Return a copy with the model, quantization and beam size of ``profile``."""
        if profile not in PROFILES:
            choices = ", ".join(PROFILES)
            raise ConfigError(f"Unknown profile {profile!r}. Choose one of: {choices}.")
        return replace(self, **PROFILES[profile])


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Voice activity detection settings (Silero VAD).

    Attributes:
        enabled: Skip non-speech audio before ASR. Disabling it sends every second of audio
            through Whisper, which is slower and invites hallucinations on silence.
        threshold: Speech probability above which a 32 ms window counts as speech.
        min_speech_ms: Speech regions shorter than this are dropped as false positives.
        min_silence_ms: Silence must last this long before a speech region is closed.
        padding_ms: Audio kept before and after each region so word edges are not clipped.
        merge_gap_ms: Padded regions separated by less than this are merged into one.
        max_region_s: Regions longer than this are split at the quietest recent pause.
    """

    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    padding_ms: int = 200
    merge_gap_ms: int = 350
    max_region_s: float = 28.0

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ConfigError(f"VadConfig.threshold must be in (0, 1), got {self.threshold}.")
        for name in ("min_speech_ms", "min_silence_ms", "padding_ms", "merge_gap_ms"):
            if getattr(self, name) < 0:
                raise ConfigError(f"VadConfig.{name} must be >= 0.")
        if self.max_region_s < 2.0:
            raise ConfigError("VadConfig.max_region_s must be at least 2 seconds.")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """How transcript segments are grouped into retrieval chunks.

    A chunk closes at the first sentence boundary after ``min_seconds``, and is forced
    closed before it would exceed ``max_seconds``. A pause longer than ``break_on_pause_s``
    also ends a chunk once it has reached half of ``min_seconds``.
    """

    min_seconds: float = 20.0
    max_seconds: float = 60.0
    break_on_pause_s: float = 8.0

    def __post_init__(self) -> None:
        if not 0 < self.min_seconds <= self.max_seconds:
            raise ConfigError("ChunkConfig requires 0 < min_seconds <= max_seconds.")


@dataclass(frozen=True, slots=True)
class VisualConfig:
    """How representative frames are chosen. Frames are never sent through a vision model.

    Strategies:
        ``auto``: adapts per video — on static content (screen recordings, slides) every
            settled visual change is kept; on camera footage, scene changes plus one frame
            per ``interval_s`` while the picture keeps changing.
        ``screen``: sensitive to small UI changes; waits for transitions to settle.
        ``scenes``: only clear scene cuts.
        ``interval``: one frame every ``interval_s`` seconds (near-duplicates skipped).
        ``off``: no frames.

    Attributes:
        sample_fps: How many decoded frames per second are examined (cheap thumbnails).
        interval_s: Spacing for ``interval`` (and the periodic fallback of ``auto``).
        max_width: Stored frames are downscaled to at most this width (JPEG).
        max_frames: Hard cap per video.
    """

    strategy: VisualStrategy = "auto"
    sample_fps: float = 1.0
    interval_s: float = 60.0
    max_width: int = 1280
    jpeg_quality: int = 3  # FFmpeg qscale: 2 (best) .. 31 (worst)
    max_frames: int = 1000

    def __post_init__(self) -> None:
        if self.strategy not in get_args(VisualStrategy):
            choices = ", ".join(get_args(VisualStrategy))
            raise ConfigError(
                f"Unknown visual strategy {self.strategy!r}. Choose one of: {choices}."
            )
        if not 0 < self.sample_fps <= 30:
            raise ConfigError("VisualConfig.sample_fps must be in (0, 30].")
        if self.interval_s <= 0 or self.max_width < 64 or self.max_frames < 1:
            raise ConfigError(
                "VisualConfig needs interval_s > 0, max_width >= 64, max_frames >= 1."
            )
        if not 2 <= self.jpeg_quality <= 31:
            raise ConfigError("VisualConfig.jpeg_quality must be between 2 and 31.")


def config_fingerprint(*parts: object) -> dict[str, Any]:
    """Flatten configuration objects into a JSON-serialisable dict for cache keys."""
    merged: dict[str, Any] = {}
    for part in parts:
        if hasattr(part, "__dataclass_fields__"):
            merged[type(part).__name__] = asdict(part)  # type: ignore[call-overload]
        elif isinstance(part, dict):
            merged.update(part)
        else:
            raise TypeError(f"Cannot fingerprint {type(part).__name__}")
    return merged


def validate_profile(profile: str | None) -> Profile | None:
    if profile is None:
        return None
    if profile not in get_args(Profile):
        choices = ", ".join(PROFILES)
        raise ConfigError(f"Unknown profile {profile!r}. Choose one of: {choices}.")
    return profile  # type: ignore[return-value]

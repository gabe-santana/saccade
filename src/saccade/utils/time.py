"""Timestamp formatting helpers."""

from __future__ import annotations

import math


def format_timestamp(seconds: float, *, precision: int = 1) -> str:
    """Compact human/LLM-friendly timestamp: ``12:22.1`` or ``1:02:22.1``."""
    seconds = max(0.0, seconds)
    scale = 10**precision
    total = round(seconds * scale)
    hours, rem = divmod(total, 3600 * scale)
    minutes, rem = divmod(rem, 60 * scale)
    secs, frac = divmod(rem, scale)
    tail = f"{secs:02d}" + (f".{frac:0{precision}d}" if precision else "")
    if hours:
        return f"{hours}:{minutes:02d}:{tail}"
    return f"{minutes}:{tail}"


def format_clock(seconds: float) -> str:
    """Fixed-width ``HH:MM:SS`` used in progress messages."""
    total = max(0, math.floor(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _hms_ms(seconds: float) -> tuple[int, int, int, int]:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return hours, minutes, secs, ms


def format_srt_time(seconds: float) -> str:
    h, m, s, ms = _hms_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    h, m, s, ms = _hms_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_span(start: float, end: float, *, precision: int = 1) -> str:
    return f"{format_timestamp(start, precision=precision)}–{format_timestamp(end, precision=precision)}"

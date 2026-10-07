from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from saccade import VadConfig, Video
from saccade.vad.silero import SileroVAD
from support import EnergyModel, ToneASR, place, tone, write_media


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def energy_vad() -> Callable[[], SileroVAD]:
    return lambda: SileroVAD(VadConfig(), model=EnergyModel())


@pytest.fixture
def make_video(cache_dir: Path, energy_vad: Callable[[], SileroVAD]) -> Callable[..., Video]:
    """Build a ``Video`` wired to deterministic test doubles (energy VAD + ToneASR)."""

    def factory(path: Path, backend: ToneASR | None = None, **kwargs: object) -> Video:
        kwargs.setdefault("cache_dir", cache_dir)
        kwargs.setdefault("vad_factory", energy_vad)
        return Video(path, asr_backend=backend or ToneASR(), **kwargs)  # type: ignore[arg-type]

    return factory


TONE_EVENTS = [(2.0, 440, 3.0), (9.0, 660, 2.0), (75.0, 880, 4.0), (81.0, 1100, 2.5)]
"""(start seconds, frequency, duration) — includes a ~60 s silent gap."""


@pytest.fixture
def tone_video(tmp_path: Path) -> Path:
    audio = place(90.0, [(start, tone(freq, dur)) for start, freq, dur in TONE_EVENTS])
    return write_media(tmp_path / "media" / "tones.mp4", audio)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("SACCADE_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set SACCADE_INTEGRATION=1 to run real-model tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

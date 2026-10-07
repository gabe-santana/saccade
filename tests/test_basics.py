from __future__ import annotations

import pytest

from saccade import ASRConfig, ChunkConfig, ConfigError, VadConfig
from saccade.config import PROFILES
from saccade.utils.time import format_clock, format_srt_time, format_timestamp, format_vtt_time


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00.0"), (742.24, "12:22.2"), (59.96, "1:00.0"), (3742.5, "1:02:22.5"), (-1, "0:00.0")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_subtitle_and_clock_formats() -> None:
    assert format_srt_time(3723.456) == "01:02:03,456"
    assert format_vtt_time(3723.456) == "01:02:03.456"
    assert format_clock(3599.9) == "00:59:59"


def test_profiles_match_spec() -> None:
    assert PROFILES["fast"] == {"model": "base", "compute_type": "auto", "beam_size": 1}
    assert PROFILES["balanced"]["model"] == "small"
    assert PROFILES["accurate"] == {"model": "medium", "compute_type": "auto", "beam_size": 5}
    asr = ASRConfig(language="pt").with_profile("accurate")
    assert (asr.model, asr.beam_size, asr.language) == ("medium", 5, "pt")


def test_defaults_are_multilingual_cpu_int8() -> None:
    asr = ASRConfig()
    assert (asr.model, asr.compute_type, asr.device, asr.beam_size) == ("small", "auto", "cpu", 1)
    from pathlib import Path

    from saccade.asr.faster_whisper import FasterWhisperBackend, resolve_compute_type

    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("auto", "cuda") == "float16"
    # CPU cache keys are unchanged by the "auto" default, so existing indexes stay valid.
    backend = FasterWhisperBackend(asr, models_dir=Path("."), threads=1)
    assert backend.identity == {
        "backend": "faster-whisper",
        "model": "small",
        "compute_type": "int8",
        "device": "cpu",
        "beam_size": 1,
    }
    assert backend.batch_size == 1
    assert not asr.model.endswith(".en")
    assert asr.language is None
    assert asr.word_timestamps is False


def test_language_normalisation() -> None:
    assert ASRConfig(language="auto").language is None
    assert ASRConfig(language=" PT ").language == "pt"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ASRConfig(beam_size=0),
        lambda: ASRConfig(model=""),
        lambda: ASRConfig().with_profile("turbo-max"),
        lambda: VadConfig(threshold=1.5),
        lambda: VadConfig(padding_ms=-1),
        lambda: ChunkConfig(min_seconds=90, max_seconds=60),
    ],
)
def test_invalid_config_is_rejected(factory) -> None:
    with pytest.raises(ConfigError):
        factory()


def test_vad_defaults_match_spec() -> None:
    vad = VadConfig()
    assert (vad.min_speech_ms, vad.min_silence_ms, vad.padding_ms, vad.merge_gap_ms) == (
        250,
        500,
        200,
        350,
    )

"""Real faster-whisper + Silero runs on synthesized speech.

Run with ``SACCADE_INTEGRATION=1``. Uses Windows SAPI voices to synthesise speech (tests
for languages without an installed voice are skipped) and the model named by
``SACCADE_TEST_MODEL`` (default ``small``), which must already be downloaded or be
downloadable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from saccade import Video
from support import silence, tts, tts_available, write_media

pytestmark = pytest.mark.integration

MODEL = os.environ.get("SACCADE_TEST_MODEL", "small")

SCRIPTS = {
    "en": (
        "Good morning everyone. The deployment failed because the authentication secret expired.",
        "We will migrate the service to managed identity next week.",
        "authentication",
    ),
    "pt": (
        "Bom dia a todos. A implantação falhou porque a configuração da autenticação estava errada.",
        "Vamos migrar o serviço para identidade gerenciada na próxima semana.",
        "autenticação",
    ),
}


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fresh index cache per session; models come from the normal cache (no re-download)."""
    from saccade.utils.cache import default_cache_dir, models_dir

    os.environ.setdefault("SACCADE_MODELS_DIR", str(models_dir(default_cache_dir())))
    return tmp_path_factory.mktemp("integration-cache")


def _speech_video(tmp_path: Path, language: str) -> tuple[Path, float]:
    if not tts_available(language):
        pytest.skip(f"no TTS voice for {language}")
    first, second, _ = SCRIPTS[language]
    a, b = tts(first, language), tts(second, language)
    gap_start = 2.0 + len(a) / 16_000
    audio = np.concatenate([silence(2.0), a, silence(45.0), b, silence(1.5)])
    second_start = gap_start + 45.0
    path = write_media(tmp_path / f"reunião {language}.mp4", audio)
    return path, second_start


@pytest.mark.parametrize("language", ["en", "pt"])
def test_real_transcription_language_and_timestamps(
    tmp_path: Path, shared_cache: Path, language: str
) -> None:
    path, second_start = _speech_video(tmp_path, language)
    from saccade import ASRConfig

    video = Video(path, cache_dir=shared_cache, asr=ASRConfig(model=MODEL))
    summary = video.index()
    assert summary.status == "complete"
    assert summary.language == language, "language is detected automatically"
    transcript = video.transcript()
    assert len(transcript) >= 2
    # The second utterance starts after 45 s of silence; it must be stamped there.
    late = [s for s in transcript.segments if s.start > 20]
    assert late, "segments after the long silence exist"
    assert late[0].start == pytest.approx(second_start, abs=0.6)
    # Silence was not transcribed: speech time is far below media time.
    assert summary.speech_seconds < summary.duration * 0.6  # type: ignore[operator]

    keyword = SCRIPTS[language][2]
    results = video.search(keyword)
    assert results and results[0].start < 20
    context = video.context(keyword, max_tokens=400)
    assert keyword.lower()[:6] in context.text.lower()

    again = Video(path, cache_dir=shared_cache, asr=ASRConfig(model=MODEL)).index()
    assert again.cached


def test_real_explicit_language(tmp_path: Path, shared_cache: Path) -> None:
    path, _ = _speech_video(tmp_path, "pt")
    from saccade import ASRConfig

    summary = Video(path, cache_dir=shared_cache, asr=ASRConfig(model=MODEL, language="pt")).index()
    assert summary.language == "pt"
    assert summary.language_probability is None

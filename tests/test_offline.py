"""Saccade must never touch the network unless a model download is explicitly allowed."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from saccade import ASRConfig, ModelNotFoundError
from saccade.asr.faster_whisper import FasterWhisperBackend, resolve_model


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_missing_model_offline_raises_without_network(tmp_path: Path, no_network: None) -> None:
    with pytest.raises(ModelNotFoundError, match="saccade models download small"):
        resolve_model("small", tmp_path / "models", allow_download=False)


def test_unknown_model_name(tmp_path: Path, no_network: None) -> None:
    with pytest.raises(ModelNotFoundError, match="Unknown Whisper model"):
        resolve_model("smal", tmp_path / "models", allow_download=True)


def test_local_model_directory_is_used_verbatim(tmp_path: Path, no_network: None) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    assert resolve_model(str(model_dir), tmp_path / "models", allow_download=False) == str(
        model_dir.resolve()
    )


def test_backend_does_not_load_model_until_needed(tmp_path: Path, no_network: None) -> None:
    backend = FasterWhisperBackend(
        ASRConfig(), models_dir=tmp_path, threads=1, allow_download=False
    )
    assert backend.identity["model"] == "small"
    assert backend._model is None


def test_offline_env_disables_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from saccade import Video

    monkeypatch.setenv("SACCADE_OFFLINE", "1")
    assert Video(tmp_path / "x.mp4").offline is True
    monkeypatch.setenv("SACCADE_OFFLINE", "0")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert Video(tmp_path / "x.mp4").offline is False

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saccade import VadConfig, cli
from saccade.vad.silero import SileroVAD
from saccade.video import Video
from support import EnergyModel, ToneASR

TEXTS = {
    440: "A autenticação falhou no deploy.",
    660: "Lunch was good today.",
    880: "We fixed the authentication configuration.",
    1100: "Ahora el despliegue funciona — ¡perfecto!",
}


@pytest.fixture(autouse=True)
def doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    def make(path, **kwargs):
        return Video(
            path,
            asr_backend=ToneASR(texts=TEXTS),
            vad_factory=lambda: SileroVAD(VadConfig(), model=EnergyModel()),
            **kwargs,
        )

    monkeypatch.setattr(cli, "Video", make)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_index_then_search_json(capsys, tone_video: Path, cache_dir: Path) -> None:
    code, out, _ = run(capsys, "--cache-dir", str(cache_dir), "index", str(tone_video))
    assert code == 0 and "complete" in out and "4 segments" in out

    code, out, _ = run(
        capsys, "--cache-dir", str(cache_dir), "search", str(tone_video), "authentication", "--json"
    )
    assert code == 0
    assert [r["text"] for r in json.loads(out)["results"]] == [TEXTS[880]]

    code, out, _ = run(
        capsys, "--cache-dir", str(cache_dir), "search", str(tone_video), "autenticacao", "--json"
    )
    assert [r["text"] for r in json.loads(out)["results"]] == [TEXTS[440]]
    assert "autenticação" in out, "JSON output keeps Unicode readable (no \\u escapes)"


def test_search_autoindexes_and_prints_human_output(
    capsys, tone_video: Path, cache_dir: Path
) -> None:
    code, out, err = run(
        capsys, "--cache-dir", str(cache_dir), "search", str(tone_video), "despliegue"
    )
    assert code == 0
    assert "1. [1:21.0–1:23.5]" in out
    assert "¡perfecto!" in out
    assert "not indexed yet" in err


def test_context_and_info(capsys, tone_video: Path, cache_dir: Path) -> None:
    code, out, _ = run(
        capsys,
        "--cache-dir",
        str(cache_dir),
        "context",
        str(tone_video),
        "authentication configuration",
    )
    assert code == 0 and out.startswith("VIDEO: tones.mp4")
    code, out, _ = run(
        capsys, "--cache-dir", str(cache_dir), "context", str(tone_video), "lunch", "--json"
    )
    assert json.loads(out)["evidence"][0]["id"].startswith("seg_")
    code, out, _ = run(capsys, "--cache-dir", str(cache_dir), "info", str(tone_video))
    assert "Transcript:  complete" in out and "Audio:" in out
    code, out, _ = run(capsys, "--cache-dir", str(cache_dir), "info", str(tone_video), "--json")
    assert json.loads(out)["segments"] == 4


@pytest.mark.parametrize(
    ("fmt", "marker"),
    [("srt", "-->"), ("vtt", "WEBVTT"), ("md", "# tones"), ("json", '"segments"')],
)
def test_transcribe_formats(
    capsys, tone_video: Path, cache_dir: Path, tmp_path: Path, fmt: str, marker: str
) -> None:
    target = tmp_path / f"out.{fmt}"
    code, _, err = run(
        capsys,
        "--cache-dir",
        str(cache_dir),
        "transcribe",
        str(tone_video),
        "-f",
        fmt,
        "-o",
        str(target),
    )
    assert code == 0 and "Wrote 4 segments" in err
    content = target.read_text(encoding="utf-8")
    assert marker in content and "autenticação" in content


def test_transcribe_streams_text(capsys, tone_video: Path, cache_dir: Path) -> None:
    code, out, _ = run(capsys, "-q", "--cache-dir", str(cache_dir), "transcribe", str(tone_video))
    assert code == 0
    assert out.splitlines()[0].startswith("[0:02.0")


def test_errors_are_actionable(capsys, tmp_path: Path, cache_dir: Path) -> None:
    code, _, err = run(
        capsys, "--cache-dir", str(cache_dir), "index", str(tmp_path / "missing.mp4")
    )
    assert code == 1 and "MediaNotFoundError" in err and "missing.mp4" in err
    bad = tmp_path / "bad.mkv"
    bad.write_bytes(b"not a video" * 100)
    code, _, err = run(capsys, "--cache-dir", str(cache_dir), "search", str(bad), "x")
    assert code == 1 and "UnsupportedFormatError" in err and "FFmpeg reported" in err


def test_time_arguments() -> None:
    assert cli.parse_time("90") == 90
    assert cli.parse_time("1:30") == 90
    assert cli.parse_time("01:01:30.5") == 3690.5
    with pytest.raises(Exception, match="invalid time"):
        cli.parse_time("1:x")

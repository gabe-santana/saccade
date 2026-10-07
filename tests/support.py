"""Test media generation (PyAV encoders only — no ffmpeg CLI needed) and test doubles."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from saccade.asr.base import ASRSegment, LanguageDetection

SR = 16_000

# container suffix -> (video codec, audio codec)
CODECS = {
    ".mp4": ("libx264", "aac"),
    ".mov": ("libx264", "aac"),
    ".mkv": ("libx264", "libopus"),
    ".webm": ("libvpx-vp9", "libopus"),
    ".wav": (None, "pcm_s16le"),
    ".mp3": (None, "libmp3lame"),
}


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def place(total: float, events: list[tuple[float, np.ndarray]], sr: int = SR) -> np.ndarray:
    """A ``total``-second track with each ``(start, samples)`` placed at its start time."""
    out = silence(total, sr)
    for start, samples in events:
        a = round(start * sr)
        out[a : a + len(samples)] += samples[: max(0, len(out) - a)]
    return out


def write_media(
    path: Path,
    audio: np.ndarray | None,
    *,
    sr: int = SR,
    video: bool = True,
    duration: float | None = None,
    stereo: bool = False,
    audio_rate: int | None = None,
    fps: int = 10,
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Encode a test file. ``audio`` is mono float32 at ``sr``; ``None`` writes video only."""
    vcodec, acodec = CODECS[path.suffix.lower()]
    if audio is not None and audio_rate and audio_rate != sr:
        audio = _resample(audio, sr, audio_rate)
        sr = audio_rate
    if duration is None:
        duration = len(audio) / sr if audio is not None else 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        vstream = None
        if video and vcodec is not None:
            vstream = container.add_stream(vcodec, rate=fps)
            vstream.width, vstream.height = size
            vstream.pix_fmt = "yuv420p"
        astream = None
        if audio is not None:
            rate = 48_000 if acodec == "libopus" else sr
            astream = container.add_stream(acodec, rate=rate)
            astream.layout = "stereo" if stereo else "mono"
            if rate != sr:
                audio = _resample(audio, sr, rate)
                sr = rate
        if vstream is not None:
            for i in range(int(duration * fps)):
                shade = (i * 7) % 255
                img = np.full((size[1], size[0], 3), shade, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pts = i
                frame.time_base = Fraction(1, fps)
                for packet in vstream.encode(frame):
                    container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)
        if astream is not None and audio is not None:
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            channels = np.stack([pcm, pcm]) if stereo else pcm[None, :]
            frame_size = astream.codec_context.frame_size or 1024
            for start in range(0, channels.shape[1], frame_size):
                block = np.ascontiguousarray(channels[:, start : start + frame_size])
                if stereo:
                    block = np.ascontiguousarray(block.T.reshape(1, -1))
                frame = av.AudioFrame.from_ndarray(
                    block, format="s16", layout="stereo" if stereo else "mono"
                )
                frame.sample_rate = sr
                frame.pts = start
                frame.time_base = Fraction(1, sr)
                for packet in astream.encode(frame):
                    container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)
    return path


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    n = round(len(audio) * dst / src)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def decode_all(path: Path) -> np.ndarray:
    from saccade.media.audio import decode_audio

    blocks = list(decode_audio(path))
    return np.concatenate([b.samples for b in blocks]) if blocks else np.zeros(0, np.float32)


# -- speech synthesis (integration tests only) ---------------------------------------------

VOICES = {"en": "en-US", "pt": "pt-BR"}


def tts_available(language: str) -> bool:
    return sys.platform == "win32" and shutil.which("powershell") is not None and language in VOICES


def tts(text: str, language: str) -> np.ndarray:
    """Speak ``text`` with an installed Windows SAPI voice; return 16 kHz mono float32."""
    culture = VOICES[language]
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        text_file = Path(tmp) / "text.txt"
        text_file.write_text(text, encoding="utf-8")
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -eq '{culture}' }} "
            "| Select-Object -First 1;"
            "if (-not $v) { exit 3 };"
            "$s.SelectVoice($v.VoiceInfo.Name);"
            f"$s.SetOutputToWaveFile('{wav}');"
            f"$s.Speak([IO.File]::ReadAllText('{text_file}', [Text.Encoding]::UTF8));"
            "$s.Dispose()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True
        )
        return decode_all(wav)


# -- test doubles -------------------------------------------------------------------------


class EnergyModel:
    """A deterministic stand-in for Silero: probability 1 where the window is loud."""

    window_samples = 512

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
        return (rms > self.threshold).astype(np.float32)


@dataclass
class ToneASR:
    """Fake ASR: one segment per contiguous loud burst, text naming the burst's pitch.

    The tests place tones of known frequencies at known times, so a correct pipeline must
    report ``tone 440`` exactly where the 440 Hz burst was placed on the original timeline.
    """

    calls: int = 0
    audio_seconds: float = 0.0
    language: str = "en"
    texts: dict[int, str] | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {"backend": "tone-asr"}

    def detect_language(self, audio: np.ndarray) -> LanguageDetection:
        return LanguageDetection(self.language, 0.99)

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> Iterator[ASRSegment]:
        self.calls += 1
        self.audio_seconds += len(audio) / SR
        win = 160  # 10 ms
        n = len(audio) // win
        if n == 0:
            return
        rms = np.sqrt(np.mean(audio[: n * win].reshape(n, win).astype(np.float64) ** 2, axis=1))
        loud = rms > 0.05
        i = 0
        while i < n:
            if not loud[i]:
                i += 1
                continue
            j = i
            while j < n and loud[j]:
                j += 1
            if j - i >= 10:  # >= 100 ms
                piece = audio[i * win : j * win]
                spectrum = np.abs(np.fft.rfft(piece))
                freq = int(round(np.argmax(spectrum) * SR / len(piece) / 10) * 10)
                text = (self.texts or {}).get(freq, f"tone {freq}.")
                yield ASRSegment(
                    start=i * win / SR,
                    end=j * win / SR,
                    text=" " + text,
                    avg_logprob=-0.1,
                    no_speech_prob=0.01,
                )
            i = j


# -- video test media ---------------------------------------------------------------------


def pattern(seed: int, size: tuple[int, int] = (160, 96)) -> np.ndarray:
    """A distinctive blocky RGB image (a fake "slide") for seed ``seed``."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    return np.kron(blocks, np.ones((16, 16, 1), dtype=np.uint8))


def write_video(
    path: Path,
    frames: list[tuple[float, np.ndarray]],
    *,
    audio: np.ndarray | None = None,
    sr: int = SR,
) -> Path:
    """Encode ``(timestamp_seconds, rgb)`` frames with exact (possibly variable) timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tb = Fraction(1, 1000)
    with av.open(str(path), mode="w") as container:
        height, width = frames[0][1].shape[:2]
        vstream = container.add_stream("libx264", rate=30)
        vstream.width, vstream.height = width, height
        vstream.pix_fmt = "yuv420p"
        vstream.codec_context.time_base = tb
        vstream.time_base = tb
        astream = None
        if audio is not None:
            astream = container.add_stream("aac", rate=sr)
            astream.layout = "mono"
        for ts, img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = round(ts * 1000)
            frame.time_base = tb
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        if astream is not None and audio is not None:
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)[None, :]
            size = astream.codec_context.frame_size or 1024
            for start in range(0, pcm.shape[1], size):
                f = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(pcm[:, start : start + size]), format="s16", layout="mono"
                )
                f.sample_rate = sr
                f.pts = start
                f.time_base = Fraction(1, sr)
                for packet in astream.encode(f):
                    container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)
    return path


def slides(
    spec: list[tuple[float, float, int]], fps: float = 10.0
) -> list[tuple[float, np.ndarray]]:
    """Frames for ``(start, end, slide_seed)`` spans at a constant frame rate."""
    out = []
    for start, end, seed in spec:
        img = pattern(seed)
        t = start
        while t < end - 1e-9:
            out.append((round(t, 3), img))
            t += 1.0 / fps
    return out

"""Media inspection via PyAV (FFmpeg)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av

from saccade.exceptions import MediaDecodeError, MediaNotFoundError, UnsupportedFormatError

_AV_TIME_BASE = 1_000_000


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    index: int
    codec: str
    sample_rate: int
    channels: int
    language: str | None
    duration: float | None
    default: bool


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    index: int
    codec: str
    width: int
    height: int
    average_fps: float | None
    duration: float | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What FFmpeg reports about a media file. Times are in seconds."""

    path: str
    format: str
    size_bytes: int
    duration: float | None
    start_time: float
    bit_rate: int | None
    audio_streams: tuple[AudioStreamInfo, ...] = field(default_factory=tuple)
    video_streams: tuple[VideoStreamInfo, ...] = field(default_factory=tuple)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)

    @property
    def has_video(self) -> bool:
        return bool(self.video_streams)

    @property
    def primary_audio(self) -> AudioStreamInfo | None:
        return select_audio_stream(self.audio_streams)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_audio_stream(streams: tuple[AudioStreamInfo, ...]) -> AudioStreamInfo | None:
    """The stream flagged as default, otherwise the first one."""
    for stream in streams:
        if stream.default:
            return stream
    return streams[0] if streams else None


def check_media_path(path: Path) -> None:
    if not path.exists():
        raise MediaNotFoundError(f'Media file not found: "{path}".')
    if not path.is_file():
        raise MediaNotFoundError(f'Expected a media file but "{path}" is not a regular file.')


def open_container(path: Path, action: str) -> Any:
    """Open ``path`` with PyAV, translating failures into Saccade errors."""
    check_media_path(path)
    try:
        return av.open(str(path))
    except av.error.InvalidDataError as exc:
        raise UnsupportedFormatError(
            f'"{path.name}" is not a media file FFmpeg can read (unable to {action}).\n\n'
            f"FFmpeg reported:\n{_strerror(exc)}"
        ) from exc
    except av.error.FFmpegError as exc:
        raise MediaDecodeError.from_ffmpeg(path, action, exc) from exc


def probe(path: str | Path) -> MediaInfo:
    """Inspect container and stream metadata without decoding any media."""
    path = Path(path)
    with open_container(path, "inspect the media") as container:
        audio = tuple(_audio_info(s) for s in container.streams.audio)
        video = tuple(
            _video_info(s)
            for s in container.streams.video
            if not (s.disposition & av.stream.Disposition.attached_pic)
        )
        if not audio and not video:
            raise UnsupportedFormatError(f'"{path.name}" contains no audio or video streams.')
        duration = _seconds(container.duration, Fraction(1, _AV_TIME_BASE))
        if duration is None:
            durations = [
                d for d in [*(a.duration for a in audio), *(v.duration for v in video)] if d
            ]
            duration = max(durations) if durations else None
        start = _seconds(container.start_time, Fraction(1, _AV_TIME_BASE)) or 0.0
        return MediaInfo(
            path=str(path),
            format=container.format.name,
            size_bytes=path.stat().st_size,
            duration=duration,
            start_time=start,
            bit_rate=container.bit_rate or None,
            audio_streams=audio,
            video_streams=video,
        )


def _audio_info(stream: Any) -> AudioStreamInfo:
    ctx = stream.codec_context
    return AudioStreamInfo(
        index=stream.index,
        codec=ctx.name if ctx is not None else "unknown",
        sample_rate=int(ctx.sample_rate or 0) if ctx is not None else 0,
        channels=int(ctx.channels or 0) if ctx is not None else 0,
        language=stream.language or None,
        duration=_seconds(stream.duration, stream.time_base),
        default=bool(stream.disposition & av.stream.Disposition.default),
    )


def _video_info(stream: Any) -> VideoStreamInfo:
    ctx = stream.codec_context
    rate = stream.average_rate
    return VideoStreamInfo(
        index=stream.index,
        codec=ctx.name if ctx is not None else "unknown",
        width=int(ctx.width or 0) if ctx is not None else 0,
        height=int(ctx.height or 0) if ctx is not None else 0,
        average_fps=float(rate) if rate else None,
        duration=_seconds(stream.duration, stream.time_base),
    )


def _seconds(value: int | None, time_base: Fraction | None) -> float | None:
    if value is None or time_base is None:
        return None
    return round(float(value * time_base), 6)


def _strerror(exc: BaseException) -> str:
    message = getattr(exc, "strerror", None) or str(exc)
    return str(message).strip()

"""Exception hierarchy.

Every error raised by Saccade derives from :class:`SaccadeError`. Messages are written
to be actionable: they name the file involved, what Saccade was trying to do and,
where available, what FFmpeg or the model runtime reported.
"""

from __future__ import annotations

from pathlib import Path


class SaccadeError(Exception):
    """Base class for all Saccade errors."""


class ConfigError(SaccadeError, ValueError):
    """An invalid configuration value was supplied."""


class MediaError(SaccadeError):
    """Base class for problems with the input media file."""


class MediaNotFoundError(MediaError, FileNotFoundError):
    """The media file does not exist or is not a regular file."""


class MediaDecodeError(MediaError):
    """FFmpeg could not open or decode the media."""

    @classmethod
    def from_ffmpeg(cls, path: str | Path, action: str, error: BaseException) -> MediaDecodeError:
        detail = _ffmpeg_detail(error)
        return cls(f'Unable to {action} "{Path(path).name}".\n\nFFmpeg reported:\n{detail}')


class UnsupportedFormatError(MediaDecodeError):
    """The file is not a media container FFmpeg recognises, or has no usable streams."""


class AudioStreamNotFoundError(MediaError):
    """The media has no audio stream, so there is nothing to transcribe."""


class TranscriptionError(SaccadeError):
    """The ASR backend failed while transcribing."""


class ModelNotFoundError(SaccadeError):
    """The requested ASR model is unknown or not available locally."""


class DatabaseError(SaccadeError):
    """The SQLite index could not be opened, read or written."""


class IndexLockedError(DatabaseError):
    """Another process (or another Video object) is currently indexing the same video."""


class NotIndexedError(SaccadeError):
    """A query was made against a video that has no transcript yet."""


def _ffmpeg_detail(error: BaseException) -> str:
    # PyAV errors carry the FFmpeg message in ``strerror``; the str() form also
    # includes the errno and filename, which is noise for a user-facing message.
    message = getattr(error, "strerror", None) or str(error) or type(error).__name__
    message = str(message).strip()
    if message and not message.endswith("."):
        message += "."
    return message[:1].upper() + message[1:]

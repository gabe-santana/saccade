"""Cache directory resolution and an inter-process lock for index writers."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from types import TracebackType

from saccade.exceptions import IndexLockedError

CACHE_ENV = "SACCADE_CACHE_DIR"
MODELS_ENV = "SACCADE_MODELS_DIR"


def default_cache_dir() -> Path:
    """Platform cache directory, overridable with ``SACCADE_CACHE_DIR``.

    Windows: ``%LOCALAPPDATA%\\saccade``; macOS: ``~/Library/Caches/saccade``;
    elsewhere: ``$XDG_CACHE_HOME/saccade`` (default ``~/.cache/saccade``).
    """
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    platform: str = sys.platform
    if platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "saccade"
    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / "saccade"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg if xdg else Path.home() / ".cache") / "saccade"


def video_cache_dir(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / "videos" / fingerprint


def models_dir(cache_dir: Path) -> Path:
    """Where Whisper models are stored: ``SACCADE_MODELS_DIR`` or ``<cache_dir>/models``."""
    override = os.environ.get(MODELS_ENV)
    return Path(override).expanduser() if override else cache_dir / "models"


class FileLock:
    """Exclusive, non-blocking lock on a file, released automatically if the process dies.

    Used so that only one writer appends to a video's index at a time. Readers never lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock(fd)
        except OSError as exc:
            os.close(fd)
            raise IndexLockedError(
                f"{self.path.parent} is being indexed by another process or Video object. "
                "Wait for it to finish; searches against the partial index already work."
            ) from exc
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
        self._fh = fd

    def release(self) -> None:
        if self._fh is None:
            return
        fd, self._fh = self._fh, None
        with contextlib.suppress(OSError):
            _unlock(fd)
        os.close(fd)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


if sys.platform == "win32":
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

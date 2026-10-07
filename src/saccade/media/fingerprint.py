"""Stable media identity without hashing multi-gigabyte files.

``sampled`` mode (default) hashes the file size, the first and last MiB, eight evenly
spaced 64 KiB windows and a few robust stream properties (duration rounded to 0.1 s,
codecs, sample rates). It reads ~2.5 MiB regardless of file size.

The modification time is deliberately *not* part of the identity: copying or moving a
file changes its mtime but not its content, and should not trigger re-transcription.

``strict`` mode hashes every byte with SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from saccade.config import FingerprintMode
from saccade.exceptions import ConfigError
from saccade.media.probe import MediaInfo

FINGERPRINT_VERSION = 1
_EDGE_BYTES = 1 << 20
_WINDOW_BYTES = 64 << 10
_WINDOWS = 8
_STRICT_READ = 8 << 20


def fingerprint(path: str | Path, info: MediaInfo, mode: FingerprintMode = "sampled") -> str:
    """Return an identifier such as ``s1-3f2a...`` that changes when the media changes."""
    path = Path(path)
    if mode == "sampled":
        return f"s{FINGERPRINT_VERSION}-{_sampled_digest(path, info)}"
    if mode == "strict":
        return f"f{FINGERPRINT_VERSION}-{_full_digest(path)}"
    raise ConfigError(f"Unknown fingerprint mode {mode!r}; use 'sampled' or 'strict'.")


def _stream_signature(info: MediaInfo) -> str:
    duration = f"{info.duration:.1f}" if info.duration is not None else "?"
    audio = ",".join(f"{s.codec}/{s.sample_rate}/{s.channels}" for s in info.audio_streams)
    video = ",".join(f"{s.codec}/{s.width}x{s.height}" for s in info.video_streams)
    return f"{info.format}|{duration}|a:{audio}|v:{video}"


def _sampled_digest(path: Path, info: MediaInfo) -> str:
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"saccade-fp{FINGERPRINT_VERSION}|{size}|{_stream_signature(info)}".encode())
    with path.open("rb") as fh:
        if size <= 2 * _EDGE_BYTES + _WINDOWS * _WINDOW_BYTES:
            digest.update(fh.read())
            return digest.hexdigest()
        digest.update(fh.read(_EDGE_BYTES))
        span = size - 2 * _EDGE_BYTES - _WINDOW_BYTES
        for i in range(_WINDOWS):
            fh.seek(_EDGE_BYTES + span * (i + 1) // (_WINDOWS + 1))
            digest.update(fh.read(_WINDOW_BYTES))
        fh.seek(size - _EDGE_BYTES)
        digest.update(fh.read(_EDGE_BYTES))
    return digest.hexdigest()


def _full_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_STRICT_READ):
            digest.update(chunk)
    return digest.hexdigest()

"""Cheap image similarity: thumbnail differences and a 256-bit difference hash."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Thumb = NDArray[np.uint8]

THUMB_SIZE = (64, 36)  # width, height of the grayscale thumbnail used for change detection
HASH_SIZE = (17, 16)  # dHash input; yields 16 x 16 = 256 bits


def mean_abs_diff(a: Thumb, b: Thumb) -> float:
    """Average per-pixel difference of two grayscale thumbnails, 0 (same) .. 255."""
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def dhash(gray: Thumb) -> int:
    """Difference hash of a ``HASH_SIZE`` grayscale image: robust to scaling and compression."""
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

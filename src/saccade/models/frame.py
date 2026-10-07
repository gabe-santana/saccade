"""Representative video frames."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from saccade.utils.time import format_timestamp


@dataclass(frozen=True, slots=True)
class Frame:
    """A stored still image from the video.

    Attributes:
        id: Evidence identifier, e.g. ``frame_00042``.
        timestamp: Presentation time of the decoded frame on the video timeline (seconds).
        path: Absolute path of the stored JPEG.
        width: Stored image width in pixels.
        height: Stored image height in pixels.
        reason: Why it was kept: ``start``, ``change`` (visual change), ``interval`` or ``end``.
    """

    id: str
    timestamp: float
    path: str
    width: int = 0
    height: int = 0
    reason: str = "change"

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def data_url(self) -> str:
        """``data:image/jpeg;base64,...`` — ready for multimodal LLM APIs."""
        return "data:image/jpeg;base64," + base64.b64encode(self.read_bytes()).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"[{format_timestamp(self.timestamp)} | {self.id}] {self.path}"

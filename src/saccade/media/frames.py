"""Video frame sampling and JPEG encoding via PyAV (no OpenCV, no Pillow).

Frames are decoded with FFmpeg's own threading; non-reference frames are skipped when
sampling sparsely, which roughly halves decode work for typical H.264/HEVC. Every
timestamp is the frame's presentation time on the container timeline (so variable
frame rate video is handled correctly) — never a frame number times a nominal fps.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from saccade.exceptions import MediaDecodeError
from saccade.media.probe import open_container
from saccade.utils.logs import get_logger, log_event
from saccade.vision.hashing import HASH_SIZE, THUMB_SIZE, Thumb

logger = get_logger(__name__)

_MAX_CONSECUTIVE_ERRORS = 50


@dataclass(slots=True)
class SampledFrame:
    time: float
    frame: Any  # av.VideoFrame
    thumb: Thumb
    hash_input: Thumb


def has_video_stream(container: Any) -> bool:
    return _pick_video(container) is not None


def _pick_video(container: Any) -> Any:
    for stream in container.streams.video:
        if not (stream.disposition & av.stream.Disposition.attached_pic):
            return stream
    return None


def sample_frames(
    path: str | Path, *, sample_fps: float, keyframes_only: bool = False
) -> Iterator[SampledFrame]:
    """Yield about ``sample_fps`` decoded frames per second, each with tiny gray thumbnails."""
    path = Path(path)
    interval = 1.0 / sample_fps
    container = open_container(path, "decode the video stream from")
    with container:
        stream = _pick_video(container)
        if stream is None:
            return
        stream.thread_type = "AUTO"
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        elif sample_fps <= 5:
            with contextlib.suppress(AttributeError, ValueError):  # older PyAV
                stream.codec_context.skip_frame = "NONREF"
        origin = (container.start_time or 0) / av.time_base
        rate = float(stream.average_rate) if stream.average_rate else 25.0
        next_time = 0.0
        last_time = -1.0
        decoded = 0
        errors = 0
        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except av.error.FFmpegError as exc:
                if not decoded:
                    raise MediaDecodeError.from_ffmpeg(
                        path, "decode the video stream from", exc
                    ) from exc
                log_event(
                    logger, logging.WARNING, "video.truncated", file=path.name, error=str(exc)
                )
                break
            try:
                frames = packet.decode()
                errors = 0
            except av.error.FFmpegError as exc:
                errors += 1
                if errors > _MAX_CONSECUTIVE_ERRORS:
                    raise MediaDecodeError.from_ffmpeg(
                        path, "decode the video stream from", exc
                    ) from exc
                continue
            for frame in frames:
                decoded += 1
                time = frame.time - origin if frame.time is not None else last_time + 1.0 / rate
                if time < next_time - 1e-6 or time <= last_time:
                    continue
                last_time = time
                next_time = time + interval
                thumb = frame.reformat(width=THUMB_SIZE[0], height=THUMB_SIZE[1], format="gray")
                small = frame.reformat(width=HASH_SIZE[0], height=HASH_SIZE[1], format="gray")
                yield SampledFrame(
                    time=round(max(0.0, time), 3),
                    frame=frame,
                    thumb=_plane(thumb, THUMB_SIZE),
                    hash_input=_plane(small, HASH_SIZE),
                )


def _plane(frame: Any, size: tuple[int, int]) -> Thumb:
    width, height = size
    array = frame.to_ndarray()
    return np.ascontiguousarray(array[:height, :width], dtype=np.uint8)


class JpegWriter:
    """Encodes frames to JPEG with FFmpeg's MJPEG encoder, downscaled to ``max_width``."""

    def __init__(self, *, max_width: int, quality: int) -> None:
        self.max_width = max_width
        self.quality = quality

    def encode(self, frame: Any) -> tuple[bytes, int, int]:
        width, height = frame.width, frame.height
        if width > self.max_width:
            height = round(height * self.max_width / width)
            width = self.max_width
        width -= width % 2
        height -= height % 2
        context = av.CodecContext.create("mjpeg", "w")
        context.width = width
        context.height = height
        context.pix_fmt = "yuvj420p"
        context.time_base = Fraction(1, 25)
        context.qmin = self.quality
        context.qmax = self.quality
        context.options = {"huffman": "optimal"}
        packets = list(
            context.encode(frame.reformat(width=width, height=height, format="yuvj420p"))
        )
        packets += list(context.encode(None))
        return b"".join(bytes(p) for p in packets), width, height


@dataclass(frozen=True, slots=True)
class GrabbedFrame:
    """A frame decoded on demand at a requested time (optionally cropped), as JPEG."""

    time: float
    jpeg: bytes
    width: int
    height: int


Box = tuple[float, float, float, float]
"""``(x, y, width, height)`` as fractions of the frame, origin top-left."""


def grab_frame(
    path: str | Path,
    time: float,
    *,
    box: Box | None = None,
    max_width: int = 1600,
    quality: int = 3,
) -> GrabbedFrame:
    """Decode the frame shown at ``time`` seconds, crop ``box`` at full resolution, encode JPEG.

    Seeks to the preceding keyframe and decodes forward, so it costs at most one GOP of
    decoding (typically well under a second) regardless of where ``time`` is.
    """
    path = Path(path)
    container = open_container(path, "decode a frame from")
    with container:
        stream = _pick_video(container)
        if stream is None:
            raise MediaDecodeError(f'"{path.name}" has no video stream to take frames from.')
        stream.thread_type = "AUTO"
        origin = (container.start_time or 0) / av.time_base
        target = max(0.0, time)
        with contextlib.suppress(av.error.FFmpegError):
            container.seek(int((target + origin) * av.time_base), backward=True, any_frame=False)
        chosen: Any = None
        chosen_time = 0.0
        try:
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                frame_time = frame.time - origin
                if chosen is not None and frame_time > target + 1e-3:
                    break  # the previous frame is the one on screen at ``target``
                chosen, chosen_time = frame, frame_time
                if frame_time >= target - 1e-3:
                    break
        except av.error.FFmpegError as exc:
            if chosen is None:
                raise MediaDecodeError.from_ffmpeg(path, "decode a frame from", exc) from exc
        if chosen is None:
            raise MediaDecodeError(f'No frame could be decoded from "{path.name}" at {time:.2f}s.')
        image = chosen
        if box is not None:
            rgb = chosen.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            x, y, w, h = (min(max(v, 0.0), 1.0) for v in box)
            left, top = int(x * width), int(y * height)
            right = max(left + 32, min(width, int((x + w) * width)))
            bottom = max(top + 32, min(height, int((y + h) * height)))
            crop = np.ascontiguousarray(rgb[top:bottom, left:right])
            image = av.VideoFrame.from_ndarray(crop, format="rgb24")
        data, out_width, out_height = JpegWriter(max_width=max_width, quality=quality).encode(image)
        return GrabbedFrame(
            time=round(max(0.0, chosen_time), 3), jpeg=data, width=out_width, height=out_height
        )

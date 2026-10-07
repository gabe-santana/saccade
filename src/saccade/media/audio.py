"""Streaming audio decoding to 16 kHz mono float32 via PyAV.

Audio is decoded packet by packet and handed out in blocks of a few seconds, so memory
use is independent of the video's length and no temporary WAV file is ever written.

Timestamps are on the *media timeline*: 0.0 is the start of the container, as shown by
video players. The decoder keeps that alignment even when the audio stream starts late,
has gaps, or overlaps (gaps are filled with silence, overlaps are dropped).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
from numpy.typing import NDArray

from saccade.exceptions import AudioStreamNotFoundError, MediaDecodeError
from saccade.media.probe import open_container
from saccade.utils.logs import get_logger, log_event

SAMPLE_RATE = 16_000
"""Sample rate expected by Whisper and Silero VAD."""

_GAP_TOLERANCE_S = 0.1
_MAX_CONSECUTIVE_ERRORS = 50

logger = get_logger(__name__)

Samples = NDArray[np.float32]


@dataclass(slots=True)
class AudioBlock:
    """A contiguous run of mono float32 samples starting at ``start`` seconds."""

    start: float
    samples: Samples
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end(self) -> float:
        return self.start + self.duration


class _BlockAssembler:
    """Accumulates resampled pieces into blocks and tracks the media-time cursor."""

    def __init__(self, *, start: float, block_samples: int, sample_rate: int) -> None:
        self.lower_bound = max(0.0, start)
        self.block_samples = block_samples
        self.sample_rate = sample_rate
        self.cursor: float | None = None
        self._pieces: list[Samples] = []
        self._size = 0
        self._block_start = 0.0

    def push(self, samples: Samples) -> AudioBlock | None:
        if self.cursor is None or not len(samples):
            return None
        if self.cursor < self.lower_bound:
            drop = min(len(samples), round((self.lower_bound - self.cursor) * self.sample_rate))
            samples = samples[drop:]
            self.cursor += drop / self.sample_rate
            if not len(samples):
                return None
        if not self._pieces:
            self._block_start = self.cursor
        self._pieces.append(samples)
        self._size += len(samples)
        self.cursor += len(samples) / self.sample_rate
        return self.take() if self._size >= self.block_samples else None

    def take(self) -> AudioBlock | None:
        if not self._pieces:
            return None
        data = self._pieces[0] if len(self._pieces) == 1 else np.concatenate(self._pieces)
        block = AudioBlock(
            start=round(self._block_start, 6),
            samples=np.ascontiguousarray(data, dtype=np.float32),
            sample_rate=self.sample_rate,
        )
        self._pieces, self._size = [], 0
        return block


def decode_audio(
    path: str | Path,
    *,
    start: float = 0.0,
    block_seconds: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
    stream_index: int | None = None,
) -> Iterator[AudioBlock]:
    """Yield the audio of ``path`` as mono float32 blocks, beginning at ``start`` seconds.

    Raises:
        AudioStreamNotFoundError: the file has no audio stream.
        MediaDecodeError: nothing could be decoded.
    """
    path = Path(path)
    container = open_container(path, "decode the audio stream from")
    with container:
        stream = _pick_stream(container, stream_index, path)
        origin = (container.start_time or 0) / av.time_base
        if start > 0:
            _seek(container, start + origin, path)

        assembler = _BlockAssembler(
            start=start,
            block_samples=max(1, int(block_seconds * sample_rate)),
            sample_rate=sample_rate,
        )
        resampler = _Resampler(sample_rate)
        expected_input: float | None = None
        decoded_frames = 0
        consecutive_errors = 0
        last_error: BaseException | None = None

        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except av.error.FFmpegError as exc:
                # A damaged container: keep what was decoded so far.
                if not decoded_frames:
                    raise MediaDecodeError.from_ffmpeg(
                        path, "decode the audio stream from", exc
                    ) from exc
                log_event(
                    logger, logging.WARNING, "audio.truncated", file=path.name, error=str(exc)
                )
                break
            try:
                frames = packet.decode()
                consecutive_errors = 0
            except av.error.FFmpegError as exc:
                consecutive_errors += 1
                last_error = exc
                if consecutive_errors > _MAX_CONSECUTIVE_ERRORS:
                    raise MediaDecodeError.from_ffmpeg(
                        path, "decode the audio stream from", exc
                    ) from exc
                continue

            for frame in frames:
                frame_time = frame.time - origin if frame.time is not None else expected_input
                if assembler.cursor is None:
                    assembler.cursor = frame_time if frame_time is not None else 0.0
                elif frame_time is not None and expected_input is not None:
                    drift = frame_time - expected_input
                    if drift > _GAP_TOLERANCE_S:
                        gap = np.zeros(round(drift * sample_rate), dtype=np.float32)
                        block = assembler.push(gap)
                        if block is not None:
                            yield block
                    elif drift < -_GAP_TOLERANCE_S:
                        continue  # overlapping audio: keep the timeline monotonic
                base = frame_time if frame_time is not None else (expected_input or 0.0)
                expected_input = base + frame.samples / frame.sample_rate
                decoded_frames += 1
                for out in resampler.resample(frame):
                    block = assembler.push(out)
                    if block is not None:
                        yield block

        for out in resampler.resample(None):
            block = assembler.push(out)
            if block is not None:
                yield block
        tail = assembler.take()
        if tail is not None:
            yield tail

        if not decoded_frames and last_error is not None:
            raise MediaDecodeError.from_ffmpeg(path, "decode the audio stream from", last_error)
        if consecutive_errors or last_error is not None:
            log_event(
                logger,
                logging.WARNING,
                "audio.decode_errors",
                file=path.name,
                error=str(last_error),
            )


def _pick_stream(container: Any, stream_index: int | None, path: Path) -> Any:
    streams = list(container.streams.audio)
    if not streams:
        raise AudioStreamNotFoundError(
            f'"{path.name}" has no audio stream, so there is nothing to transcribe.'
        )
    if stream_index is not None:
        for stream in streams:
            if stream.index == stream_index:
                return stream
        raise AudioStreamNotFoundError(
            f'"{path.name}" has no audio stream with index {stream_index}.'
        )
    for stream in streams:
        if stream.disposition & av.stream.Disposition.default:
            return stream
    return streams[0]


def _seek(container: Any, seconds: float, path: Path) -> None:
    try:
        container.seek(int(seconds * av.time_base), backward=True, any_frame=False)
    except av.error.FFmpegError as exc:
        # Unseekable input: decoding from the start and discarding is still correct.
        log_event(logger, logging.DEBUG, "audio.seek_failed", file=path.name, error=str(exc))


class _Resampler:
    """Converts any input layout/rate to mono float32, rebuilding itself if the input changes."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._resampler = self._create()

    def _create(self) -> Any:
        return av.AudioResampler(format="flt", layout="mono", rate=self.sample_rate)

    def resample(self, frame: Any) -> Iterator[Samples]:
        try:
            outputs = self._resampler.resample(frame)
        except ValueError:
            # Input layout or rate changed mid-stream: flush the old graph, start a new one.
            outputs = list(self._resampler.resample(None))
            self._resampler = self._create()
            outputs += self._resampler.resample(frame)
        for out in outputs:
            yield out.to_ndarray().reshape(-1).astype(np.float32, copy=False)

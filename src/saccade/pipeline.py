"""The streaming transcription pipeline.

::

    [background thread]  decode ─► VAD ─► pack speech (~30 s)  ─┐  bounded queue
    [caller's thread]    ASR (one model, N threads) ─► map timestamps ─► SQLite + FTS ─► yield

Speech regions are concatenated into packs of about one Whisper window, so silence is
never transcribed and each ASR call is fully used. Every pack remembers where each
region came from, and segment times are mapped back onto the original timeline before
anything is stored. Each pack is committed in its own transaction together with the
resume point, so an interrupted run continues exactly where it stopped.
"""

from __future__ import annotations

import bisect
import logging
import math
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from saccade.asr.base import ASRBackend, ASRSegment
from saccade.config import ASRConfig, ChunkConfig
from saccade.index.chunking import Chunker
from saccade.index.database import Database, NewChunk, RunRecord, utcnow
from saccade.media.audio import SAMPLE_RATE, AudioBlock, Samples, decode_audio
from saccade.models.segment import TranscriptSegment, TranscriptWord, segment_id
from saccade.progress import Reporter, Stage, span_message
from saccade.utils.concurrency import BoundedProducer
from saccade.utils.logs import get_logger, log_event
from saccade.utils.time import format_clock
from saccade.vad.base import SpeechRegion, VoiceActivityDetector

logger = get_logger(__name__)

PACK_SECONDS = 30.0
"""Target amount of speech per ASR call: one Whisper window."""

FIRST_PACK_SECONDS = 10.0
"""The first pack is smaller so the first transcript lines arrive quickly."""

MAX_PACK_SPAN_S = 120.0
"""A pack is sent once it spans this much of the timeline, even if it is not full, so
videos with sparse speech still produce results steadily."""

MAX_JOIN_GAP_S = 3.0
"""Regions further apart than this on the timeline are never concatenated into one pack.
Whisper does not respect seams, so joining across a long silence can yield one segment
whose timestamps span the silence."""

_SEAM_TOLERANCE_S = 1.0

_HEARTBEAT_S = 60.0
_LANGUAGE_PROBE_S = 30.0
_PROMPT_CHARS = 200
_QUEUE_PACKS = 3


# -- audio buffering and packing --------------------------------------------------------


class AudioBuffer:
    """Recent decoded audio, trimmed as soon as VAD no longer needs it."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._blocks: deque[AudioBlock] = deque()

    def append(self, block: AudioBlock) -> None:
        self._blocks.append(block)

    def slice(self, start: float, end: float) -> Samples:
        sr = self.sample_rate
        parts = []
        for block in self._blocks:
            if block.end <= start or block.start >= end:
                continue
            a = max(0, round((start - block.start) * sr))
            b = min(len(block.samples), round((end - block.start) * sr))
            if b > a:
                parts.append(block.samples[a:b])
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

    def discard_before(self, time_s: float) -> None:
        while self._blocks and self._blocks[0].end <= time_s:
            self._blocks.popleft()

    @property
    def seconds(self) -> float:
        return sum(b.duration for b in self._blocks)


@dataclass(slots=True)
class Pack:
    """Concatenated speech regions plus the map back to the media timeline."""

    regions: list[SpeechRegion]
    audio: Samples
    offsets: list[int]
    lengths: list[int]
    sample_rate: int = SAMPLE_RATE

    @property
    def start(self) -> float:
        return self.regions[0].start

    @property
    def end(self) -> float:
        return self.regions[-1].end

    @property
    def speech_seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    def map_span(self, start: float, end: float) -> tuple[float, float]:
        """Map a segment's ``[start, end]`` within ``audio`` to the source timeline.

        Whisper's segment boundaries are approximate. A segment that only grazes a
        neighbouring region across a seam (less than a second, or under a quarter of its
        length) is trimmed to the regions it really covers; otherwise its start could be
        mapped to the end of the *previous* utterance, possibly minutes earlier.
        """
        sr = self.sample_rate
        s, e = start * sr, max(start, end) * sr
        covered = [
            i
            for i, (offset, length) in enumerate(zip(self.offsets, self.lengths, strict=True))
            if min(e, offset + length) > max(s, offset)
        ]
        if len(covered) > 1:
            minimum = min(_SEAM_TOLERANCE_S * sr, 0.25 * (e - s))
            kept = [
                i
                for i in covered
                if min(e, self.offsets[i] + self.lengths[i]) - max(s, self.offsets[i]) >= minimum
            ] or covered
            s = max(s, self.offsets[kept[0]])
            e = min(e, self.offsets[kept[-1]] + self.lengths[kept[-1]])
        media_start = self.to_media_time(s / sr)
        return media_start, max(media_start, self.to_media_time(e / sr, is_end=True))

    def to_media_time(self, t: float, *, is_end: bool = False) -> float:
        """Map a time within ``audio`` to the source timeline.

        A time exactly on the seam between two regions belongs to the earlier region when
        it is an end time and to the later region when it is a start time.
        """
        sample = max(0.0, t * self.sample_rate)
        if is_end:
            i = bisect.bisect_left(self.offsets, sample) - 1
        else:
            i = bisect.bisect_right(self.offsets, sample) - 1
        i = min(max(i, 0), len(self.regions) - 1)
        local = min(max(sample - self.offsets[i], 0.0), float(self.lengths[i]))
        return self.regions[i].start + local / self.sample_rate


class Packer:
    """Accumulates speech regions into packs of roughly one Whisper window."""

    def __init__(
        self,
        max_seconds: float = PACK_SECONDS,
        *,
        first_seconds: float = FIRST_PACK_SECONDS,
        max_span: float = MAX_PACK_SPAN_S,
        max_join_gap: float = MAX_JOIN_GAP_S,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.max_join_gap = max_join_gap
        self.max_samples = int(max_seconds * sample_rate)
        self.first_samples = int(min(first_seconds, max_seconds) * sample_rate)
        self.max_span = max_span
        self.sample_rate = sample_rate
        self._regions: list[SpeechRegion] = []
        self._audio: list[Samples] = []
        self._size = 0
        self._emitted = 0

    @property
    def _limit(self) -> int:
        return self.first_samples if self._emitted == 0 else self.max_samples

    def add(self, region: SpeechRegion, audio: Samples) -> Pack | None:
        if not len(audio):
            return None
        done = None
        if self._regions and (
            self._size + len(audio) > self._limit
            or region.start - self._regions[-1].end > self.max_join_gap
        ):
            done = self.flush()
        self._regions.append(region)
        self._audio.append(audio)
        self._size += len(audio)
        if done is None and self._size >= self._limit:
            done = self.flush()
        return done

    def due(self, position: float, next_region_from: float) -> Pack | None:
        """Flush a partially filled pack that can no longer grow usefully.

        That is the case when it already spans ``max_span`` seconds of timeline, or when no
        future region (which cannot start before ``next_region_from``) could be joined to it.
        """
        if not self._regions:
            return None
        if position - self._regions[0].start >= self.max_span:
            return self.flush()
        if next_region_from - self._regions[-1].end > self.max_join_gap:
            return self.flush()
        return None

    def flush(self) -> Pack | None:
        if not self._regions:
            return None
        lengths = [len(a) for a in self._audio]
        offsets = [0, *np.cumsum(lengths)[:-1].tolist()]
        pack = Pack(
            regions=self._regions,
            audio=np.concatenate(self._audio) if len(self._audio) > 1 else self._audio[0],
            offsets=[int(o) for o in offsets],
            lengths=lengths,
            sample_rate=self.sample_rate,
        )
        self._regions, self._audio, self._size = [], [], 0
        self._emitted += 1
        return pack


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """Emitted during long stretches without speech so progress keeps moving."""

    position: float


def produce_packs(
    path: Path,
    *,
    start: float,
    vad: VoiceActivityDetector,
    pack_seconds: float = PACK_SECONDS,
    stream_index: int | None = None,
) -> Iterator[Pack | Heartbeat]:
    """Decode → VAD → pack, yielding packs as soon as they are complete."""
    buffer = AudioBuffer()
    packer = Packer(pack_seconds)
    last_beat = start
    position = start
    for block in decode_audio(path, start=start, stream_index=stream_index):
        buffer.append(block)
        position = block.end
        for region in vad.feed(block):
            pack = packer.add(region, buffer.slice(region.start, region.end))
            if pack is not None:
                last_beat = position
                yield pack
        buffer.discard_before(vad.retain_from)
        pack = packer.due(position, vad.retain_from)
        if pack is not None:
            last_beat = position
            yield pack
        if position - last_beat >= _HEARTBEAT_S:
            last_beat = position
            yield Heartbeat(position)
    for region in vad.flush():
        pack = packer.add(region, buffer.slice(region.start, region.end))
        if pack is not None:
            yield pack
    pack = packer.flush()
    if pack is not None:
        yield pack
    yield Heartbeat(position)


# -- the pipeline -----------------------------------------------------------------------


@dataclass
class PipelineContext:
    path: Path
    db: Database
    backend: ASRBackend
    asr: ASRConfig
    chunking: ChunkConfig
    vad_factory: Callable[[], VoiceActivityDetector]
    reporter: Reporter
    duration: float | None
    workers: int = 1
    pack_seconds: float = PACK_SECONDS
    stream_index: int | None = None


@dataclass
class _State:
    run: RunRecord
    language: str | None
    next_segment: int
    chunker: Chunker
    prompt: str | None = None
    position: float = 0.0
    new_segments: int = 0
    speech_seconds: float = 0.0
    asr_seconds: float = 0.0
    started: float = field(default_factory=time.perf_counter)


def run_transcription(ctx: PipelineContext, run: RunRecord) -> Iterator[TranscriptSegment]:
    """Transcribe from ``run.processed_until`` to the end, committing as it goes.

    Yields only newly produced segments. The caller must hold the index lock.
    """
    db = ctx.db
    next_segment, next_chunk = db.next_indices(run.id)
    state = _State(
        run=run,
        language=run.language or ctx.asr.language,
        next_segment=next_segment,
        chunker=Chunker(
            ctx.chunking, next_index=next_chunk, pending=db.segments(run.id, unchunked_only=True)
        ),
        position=run.processed_until,
    )
    if ctx.asr.carry_context and run.processed_until > 0:
        tail = db.segments(run.id, start=max(0.0, run.processed_until - 30))
        state.prompt = tail[-1][1].text[-_PROMPT_CHARS:] if tail else None
    if run.processed_until > 0:
        ctx.reporter(
            Stage.RESUME,
            f"Resuming at {format_clock(run.processed_until)}",
            position=run.processed_until,
        )
    db.update_run(run.id, status="running", error=None)

    start = run.processed_until
    producer: BoundedProducer[Pack | Heartbeat] = BoundedProducer(
        lambda: produce_packs(
            ctx.path,
            start=start,
            vad=ctx.vad_factory(),
            pack_seconds=ctx.pack_seconds,
            stream_index=ctx.stream_index,
        ),
        maxsize=max(_QUEUE_PACKS, _batch_size(ctx) + 2),
        name="saccade-decode-vad",
    )
    workers = max(1, ctx.workers)
    executor = (
        ThreadPoolExecutor(workers, thread_name_prefix="saccade-asr") if workers > 1 else None
    )
    in_flight: deque[tuple[Pack, Future[tuple[list[ASRSegment], float]]]] = deque()
    batch_size = _batch_size(ctx)
    batch: list[Pack] = []
    conn = db.connect()
    try:
        for item in producer:
            if isinstance(item, Heartbeat):
                state.position = max(state.position, item.position)
                ctx.reporter(
                    Stage.DETECT_SPEECH,
                    f"Scanning for speech at {format_clock(item.position)}",
                    position=item.position,
                )
                continue
            pack = item
            if state.language is None:
                _detect_language(ctx, state, pack)
            if batch_size > 1:
                batch.append(pack)
                if len(batch) >= batch_size:
                    yield from _run_batch(ctx, conn, state, batch)
                    batch = []
                continue
            ctx.reporter(
                Stage.TRANSCRIBE,
                span_message("Transcribing", pack.start, pack.end),
                position=pack.end,
                start=pack.start,
                end=pack.end,
            )
            if executor is None:
                raw, elapsed = _transcribe_pack(ctx, pack, state.language, state.prompt)
                yield from _commit(ctx, conn, state, pack, raw, elapsed)
                continue
            in_flight.append(
                (pack, executor.submit(_transcribe_pack, ctx, pack, state.language, None))
            )
            while len(in_flight) >= workers:
                done_pack, future = in_flight.popleft()
                yield from _commit(ctx, conn, state, done_pack, *future.result())
        while in_flight:
            done_pack, future = in_flight.popleft()
            yield from _commit(ctx, conn, state, done_pack, *future.result())
        if batch:
            yield from _run_batch(ctx, conn, state, batch)

        final_chunks = state.chunker.flush()
        if final_chunks:
            db.commit_chunks(conn, run.id, final_chunks)
        end = ctx.duration if ctx.duration is not None else state.position
        db.update_run(
            run.id,
            status="complete",
            processed_until=max(end, state.position),
            completed_at=utcnow(),
            language=state.language,
        )
        _log_summary(ctx, state)
    except GeneratorExit:
        _mark_interrupted(db, run.id, None)
        raise
    except BaseException as exc:
        _mark_interrupted(db, run.id, exc)
        raise
    finally:
        producer.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        conn.close()


def _batch_size(ctx: PipelineContext) -> int:
    """Packs per ASR call: >1 only for backends that batch (faster-whisper on GPU)."""
    if not hasattr(ctx.backend, "transcribe_batch"):
        return 1
    return max(1, int(getattr(ctx.backend, "batch_size", 1) or 1))


def _run_batch(
    ctx: PipelineContext, conn: sqlite3.Connection, state: _State, packs: list[Pack]
) -> Iterator[TranscriptSegment]:
    ctx.reporter(
        Stage.TRANSCRIBE,
        span_message("Transcribing", packs[0].start, packs[-1].end),
        position=packs[-1].end,
        start=packs[0].start,
        end=packs[-1].end,
    )
    began = time.perf_counter()
    results = ctx.backend.transcribe_batch(  # type: ignore[attr-defined]
        [p.audio for p in packs], language=state.language, word_timestamps=ctx.asr.word_timestamps
    )
    elapsed = (time.perf_counter() - began) / len(packs)
    for pack, raw in zip(packs, results, strict=True):
        yield from _commit(ctx, conn, state, pack, raw, elapsed)


def _detect_language(ctx: PipelineContext, state: _State, pack: Pack) -> None:
    ctx.reporter(Stage.DETECT_LANGUAGE, "Detecting spoken language", position=pack.start)
    probe = pack.audio[: int(_LANGUAGE_PROBE_S * pack.sample_rate)]
    detection = ctx.backend.detect_language(probe)
    state.language = detection.language
    ctx.db.update_run(
        state.run.id, language=detection.language, language_probability=detection.probability
    )
    ctx.reporter(
        Stage.DETECT_LANGUAGE,
        f"Detected language: {detection.language} (p={detection.probability:.2f})",
        position=pack.start,
    )
    log_event(
        logger,
        logging.INFO,
        "asr.language",
        language=detection.language,
        probability=detection.probability,
    )


def _transcribe_pack(
    ctx: PipelineContext, pack: Pack, language: str | None, prompt: str | None
) -> tuple[list[ASRSegment], float]:
    began = time.perf_counter()
    raw = list(
        ctx.backend.transcribe(
            pack.audio, language=language, word_timestamps=ctx.asr.word_timestamps, prompt=prompt
        )
    )
    return raw, time.perf_counter() - began


def _commit(
    ctx: PipelineContext,
    conn: sqlite3.Connection,
    state: _State,
    pack: Pack,
    raw: list[ASRSegment],
    elapsed: float,
) -> Iterator[TranscriptSegment]:
    segments: list[tuple[int, TranscriptSegment]] = []
    for item in raw:
        text = item.text.strip()
        if not text:
            continue
        start, end = pack.map_span(item.start, item.end)
        words = tuple(
            TranscriptWord(
                text=w.text.strip(),
                start=round(pack.to_media_time(w.start), 3),
                end=round(pack.to_media_time(w.end, is_end=True), 3),
                probability=w.probability,
            )
            for w in item.words
        )
        segment = TranscriptSegment(
            id=segment_id(state.next_segment),
            start=round(start, 3),
            end=round(end, 3),
            text=text,
            language=state.language,
            confidence=round(math.exp(item.avg_logprob), 4)
            if item.avg_logprob is not None
            else None,
            no_speech_prob=round(item.no_speech_prob, 4)
            if item.no_speech_prob is not None
            else None,
            words=words,
        )
        segments.append((state.next_segment, segment))
        state.next_segment += 1

    chunks: list[NewChunk] = []
    for idx, segment in segments:
        chunks += state.chunker.add(idx, segment)

    ctx.db.commit_batch(
        conn,
        state.run.id,
        segments=segments,
        regions=pack.regions,
        chunks=chunks,
        processed_until=pack.end,
        speech_seconds=pack.speech_seconds,
        elapsed_seconds=elapsed,
    )
    state.position = max(state.position, pack.end)
    state.new_segments += len(segments)
    state.speech_seconds += pack.speech_seconds
    state.asr_seconds += elapsed
    if ctx.asr.carry_context and segments:
        state.prompt = segments[-1][1].text[-_PROMPT_CHARS:]
    log_event(
        logger,
        logging.DEBUG,
        "asr.pack",
        start=pack.start,
        end=pack.end,
        speech=pack.speech_seconds,
        segments=len(segments),
        seconds=elapsed,
    )
    for _, segment in segments:
        yield segment


def _mark_interrupted(db: Database, run_id: int, error: BaseException | None) -> None:
    try:
        db.update_run(
            run_id,
            status="partial",
            error=None if error is None else f"{type(error).__name__}: {error}",
        )
    except Exception:  # never mask the original error
        logger.debug("could not record interrupted run", exc_info=True)


def _log_summary(ctx: PipelineContext, state: _State) -> None:
    wall = time.perf_counter() - state.started
    log_event(
        logger,
        logging.INFO,
        "transcription.complete",
        file=ctx.path.name,
        segments=state.new_segments,
        speech_seconds=state.speech_seconds,
        asr_seconds=state.asr_seconds,
        wall_seconds=wall,
    )

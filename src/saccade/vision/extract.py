"""Representative-frame extraction: decode → thumbnails → select → JPEG → SQLite."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from saccade.config import VisualConfig
from saccade.index.database import Database, NewFrame, RunRecord, utcnow
from saccade.media.frames import JpegWriter, sample_frames
from saccade.progress import Reporter, Stage
from saccade.utils.logs import get_logger, log_event
from saccade.utils.time import format_clock
from saccade.vision.hashing import dhash
from saccade.vision.sampling import Candidate, Decision, FrameSelector

logger = get_logger(__name__)

_COMMIT_EVERY = 16
_REPORT_EVERY_S = 30.0


def frame_id(index: int) -> str:
    return f"frame_{index:05d}"


@dataclass(frozen=True, slots=True)
class FrameStats:
    sampled: int
    kept: int
    seconds: float


def extract_frames(
    path: Path,
    *,
    db: Database,
    run: RunRecord,
    out_dir: Path,
    config: VisualConfig,
    reporter: Reporter,
) -> FrameStats:
    """Store representative frames of ``path`` for ``run``; returns counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = FrameSelector(config)
    writer = JpegWriter(max_width=config.max_width, quality=config.jpeg_quality)
    began = time.perf_counter()
    batch: list[NewFrame] = []
    sampled = 0
    index = 0
    last_report = -_REPORT_EVERY_S
    conn = db.connect()

    def save(decision: Decision) -> None:
        nonlocal index
        index += 1
        data, width, height = writer.encode(decision.candidate.token)
        name = f"{frame_id(index)}.jpg"
        (out_dir / name).write_bytes(data)
        batch.append(
            NewFrame(
                idx=index,
                timestamp=decision.candidate.time,
                path=name,
                width=width,
                height=height,
                dhash=f"{decision.candidate.hash:064x}",
                reason=decision.reason,
            )
        )

    reporter(Stage.FRAMES, "Extracting representative frames", position=0.0)
    try:
        for sample in sample_frames(path, sample_fps=config.sample_fps):
            sampled += 1
            candidate = Candidate(sample.time, sample.thumb, dhash(sample.hash_input), sample.frame)
            for decision in selector.offer(candidate):
                save(decision)
            if len(batch) >= _COMMIT_EVERY:
                db.insert_frames(conn, run.id, batch)
                batch.clear()
            if sample.time - last_report >= _REPORT_EVERY_S:
                last_report = sample.time
                reporter(
                    Stage.FRAMES,
                    f"Extracting frames at {format_clock(sample.time)} ({index} kept)",
                    position=sample.time,
                )
        for decision in selector.finish():
            save(decision)
        db.insert_frames(conn, run.id, batch)
    finally:
        conn.close()

    elapsed = time.perf_counter() - began
    db.update_run(run.id, status="complete", completed_at=utcnow(), elapsed_seconds=elapsed)
    log_event(
        logger,
        logging.INFO,
        "frames.complete",
        file=path.name,
        sampled=sampled,
        kept=index,
        seconds=elapsed,
    )
    return FrameStats(sampled=sampled, kept=index, seconds=elapsed)

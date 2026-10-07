"""SQLite storage for transcripts, chunks and the full-text index.

The database runs in WAL mode: one writer (the indexing pipeline) commits after every
transcribed block while any number of readers search concurrently, which is how search
works while a long video is still being processed. Connections are short-lived and
per-thread; nothing is shared globally.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from saccade.exceptions import DatabaseError
from saccade.index.fts import TOKENIZER, TOKENIZER_FALLBACK, normalize_text
from saccade.index.schema import FTS_TABLE, SCHEMA_VERSION, TABLES
from saccade.models.segment import TranscriptSegment, TranscriptWord, chunk_id, segment_id
from saccade.models.transcript import TranscriptChunk
from saccade.vad.base import SpeechRegion

RunStatus = Literal["running", "partial", "complete", "failed"]


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: int
    kind: str
    config_key: str
    config: dict[str, Any]
    status: RunStatus
    language: str | None
    language_probability: float | None
    processed_until: float
    speech_seconds: float
    elapsed_seconds: float
    library_version: str
    error: str | None
    started_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class NewChunk:
    idx: int
    start: float
    end: float
    text: str
    first_segment: int
    last_segment: int


@dataclass(frozen=True, slots=True)
class NewFrame:
    idx: int
    timestamp: float
    path: str
    width: int
    height: int
    dhash: str
    reason: str


@dataclass(frozen=True, slots=True)
class SegmentHit:
    idx: int
    segment: TranscriptSegment
    rank: float  # bm25(): lower is better


@dataclass(frozen=True, slots=True)
class RunStats:
    segments: int
    chunks: int
    words: int
    regions: int


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        self._initialise()

    # -- connections ---------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                self.path, timeout=30, isolation_level=None, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            return conn
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open index database {self.path}: {exc}") from exc

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """This thread's connection (opened once, reused; closed when the thread ends)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self.connect()
            self._local.conn = conn
        try:
            yield conn
        except sqlite3.Error as exc:
            raise DatabaseError(f"Index database error ({self.path.name}): {exc}") from exc

    def close(self) -> None:
        """Close the calling thread's cached connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            self._local.conn = None
            conn.close()

    @staticmethod
    @contextmanager
    def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            row = None
            with contextlib.suppress(sqlite3.OperationalError):  # fresh database
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
            if row is not None and int(row["value"]) > SCHEMA_VERSION:
                raise DatabaseError(
                    f"{self.path} was written by a newer Saccade (schema {row['value']}); upgrade Saccade."
                )
            with self.transaction(conn):
                # Statement by statement: executescript() would commit our transaction.
                for statement in filter(str.strip, TABLES.split(";")):
                    conn.execute(statement)
                self._create_fts(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    @staticmethod
    def _create_fts(conn: sqlite3.Connection) -> None:
        try:
            conn.execute(FTS_TABLE.format(tokenizer=TOKENIZER))
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower() and "no such module" in str(exc).lower():
                raise DatabaseError(
                    "This Python's SQLite library was built without FTS5, which Saccade needs for search. "
                    "Use an official python.org / conda build of Python."
                ) from exc
            # Older SQLite without the ``categories`` option.
            conn.execute(FTS_TABLE.format(tokenizer=TOKENIZER_FALLBACK))

    # -- videos ---------------------------------------------------------------------

    def upsert_video(
        self, *, fingerprint: str, path: Path, probe: dict[str, Any], duration: float | None
    ) -> None:
        stat = path.stat()
        now = utcnow()
        with self.session() as conn, self.transaction(conn):
            conn.execute(
                """
                INSERT INTO videos (fingerprint, path, name, size_bytes, mtime_ns, duration, probe_json,
                                    created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (fingerprint) DO UPDATE SET
                    path = excluded.path, name = excluded.name, mtime_ns = excluded.mtime_ns,
                    duration = excluded.duration, probe_json = excluded.probe_json,
                    updated_at = excluded.updated_at
                """,
                (
                    fingerprint,
                    str(path),
                    path.name,
                    stat.st_size,
                    stat.st_mtime_ns,
                    duration,
                    json.dumps(probe, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    # -- runs -----------------------------------------------------------------------

    def get_run(self, kind: str, config_key: str) -> RunRecord | None:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM processing_runs WHERE kind = ? AND config_key = ?",
                (kind, config_key),
            ).fetchone()
        return _run(row) if row else None

    def get_run_by_id(self, run_id: int) -> RunRecord | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM processing_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None

    def best_run(self, kind: str, preferred_key: str | None = None) -> RunRecord | None:
        """The run for ``preferred_key`` if any, else the most useful other run.

        Preference: same configuration → latest complete run → latest run with data.
        """
        if preferred_key is not None:
            run = self.get_run(kind, preferred_key)
            if run is not None:
                return run
        with self.session() as conn:
            row = conn.execute(
                """
                SELECT * FROM processing_runs WHERE kind = ?
                ORDER BY (status = 'complete') DESC, processed_until DESC, updated_at DESC LIMIT 1
                """,
                (kind,),
            ).fetchone()
        return _run(row) if row else None

    def create_run(
        self, *, kind: str, config_key: str, config: dict[str, Any], library_version: str
    ) -> RunRecord:
        now = utcnow()
        with self.session() as conn, self.transaction(conn):
            conn.execute(
                """
                INSERT INTO processing_runs (kind, config_key, config_json, status, library_version,
                                             started_at, updated_at)
                VALUES (?, ?, ?, 'running', ?, ?, ?)
                """,
                (kind, config_key, json.dumps(config, sort_keys=True), library_version, now, now),
            )
        run = self.get_run(kind, config_key)
        assert run is not None
        return run

    def delete_run(self, run_id: int) -> None:
        with self.session() as conn, self.transaction(conn):
            conn.execute(
                "DELETE FROM segment_fts WHERE rowid IN (SELECT id FROM transcript_segments WHERE run_id = ?)",
                (run_id,),
            )
            conn.execute("DELETE FROM processing_runs WHERE id = ?", (run_id,))

    def update_run(self, run_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utcnow()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.session() as conn, self.transaction(conn):
            conn.execute(
                f"UPDATE processing_runs SET {assignments} WHERE id = ?", (*fields.values(), run_id)
            )

    # -- writes ---------------------------------------------------------------------

    def commit_batch(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        *,
        segments: Sequence[tuple[int, TranscriptSegment]],
        regions: Sequence[SpeechRegion],
        chunks: Sequence[NewChunk],
        processed_until: float,
        speech_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        """Atomically append one transcribed block: segments, FTS rows, chunks, progress."""
        with self.transaction(conn):
            conn.executemany(
                "INSERT OR IGNORE INTO audio_regions (run_id, start_s, end_s) VALUES (?, ?, ?)",
                [(run_id, r.start, r.end) for r in regions],
            )
            for idx, seg in segments:
                cursor = conn.execute(
                    """
                    INSERT INTO transcript_segments (run_id, idx, start_s, end_s, text, language,
                                                     confidence, no_speech_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        idx,
                        seg.start,
                        seg.end,
                        seg.text,
                        seg.language,
                        seg.confidence,
                        seg.no_speech_prob,
                    ),
                )
                rowid = cursor.lastrowid
                conn.execute(
                    "INSERT INTO segment_fts (rowid, body) VALUES (?, ?)",
                    (rowid, normalize_text(seg.text)),
                )
                if seg.words:
                    conn.executemany(
                        """
                        INSERT INTO transcript_words (segment_id, idx, start_s, end_s, text, probability)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (rowid, i, w.start, w.end, w.text, w.probability)
                            for i, w in enumerate(seg.words)
                        ],
                    )
            self._insert_chunks(conn, run_id, chunks)
            conn.execute(
                """
                UPDATE processing_runs
                SET processed_until = ?, speech_seconds = speech_seconds + ?,
                    elapsed_seconds = elapsed_seconds + ?, updated_at = ?
                WHERE id = ?
                """,
                (processed_until, speech_seconds, elapsed_seconds, utcnow(), run_id),
            )

    def commit_chunks(
        self, conn: sqlite3.Connection, run_id: int, chunks: Sequence[NewChunk]
    ) -> None:
        with self.transaction(conn):
            self._insert_chunks(conn, run_id, chunks)

    @staticmethod
    def _insert_chunks(conn: sqlite3.Connection, run_id: int, chunks: Sequence[NewChunk]) -> None:
        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO transcript_chunks (run_id, idx, start_s, end_s, text, first_segment, last_segment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    chunk.idx,
                    chunk.start,
                    chunk.end,
                    chunk.text,
                    chunk.first_segment,
                    chunk.last_segment,
                ),
            )
            conn.execute(
                "UPDATE transcript_segments SET chunk_idx = ? WHERE run_id = ? AND idx BETWEEN ? AND ?",
                (chunk.idx, run_id, chunk.first_segment, chunk.last_segment),
            )

    # -- reads ----------------------------------------------------------------------

    def segments(
        self,
        run_id: int,
        *,
        start: float | None = None,
        end: float | None = None,
        unchunked_only: bool = False,
        with_words: bool = False,
    ) -> list[tuple[int, TranscriptSegment]]:
        """Segments ordered by time, optionally only those overlapping [start, end]."""
        sql = "SELECT * FROM transcript_segments WHERE run_id = ?"
        params: list[Any] = [run_id]
        if start is not None:
            sql += " AND end_s >= ?"
            params.append(start)
        if end is not None:
            sql += " AND start_s <= ?"
            params.append(end)
        if unchunked_only:
            sql += " AND chunk_idx IS NULL"
        sql += " ORDER BY idx"
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
            words = self._words(conn, [r["id"] for r in rows]) if with_words else {}
        return [(row["idx"], _segment(row, words.get(row["id"], ()))) for row in rows]

    def segments_between(
        self, run_id: int, first_idx: int, last_idx: int
    ) -> list[tuple[int, TranscriptSegment]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM transcript_segments WHERE run_id = ? AND idx BETWEEN ? AND ? ORDER BY idx",
                (run_id, first_idx, last_idx),
            ).fetchall()
        return [(row["idx"], _segment(row, ())) for row in rows]

    @staticmethod
    def _words(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[TranscriptWord, ...]]:
        out: dict[int, list[TranscriptWord]] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for row in conn.execute(
                f"SELECT * FROM transcript_words WHERE segment_id IN ({marks}) ORDER BY segment_id, idx",
                batch,
            ):
                out.setdefault(row["segment_id"], []).append(
                    TranscriptWord(row["text"], row["start_s"], row["end_s"], row["probability"])
                )
        return {k: tuple(v) for k, v in out.items()}

    def chunks(self, run_id: int) -> list[TranscriptChunk]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM transcript_chunks WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
        return [
            TranscriptChunk(
                id=chunk_id(row["idx"]),
                start=row["start_s"],
                end=row["end_s"],
                text=row["text"],
                segment_ids=tuple(
                    segment_id(i) for i in range(row["first_segment"], row["last_segment"] + 1)
                ),
            )
            for row in rows
        ]

    def next_indices(self, run_id: int) -> tuple[int, int]:
        with self.session() as conn:
            seg = conn.execute(
                "SELECT MAX(idx) FROM transcript_segments WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            chk = conn.execute(
                "SELECT MAX(idx) FROM transcript_chunks WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        return (0 if seg is None else seg + 1, 0 if chk is None else chk + 1)

    def search(
        self, run_id: int, match: str, *, start: float | None, end: float | None, limit: int
    ) -> list[SegmentHit]:
        sql = """
            SELECT s.*, bm25(segment_fts) AS rank
            FROM segment_fts JOIN transcript_segments AS s ON s.id = segment_fts.rowid
            WHERE segment_fts MATCH ? AND s.run_id = ?
        """
        params: list[Any] = [match, run_id]
        if start is not None:
            sql += " AND s.end_s >= ?"
            params.append(start)
        if end is not None:
            sql += " AND s.start_s <= ?"
            params.append(end)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [SegmentHit(row["idx"], _segment(row, ()), float(row["rank"])) for row in rows]

    def stats(self, run_id: int) -> RunStats:
        with self.session() as conn:

            def count(sql: str) -> int:
                return int(conn.execute(sql, (run_id,)).fetchone()[0])

            return RunStats(
                segments=count("SELECT COUNT(*) FROM transcript_segments WHERE run_id = ?"),
                chunks=count("SELECT COUNT(*) FROM transcript_chunks WHERE run_id = ?"),
                words=count(
                    "SELECT COUNT(*) FROM transcript_words w JOIN transcript_segments s "
                    "ON s.id = w.segment_id WHERE s.run_id = ?"
                ),
                regions=count("SELECT COUNT(*) FROM audio_regions WHERE run_id = ?"),
            )

    def insert_frames(
        self, conn: sqlite3.Connection, run_id: int, frames: Sequence[NewFrame]
    ) -> None:
        if not frames:
            return
        with self.transaction(conn):
            conn.executemany(
                """
                INSERT INTO frames (run_id, idx, timestamp, path, width, height, dhash, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (run_id, f.idx, f.timestamp, f.path, f.width, f.height, f.dhash, f.reason)
                    for f in frames
                ],
            )

    def frames(
        self, run_id: int, *, start: float | None = None, end: float | None = None
    ) -> list[NewFrame]:
        sql = "SELECT * FROM frames WHERE run_id = ?"
        params: list[Any] = [run_id]
        if start is not None:
            sql += " AND timestamp >= ?"
            params.append(start)
        if end is not None:
            sql += " AND timestamp <= ?"
            params.append(end)
        with self.session() as conn:
            rows = conn.execute(sql + " ORDER BY timestamp", params).fetchall()
        return [
            NewFrame(
                r["idx"],
                r["timestamp"],
                r["path"],
                r["width"],
                r["height"],
                r["dhash"],
                r["reason"],
            )
            for r in rows
        ]

    def frame_count(self, run_id: int) -> int:
        with self.session() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM frames WHERE run_id = ?", (run_id,)).fetchone()[
                    0
                ]
            )

    def runs(self, kind: str | None = None) -> list[RunRecord]:
        with self.session() as conn:
            if kind is None:
                rows = conn.execute("SELECT * FROM processing_runs ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM processing_runs WHERE kind = ? ORDER BY id", (kind,)
                ).fetchall()
        return [_run(r) for r in rows]


def _segment(row: sqlite3.Row, words: tuple[TranscriptWord, ...]) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id(row["idx"]),
        start=row["start_s"],
        end=row["end_s"],
        text=row["text"],
        language=row["language"],
        confidence=row["confidence"],
        no_speech_prob=row["no_speech_prob"],
        words=words,
    )


def _run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        kind=row["kind"],
        config_key=row["config_key"],
        config=json.loads(row["config_json"]),
        status=row["status"],
        language=row["language"],
        language_probability=row["language_probability"],
        processed_until=row["processed_until"],
        speech_seconds=row["speech_seconds"],
        elapsed_seconds=row["elapsed_seconds"],
        library_version=row["library_version"],
        error=row["error"],
        started_at=row["started_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )

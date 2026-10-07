"""SQLite schema (one database per video fingerprint).

Times are stored in seconds on the media timeline. Column names avoid the SQL keyword
``end`` (``start_s``/``end_s``). Every transcript row belongs to a processing run, keyed
by the configuration that produced it, so changing model or VAD settings never mixes
results from different configurations.
"""

from __future__ import annotations

SCHEMA_VERSION = 2

TABLES = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    path        TEXT NOT NULL,
    name        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    mtime_ns    INTEGER,
    duration    REAL,
    probe_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_runs (
    id                   INTEGER PRIMARY KEY,
    kind                 TEXT NOT NULL,
    config_key           TEXT NOT NULL,
    config_json          TEXT NOT NULL,
    status               TEXT NOT NULL,
    language             TEXT,
    language_probability REAL,
    processed_until      REAL NOT NULL DEFAULT 0,
    speech_seconds       REAL NOT NULL DEFAULT 0,
    elapsed_seconds      REAL NOT NULL DEFAULT 0,
    library_version      TEXT NOT NULL,
    error                TEXT,
    started_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    completed_at         TEXT,
    UNIQUE (kind, config_key)
);

CREATE TABLE IF NOT EXISTS audio_regions (
    run_id  INTEGER NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s   REAL NOT NULL,
    PRIMARY KEY (run_id, start_s)
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    idx            INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    text           TEXT NOT NULL,
    language       TEXT,
    confidence     REAL,
    no_speech_prob REAL,
    chunk_idx      INTEGER,
    UNIQUE (run_id, idx)
);
CREATE INDEX IF NOT EXISTS transcript_segments_time ON transcript_segments (run_id, start_s);

CREATE TABLE IF NOT EXISTS transcript_words (
    segment_id  INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL,
    probability REAL,
    PRIMARY KEY (segment_id, idx)
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    text          TEXT NOT NULL,
    first_segment INTEGER NOT NULL,
    last_segment  INTEGER NOT NULL,
    UNIQUE (run_id, idx)
);

-- v2: representative frames. ``path`` is relative to the run's frame directory.
CREATE TABLE IF NOT EXISTS frames (
    id        INTEGER PRIMARY KEY,
    run_id    INTEGER NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    idx       INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    path      TEXT NOT NULL,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    dhash     TEXT NOT NULL,
    reason    TEXT NOT NULL,
    UNIQUE (run_id, idx)
);
CREATE INDEX IF NOT EXISTS frames_time ON frames (run_id, timestamp);
"""

# rowid of segment_fts == transcript_segments.id; ``body`` is the normalised text.
FTS_TABLE = (
    'CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(body, tokenize = "{tokenizer}")'
)

"""Lexical transcript search (SQLite FTS5 / BM25). No embeddings required."""

from __future__ import annotations

from dataclasses import dataclass

from saccade.exceptions import ConfigError
from saccade.index.chunking import join_texts
from saccade.index.database import Database
from saccade.index.fts import build_match
from saccade.models.result import SearchResult
from saccade.retrieval.ranking import group_hits, score_window


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Tuning knobs for :func:`search_transcript`.

    Attributes:
        merge_gap: Hits closer than this (seconds) are reported as one passage.
        max_window: Upper bound on a merged passage's length (seconds).
        candidates_per_result: How many segment hits to consider per requested result.
    """

    merge_gap: float = 8.0
    max_window: float = 90.0
    candidates_per_result: int = 16


def search_transcript(
    db: Database,
    run_id: int,
    query: str,
    *,
    limit: int = 5,
    start: float | None = None,
    end: float | None = None,
    expand: float = 0.0,
    options: SearchOptions | None = None,
) -> list[SearchResult]:
    """Rank passages of the transcript against ``query``.

    ``start``/``end`` restrict the search to a part of the timeline. ``expand`` adds that
    many seconds of surrounding transcript to each result's text and span.
    """
    if limit < 1:
        raise ConfigError("limit must be at least 1.")
    if expand < 0:
        raise ConfigError("expand must be >= 0.")
    options = options or SearchOptions()
    match = build_match(query)
    if match is None:
        return []

    hits = db.search(
        run_id,
        match.expression,
        start=start,
        end=end,
        limit=max(64, limit * options.candidates_per_result),
    )
    if not hits:
        return []

    windows = group_hits(hits, merge_gap=options.merge_gap, max_window=options.max_window)
    for window in windows:
        hit_text = join_texts(h.segment.text for h in sorted(window.hits, key=lambda h: h.idx))
        score_window(window, match, hit_text)
    ranked = sorted(range(len(windows)), key=lambda i: windows[i].score, reverse=True)[:limit]
    top = windows[ranked[0]].score or 1.0

    results = []
    for i in ranked:
        window = windows[i]
        if expand > 0:
            rows = db.segments(run_id, start=window.start - expand, end=window.end + expand)
        else:
            rows = db.segments_between(run_id, window.first_idx, window.last_idx)
        segments = [s for _, s in rows]
        results.append(
            SearchResult(
                start=segments[0].start,
                end=max(s.end for s in segments),
                text=join_texts(s.text for s in segments),
                score=max(round(window.score / top, 4), 0.0001),
                segment_ids=tuple(s.id for s in segments),
                matched_terms=window.matched,
            )
        )
    return results

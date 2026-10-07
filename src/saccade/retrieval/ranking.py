"""Turning segment-level BM25 hits into ranked time windows.

Neighbouring hits (within ``merge_gap`` seconds) are grouped, because several query
terms said close together is stronger evidence than one isolated mention. A window's
score combines:

* its best BM25 hit plus a damped share of the others (temporal density);
* the fraction of distinct query terms present (coverage);
* a bonus when the query occurs verbatim as a phrase.
"""

from __future__ import annotations

from dataclasses import dataclass

from saccade.index.database import SegmentHit
from saccade.index.fts import MatchQuery, contains_phrase, matched_terms, tokenize

_EXTRA_HIT_WEIGHT = 0.5
_PHRASE_BONUS = 1.25


@dataclass(slots=True)
class HitWindow:
    hits: list[SegmentHit]
    start: float
    end: float
    first_idx: int
    last_idx: int
    score: float = 0.0
    matched: tuple[str, ...] = ()

    def add(self, hit: SegmentHit) -> None:
        self.hits.append(hit)
        self.start = min(self.start, hit.segment.start)
        self.end = max(self.end, hit.segment.end)
        self.first_idx = min(self.first_idx, hit.idx)
        self.last_idx = max(self.last_idx, hit.idx)


def group_hits(hits: list[SegmentHit], *, merge_gap: float, max_window: float) -> list[HitWindow]:
    windows: list[HitWindow] = []
    current: HitWindow | None = None
    for hit in sorted(hits, key=lambda h: h.idx):
        seg = hit.segment
        if (
            current is not None
            and seg.start - current.end <= merge_gap
            and seg.end - current.start <= max_window
        ):
            current.add(hit)
            continue
        current = HitWindow([hit], seg.start, seg.end, hit.idx, hit.idx)
        windows.append(current)
    return windows


def score_window(window: HitWindow, match: MatchQuery, text: str) -> float:
    strengths = sorted((-h.rank for h in window.hits), reverse=True)
    relevance = strengths[0] + _EXTRA_HIT_WEIGHT * sum(strengths[1:])
    tokens = tokenize(text)
    window.matched = matched_terms(match, tokens)
    coverage = len(window.matched) / len(match.terms) if match.terms else 0.0
    score = relevance * (0.5 + coverage)
    if contains_phrase(match, tokens):
        score *= _PHRASE_BONUS
    window.score = score
    return score

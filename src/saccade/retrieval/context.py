"""Assembling token-budgeted, timestamped, LLM-ready context.

Search hits are widened by a configurable amount of surrounding transcript (so the
LLM sees what was said around a mention, not an isolated sentence), overlapping
windows are merged, and passages are added in relevance order until the token budget
is spent. The selected passages are then printed in timeline order.

Nothing in the output is generated: every line is source transcript with the source
timestamps and segment identifiers it came from.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from saccade.exceptions import ConfigError
from saccade.index.chunking import join_texts
from saccade.index.database import Database, RunRecord
from saccade.index.fts import is_cjk
from saccade.models.context import ContextMetadata, Evidence, VideoContext
from saccade.models.frame import Frame
from saccade.models.segment import TranscriptSegment
from saccade.models.transcript import TranscriptChunk, TranscriptStatus
from saccade.retrieval.search import search_transcript
from saccade.utils.time import format_span, format_timestamp

TokenCounter = Callable[[str], int]
FrameLookup = Callable[[float, float], list[Frame]]

_CANDIDATES = 24
_MERGE_GAP_S = 1.0


@dataclass(frozen=True, slots=True)
class ContextOptions:
    """
    Attributes:
        before: Seconds of transcript to include before each hit.
        after: Seconds of transcript to include after each hit.
        segment_timestamps: Prefix every segment inside a passage with its own start time,
            which helps an LLM cite precise moments at the cost of a few tokens.
        frames_per_passage: Representative frames attached to each passage (0 disables).
    """

    before: float = 12.0
    after: float = 15.0
    segment_timestamps: bool = False
    frames_per_passage: int = 2

    def __post_init__(self) -> None:
        if self.before < 0 or self.after < 0:
            raise ConfigError("Context expansion (before/after) must be >= 0 seconds.")


def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate: ~1 token per CJK character, ~3.5 characters otherwise.

    Deliberately on the high side for Latin-script languages so budgets are rarely exceeded.
    Pass ``token_counter=`` to use your model's real tokenizer.
    """
    cjk = sum(1 for ch in text if is_cjk(ch))
    return cjk + math.ceil((len(text) - cjk) / 3.5)


@dataclass(slots=True)
class _Passage:
    start: float
    end: float
    rank: int
    segments: list[tuple[int, TranscriptSegment]]


def build_context(
    db: Database,
    run: RunRecord | None,
    *,
    query: str,
    video_name: str,
    duration: float | None,
    status: TranscriptStatus,
    max_tokens: int = 4000,
    options: ContextOptions | None = None,
    token_counter: TokenCounter | None = None,
    frames: FrameLookup | None = None,
) -> VideoContext:
    if max_tokens < 64:
        raise ConfigError("max_tokens must be at least 64.")
    options = options or ContextOptions()
    count = token_counter or estimate_tokens
    language = run.language if run else None
    indexed_until = run.processed_until if run else None

    passages = _passages(db, run, query, options) if run is not None else []
    hits = len(passages)

    header = _header(video_name, duration, language, status, indexed_until, query)
    used = count(header)
    chosen: list[tuple[_Passage, str]] = []
    for passage in sorted(passages, key=lambda p: p.rank):
        block = _render(passage.segments, options)
        cost = count(block) + 2
        if used + cost <= max_tokens:
            chosen.append((passage, block))
            used += cost
        elif not chosen:
            trimmed = _trim_to_budget(passage, options, max_tokens - used - 2, count)
            if trimmed is not None:
                chosen.append(trimmed)
                used += count(trimmed[1]) + 2
    chosen.sort(key=lambda item: item[0].start)

    body = [block for _, block in chosen] or ["No transcript passages matched this query."]
    text = header + "\n\n" + "\n\n".join(body)

    picked: list[Frame] = []
    if frames is not None and options.frames_per_passage > 0 and chosen:
        section = "\n\nRelevant visual evidence (frames from the video at these times):"
        candidates = _frames_for(chosen, frames, options.frames_per_passage)
        lines = [f"[{format_timestamp(f.timestamp)} | {f.id}]\n{f.path}" for f in candidates]
        if candidates and used + count(section) <= max_tokens:
            used += count(section)
            for frame, line in zip(candidates, lines, strict=True):
                cost = count(line) + 2
                if used + cost > max_tokens:
                    break
                picked.append(frame)
                used += cost
        if picked:
            text += (
                section
                + "\n\n"
                + "\n\n".join(
                    f"[{format_timestamp(f.timestamp)} | {f.id}]\n{f.path}" for f in picked
                )
            )

    segments_out = [_as_chunk(p.segments) for p, _ in chosen]
    evidence = [
        Evidence(id=seg.id, type="transcript", start=seg.start, end=seg.end, text=seg.text)
        for passage, _ in chosen
        for _, seg in passage.segments
    ] + [Evidence(id=f.id, type="frame", timestamp=f.timestamp) for f in picked]
    metadata = ContextMetadata(
        video=video_name,
        duration=duration,
        language=language,
        transcript_status=status,
        indexed_until=indexed_until,
        max_tokens=max_tokens,
        estimated_tokens=count(text),
        passages=len(chosen),
        hits=hits,
        model=(run.config.get("asr") or {}).get("model") if run else None,
    )
    return VideoContext(
        query=query,
        text=text,
        segments=segments_out,
        frames=picked,
        evidence=evidence,
        metadata=metadata,
    )


def _frames_for(
    chosen: list[tuple[_Passage, str]], lookup: FrameLookup, per_passage: int
) -> list[Frame]:
    """Frames inside each passage (falling back to the nearest one before it), best first."""
    out: list[Frame] = []
    seen: set[str] = set()
    for passage, _ in chosen:
        inside = lookup(passage.start, passage.end)
        if not inside:
            before = lookup(max(0.0, passage.start - 120.0), passage.start)
            inside = before[-1:]
        if len(inside) > per_passage:
            # Spread the picks across the passage.
            step = (len(inside) - 1) / (per_passage - 1) if per_passage > 1 else 0
            inside = [inside[round(i * step)] for i in range(per_passage)]
        for frame in inside:
            if frame.id not in seen:
                seen.add(frame.id)
                out.append(frame)
    return sorted(out, key=lambda f: f.timestamp)


def _passages(db: Database, run: RunRecord, query: str, options: ContextOptions) -> list[_Passage]:
    results = search_transcript(db, run.id, query, limit=_CANDIDATES)
    windows = [
        _Passage(max(0.0, r.start - options.before), r.end + options.after, rank, [])
        for rank, r in enumerate(results)
    ]
    # Overlapping windows become one passage that keeps the best rank of its parts.
    windows.sort(key=lambda w: w.start)
    merged: list[_Passage] = []
    for window in windows:
        if merged and window.start <= merged[-1].end + _MERGE_GAP_S:
            last = merged[-1]
            last.end = max(last.end, window.end)
            last.rank = min(last.rank, window.rank)
        else:
            merged.append(window)

    seen: set[int] = set()
    for window in merged:
        rows = db.segments(run.id, start=window.start, end=window.end)
        window.segments = [(idx, seg) for idx, seg in rows if idx not in seen]
        seen.update(idx for idx, _ in window.segments)
    return [w for w in merged if w.segments]


def _render(segments: list[tuple[int, TranscriptSegment]], options: ContextOptions) -> str:
    segs = [s for _, s in segments]
    first, last = segs[0], segs[-1]
    ident = first.id if first.id == last.id else f"{first.id}–{last.id}"
    head = f"[{format_span(first.start, max(s.end for s in segs))} | {ident}]"
    if options.segment_timestamps:
        lines = [f"({format_timestamp(s.start)}) {s.text.strip()}" for s in segs]
        return head + "\n" + "\n".join(lines)
    return head + "\n" + join_texts(s.text for s in segs)


def _trim_to_budget(
    passage: _Passage, options: ContextOptions, budget: int, count: TokenCounter
) -> tuple[_Passage, str] | None:
    """Shrink the best passage from its edges until it fits (used when even one is too big)."""
    segments = list(passage.segments)
    while segments:
        block = _render(segments, options)
        if count(block) <= budget:
            trimmed = _Passage(segments[0][1].start, segments[-1][1].end, passage.rank, segments)
            return trimmed, block
        # Drop from whichever edge is further from the passage centre.
        centre = (passage.start + passage.end) / 2
        if abs(segments[0][1].start - centre) >= abs(segments[-1][1].end - centre):
            segments.pop(0)
        else:
            segments.pop()
    return None


def _as_chunk(segments: list[tuple[int, TranscriptSegment]]) -> TranscriptChunk:
    segs = [s for _, s in segments]
    first, last = segs[0], segs[-1]
    return TranscriptChunk(
        id=first.id if first.id == last.id else f"{first.id}–{last.id}",
        start=first.start,
        end=max(s.end for s in segs),
        text=join_texts(s.text for s in segs),
        segment_ids=tuple(s.id for s in segs),
    )


def _header(
    name: str,
    duration: float | None,
    language: str | None,
    status: TranscriptStatus,
    indexed_until: float | None,
    query: str,
) -> str:
    facts = []
    if duration:
        facts.append(f"DURATION: {format_timestamp(duration, precision=0)}")
    if language:
        facts.append(f"LANGUAGE: {language}")
    if status == "complete":
        facts.append("TRANSCRIPT: complete")
    elif status == "empty":
        facts.append("TRANSCRIPT: none (no speech found)")
    elif indexed_until is not None:
        facts.append(
            f"TRANSCRIPT: partial, indexed up to {format_timestamp(indexed_until, precision=0)}"
        )
    lines = [f"VIDEO: {name}"]
    if facts:
        lines.append(" · ".join(facts))
    lines += [
        f"QUERY: {query}",
        "",
        "Relevant evidence (verbatim transcript; times refer to the source video):",
    ]
    return "\n".join(lines)

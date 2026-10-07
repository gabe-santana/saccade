"""Groups transcript segments into time-aware retrieval chunks, incrementally.

Rules, applied as each segment arrives:

* never let a chunk exceed ``max_seconds``;
* close a chunk at the first sentence end once it reaches ``min_seconds``;
* after the midpoint between min and max, close at the next audible pause, so text
  without punctuation still produces reasonably sized chunks;
* a long silence (``break_on_pause_s``) ends a chunk early, since it usually marks a
  change of topic.

Chunks never overlap and every segment belongs to exactly one chunk.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from saccade.config import ChunkConfig
from saccade.index.database import NewChunk
from saccade.index.fts import is_cjk
from saccade.models.segment import TranscriptSegment

_SENTENCE_END = re.compile(r"[.!?。！？…؟।][\"'”’»)\]]*$")
_PAUSE_S = 0.5


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.rstrip()))


def join_texts(texts: Iterable[str]) -> str:
    out = ""
    for raw in texts:
        text = raw.strip()
        if not text:
            continue
        if out and not (is_cjk(out[-1]) and is_cjk(text[0])):
            out += " "
        out += text
    return out


class Chunker:
    def __init__(
        self,
        config: ChunkConfig | None = None,
        *,
        next_index: int = 0,
        pending: Iterable[tuple[int, TranscriptSegment]] = (),
    ) -> None:
        self.config = config or ChunkConfig()
        self._next = next_index
        self._pending: list[tuple[int, TranscriptSegment]] = list(pending)

    def add(self, idx: int, segment: TranscriptSegment) -> list[NewChunk]:
        cfg = self.config
        out: list[NewChunk] = []
        if self._pending:
            first = self._pending[0][1]
            last = self._pending[-1][1]
            gap = segment.start - last.end
            current = last.end - first.start
            if (
                segment.end - first.start > cfg.max_seconds
                or (gap >= cfg.break_on_pause_s and current >= cfg.min_seconds / 2)
                or (gap >= _PAUSE_S and current >= (cfg.min_seconds + cfg.max_seconds) / 2)
            ):
                out += self._close()
        self._pending.append((idx, segment))
        duration = segment.end - self._pending[0][1].start
        if (
            duration >= cfg.min_seconds and ends_sentence(segment.text)
        ) or duration >= cfg.max_seconds:
            out += self._close()
        return out

    def flush(self) -> list[NewChunk]:
        return self._close()

    def _close(self) -> list[NewChunk]:
        if not self._pending:
            return []
        items, self._pending = self._pending, []
        chunk = NewChunk(
            idx=self._next,
            start=items[0][1].start,
            end=max(s.end for _, s in items),
            text=join_texts(s.text for _, s in items),
            first_segment=items[0][0],
            last_segment=items[-1][0],
        )
        self._next += 1
        return [chunk]

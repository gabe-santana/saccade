"""Search results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from saccade.utils.time import format_span


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A passage of the video that matches a query.

    Attributes:
        start: Start of the passage on the source timeline (seconds).
        end: End of the passage (seconds).
        text: Transcript text of the passage.
        score: Relevance relative to the best result for this query, in (0, 1]. It ranks
            results; it is not a probability that the passage answers the query.
        source: Where the evidence comes from (``"transcript"``).
        segment_ids: Source segments the passage consists of.
        matched_terms: Query terms that occur in the passage.
    """

    start: float
    end: float
    text: str
    score: float
    source: Literal["transcript"] = "transcript"
    segment_ids: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        if not self.segment_ids:
            return ""
        first, last = self.segment_ids[0], self.segment_ids[-1]
        return first if first == last else f"{first}–{last}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segment_ids"] = list(self.segment_ids)
        data["matched_terms"] = list(self.matched_terms)
        return data

    def __str__(self) -> str:
        return f"[{format_span(self.start, self.end)} | {self.score:.2f}] {self.text}"

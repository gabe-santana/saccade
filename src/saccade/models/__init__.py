"""Public data types returned by Saccade."""

from saccade.models.context import ContextMetadata, Evidence, VideoContext
from saccade.models.frame import Frame
from saccade.models.result import SearchResult
from saccade.models.segment import TranscriptSegment, TranscriptWord
from saccade.models.transcript import Transcript, TranscriptChunk

__all__ = [
    "ContextMetadata",
    "Evidence",
    "Frame",
    "SearchResult",
    "Transcript",
    "TranscriptChunk",
    "TranscriptSegment",
    "TranscriptWord",
    "VideoContext",
]

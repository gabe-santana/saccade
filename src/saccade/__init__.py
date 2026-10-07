"""Saccade — fast, local video context for LLMs.

::

    import saccade

    video = saccade.Video("meeting.mp4", llm=saccade.azure(endpoint=..., api_key=..., deployment="gpt-4o"))
    print(video.ask("Why did the deployment fail?"))

Without an LLM everything stays local: ``video.index()``, ``video.search()``,
``video.context()`` (prompt-ready evidence) and ``video.frames()``.
"""

from saccade._version import __version__
from saccade.config import PROFILES, ASRConfig, ChunkConfig, VadConfig, VisualConfig
from saccade.exceptions import (
    AudioStreamNotFoundError,
    ConfigError,
    DatabaseError,
    IndexLockedError,
    MediaDecodeError,
    MediaError,
    MediaNotFoundError,
    ModelNotFoundError,
    NotIndexedError,
    SaccadeError,
    TranscriptionError,
    UnsupportedFormatError,
)
from saccade.llm import LLM, AzureFoundry, LLMError, OpenAICompatible, azure, ollama, openai
from saccade.models import (
    ContextMetadata,
    Evidence,
    Frame,
    SearchResult,
    Transcript,
    TranscriptChunk,
    TranscriptSegment,
    TranscriptWord,
    VideoContext,
)
from saccade.models.answer import Answer
from saccade.progress import ProgressEvent, Stage
from saccade.video import IndexJob, IndexSummary, Video, VideoInfo

__all__ = [
    "LLM",
    "PROFILES",
    "ASRConfig",
    "Answer",
    "AudioStreamNotFoundError",
    "AzureFoundry",
    "ChunkConfig",
    "ConfigError",
    "ContextMetadata",
    "DatabaseError",
    "Evidence",
    "Frame",
    "IndexJob",
    "IndexLockedError",
    "IndexSummary",
    "LLMError",
    "MediaDecodeError",
    "MediaError",
    "MediaNotFoundError",
    "ModelNotFoundError",
    "NotIndexedError",
    "OpenAICompatible",
    "ProgressEvent",
    "SaccadeError",
    "SearchResult",
    "Stage",
    "Transcript",
    "TranscriptChunk",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptionError",
    "UnsupportedFormatError",
    "VadConfig",
    "Video",
    "VideoContext",
    "VideoInfo",
    "VisualConfig",
    "__version__",
    "azure",
    "ollama",
    "openai",
]

"""Representative frames: cheap change detection, deduplication and JPEG storage."""

from saccade.vision.extract import FrameStats, extract_frames, frame_id
from saccade.vision.sampling import FrameSelector

__all__ = ["FrameSelector", "FrameStats", "extract_frames", "frame_id"]

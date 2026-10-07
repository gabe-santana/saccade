__version__ = "0.1.0"

PROCESSING_VERSION = 1
"""Bump whenever a change alters stored output (segmentation, timestamps, chunking).

It is part of every cache key, so existing indexes are transparently rebuilt instead of
silently mixing results from different pipeline versions.
"""

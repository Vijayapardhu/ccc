"""Frame ingest and camera-side tracking."""

from __future__ import annotations

from campus.capture.source import (
    CameraWorkerStats,
    DecodeBackend,
    Frame,
    FrameSource,
    StreamConfig,
    StreamHealth,
    StreamState,
    frame_gray,
)

__all__ = [
    "CameraWorkerStats",
    "DecodeBackend",
    "Frame",
    "FrameSource",
    "StreamConfig",
    "StreamHealth",
    "StreamState",
    "frame_gray",
]

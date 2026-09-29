"""Camera-scoped multi-object tracking (ByteTrack)."""

from __future__ import annotations

from campus.track.bytetrack import ByteTracker, Track, TrackConfig, TrackState, iou_matrix

__all__ = ["ByteTracker", "Track", "TrackConfig", "TrackState", "iou_matrix"]

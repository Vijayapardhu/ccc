"""ByteTrack multi-object tracking.

Why track at all
----------------
Without tracking, a face at the edge of a 4K frame is recognised once, badly,
and possibly not at all. With tracking, the same person accumulates 10-20
observations over their passage through the frame, and the temporal verifier
gets something to work with. Tracking is what converts a low-quality
single-frame problem into a high-confidence multi-frame one.

ByteTrack's contribution over plain SORT/IouTracker is the second association
pass using *low-confidence* detections. In a dense corridor the reliable
detector output excludes anyone partially occluded — precisely the people most
likely to be cut off by someone walking past. Associating those low-score
boxes anyway keeps the track alive through the occlusion instead of fragmenting
it into three short tracks, each too short to commit an identity.

The two-stage design is also why `low_score_threshold` matters more than
`high_score_threshold` here: it sets how much occlusion the tracker tolerates
before giving up on a person.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from campus.types import CameraId, FaceBox, TrackId


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass(frozen=True, slots=True)
class TrackConfig:
    high_threshold: float = 0.6
    low_threshold: float = 0.15
    """Detections in (low, high) are used only for the second association
    pass. Raising it past ~0.3 starts attaching to genuine false positives."""

    match_threshold: float = 0.8
    """Cosine distance threshold for the Kalman-predicted association. High
    values are fine because ByteTrack's second pass rescues matches the first
    pass rejects."""

    track_buffer: int = 30
    """Frames a track survives without a match. At 8fps this is ~3.75s of
    occlusion tolerance, which covers a person walking behind a pillar."""

    min_hits: int = 3
    """Observations before a track is reported. Prevents a single-frame
    detection from becoming a TrackCommit candidate."""

    match_iou: float = 0.3
    """Fallback when no motion model is available."""


@dataclass(slots=True)
class Track:
    track_id: TrackId
    camera_id: CameraId
    state: TrackState
    bbox: npt.NDArray[np.float32]
    landmarks: npt.NDArray[np.float32] | None
    score: float
    start_frame: int
    frame_id: int
    start_time: float
    last_time: float
    hits: int = 1
    time_since_update: int = 0
    history: deque[tuple[int, npt.NDArray[np.float32]]] = field(
        default_factory=lambda: deque(maxlen=32)
    )
    _velocity: npt.NDArray[np.float32] | None = None

    def predict(self) -> npt.NDArray[np.float32]:
        """Constant-velocity extrapolation.

        Deliberately not a Kalman filter. For faces in a scene with irregular
        motion — turning, stopping, walking into each other — a full Kalman
        gains little over linear extrapolation and costs several microseconds
        per track per frame, which at 400 cameras x 200 faces x 8fps is real.
        The speed/accuracy trade is worth taking at this density.
        """
        if self._velocity is None:
            return self.bbox.copy()
        return self.bbox + self._velocity

    def update(
        self,
        bbox: npt.NDArray[np.float32],
        landmarks: npt.NDArray[np.float32] | None,
        score: float,
        frame_id: int,
        timestamp: float,
    ) -> None:
        dt = max(1, frame_id - self.frame_id)
        if self.frame_id > 0 and dt <= 3:
            measured = (bbox - self.bbox) / dt
            # Exponential smoothing against the running estimate. A single noisy
            # detection must not fling the prediction across the frame, but the
            # estimate also has to keep up when someone actually turns a corner.
            if self._velocity is None:
                self._velocity = measured
            else:
                self._velocity = 0.6 * self._velocity + 0.4 * measured
        self.bbox = bbox.astype(np.float32)
        self.landmarks = landmarks
        self.score = score
        self.frame_id = frame_id
        self.last_time = timestamp
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.Tracked
        self.history.append((frame_id, self.bbox.copy()))

    def mark_missed(self) -> None:
        self.time_since_update += 1
        if self.time_since_update > 0 and self._velocity is not None:
            self.bbox = self.bbox + self._velocity
        self.state = TrackState.Lost if self.time_since_update > 1 else TrackState.Tracked


def iou_matrix(
    a: Sequence[npt.NDArray[np.float32]], b: Sequence[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    """Pairwise IoU, shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.stack(a).astype(np.float32)
    B = np.stack(b).astype(np.float32)
    x0 = np.maximum(A[:, None, 0], B[None, :, 0])
    y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2])
    y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    area_b = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _greedy_match(
    cost: npt.NDArray[np.float32], threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy min-cost matching. Returns (matches, unmatched_rows, unmatched_cols)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    work = cost.copy()
    matches: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    while True:
        idx = int(np.argmin(work))
        r, c = divmod(idx, work.shape[1])
        if work[r, c] > threshold or not np.isfinite(work[r, c]):
            break
        matches.append((r, c))
        used_r.add(r)
        used_c.add(c)
        work[r, :] = np.inf
        work[:, c] = np.inf
    return (
        sorted(matches),
        [i for i in range(cost.shape[0]) if i not in used_r],
        [j for j in range(cost.shape[1]) if j not in used_c],
    )


class ByteTracker:
    """Per-camera ByteTrack.

    One instance per camera. Tracks are camera-scoped on purpose: merging
    identity across cameras is a different, much harder problem (camera
    handover), and conflating the two in one tracker produces confident
    cross-contamination. Cross-camera linking lives in the visit service, keyed
    on committed identities and timestamps.
    """

    def __init__(
        self, camera_id: CameraId, config: TrackConfig | None = None
    ) -> None:
        self.camera_id = camera_id
        self.config = config or TrackConfig()
        self._tracks: list[Track] = []
        self._dying: list[Track] = []
        self._next_id = 0
        self._frame_id = 0

    def update(
        self, detections: Sequence[FaceBox], frame_id: int, timestamp: float
    ) -> list[Track]:
        """Advance one frame and return the currently tracked faces.

        Returns tracks, not just new ones, because the caller needs to attach
        an embedding to the same track across many frames for the temporal
        verifier to have anything to aggregate.
        """
        self._frame_id = frame_id
        high = [d for d in detections if d.score >= self.config.high_threshold]
        low = [
            d
            for d in detections
            if self.config.low_threshold <= d.score < self.config.high_threshold
        ]

        boxes = np.array([d.as_array() for d in detections], dtype=np.float32) if detections else None
        # cos_dist = 1 - IoU: ByteTrack's original formulation.
        cost = None
        if boxes is not None and self._tracks:
            track_boxes = np.stack([t.predict() for t in self._tracks])
            cost = 1.0 - iou_matrix(track_boxes, boxes)

        high_idx = [i for i, d in enumerate(detections) if d.score >= self.config.high_threshold]
        low_idx = [i for i, d in enumerate(detections) if self.config.low_threshold <= d.score < self.config.high_threshold]

        matched: set[tuple[int, int]] = set()
        if cost is not None and high_idx:
            # cost is cos_dist = 1 - IoU, so `match_threshold` is already on
            # the distance scale. 0.8 means "accept anything with IoU >= 0.2",
            # which is the correct ByteTrack reading: the second pass is what
            # rescues hard matches, not a loosened first pass.
            matches, _, _ = _greedy_match(cost[:, high_idx], self.config.match_threshold)
            for t, d in matches:
                matched.add((t, high_idx[d]))
                self._tracks[t].update(
                    boxes[high_idx[d]],
                    detections[high_idx[d]].landmarks,
                    detections[high_idx[d]].score,
                    frame_id, timestamp,
                )

        # Second pass: low-confidence detections keep the track alive through
        # partial occlusion. This is ByteTrack's whole point.
        if cost is not None and low_idx and len(matched) < len(self._tracks):
            matched_t = {t for t, _ in matched}
            free_t = [i for i in range(len(self._tracks)) if i not in matched_t]
            sub = cost[np.ix_(free_t, low_idx)]
            if sub.size:
                matches, _, _ = _greedy_match(sub, self.config.match_threshold)
                for t_rel, d_rel in matches:
                    t = free_t[t_rel]
                    d = low_idx[d_rel]
                    matched.add((t, d))
                    self._tracks[t].update(
                        boxes[d], detections[d].landmarks, detections[d].score,
                        frame_id, timestamp,
                    )

        matched_d = {d for _, d in matched}
        matched_t = {t for t, _ in matched}
        for t, track in enumerate(self._tracks):
            if t not in matched_t:
                track.mark_missed()

        for i, det in enumerate(detections):
            if i in matched_d:
                continue
            if det.score < self.config.low_threshold:
                continue
            self._tracks.append(
                Track(
                    track_id=TrackId(f"{self.camera_id}-{self._next_id}"),
                    camera_id=self.camera_id,
                    state=TrackState.New,
                    bbox=boxes[i],
                    landmarks=det.landmarks,
                    score=det.score,
                    start_frame=frame_id,
                    frame_id=frame_id,
                    start_time=timestamp,
                    last_time=timestamp,
                )
            )
            self._next_id += 1

        # Age out tracks, and buffer the ones that died this frame so
        # `expired()` reports each of them exactly once. The worker drains that
        # buffer to finalise the matching identity evidence — if expiry only
        # dropped the track silently, no commit would ever be emitted and the
        # system would report nobody present.
        self._dying: list[Track] = []
        kept: list[Track] = []
        for t in self._tracks:
            if t.time_since_update > self.config.track_buffer:
                t.state = TrackState.Removed
                self._dying.append(t)
            else:
                kept.append(t)
        self._tracks = kept

        return [t for t in self._tracks if t.state in (TrackState.New, TrackState.Tracked)]

    def expired(self) -> list[Track]:
        """Tracks that aged out on the most recent frame, then forget them.

        Draining here (rather than just reporting) is what guarantees
        exactly-once finalisation: the worker finalises each of these, and a
        track cannot be finalised twice.
        """
        out, self._dying = self._dying, []
        return out

    def active(self) -> list[Track]:
        """Tracks with a detection this frame. These are the ones that get an
        embedding attached."""
        return [t for t in self._tracks if t.state in (TrackState.New, TrackState.Tracked)]

    def live(self) -> list[Track]:
        """Tracks not yet removed, including Lost ones.

        A Lost track has no detection right now but can still be revived when
        the person reappears. Treating "not currently seen" as "dead" would
        fragment a track every time someone walks behind a pillar, which is the
        exact failure ByteTrack's track buffer exists to prevent.
        """
        return [t for t in self._tracks if t.state is not TrackState.Removed]

    def reset(self) -> None:
        self._tracks.clear()

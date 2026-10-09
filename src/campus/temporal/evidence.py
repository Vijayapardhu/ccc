"""Per-track evidence buffer.

The problem this solves
-----------------------
A single frame of a walking person is a coin flip. Across the track there are
usually a handful of frames where the face is frontal, sharp and well exposed,
and the identity is obvious in those and nowhere else. Searching on the first
frame throws those away; searching on the *current* frame means a person who
turns away mid-crossing stops matching, and any commitment made while they were
facing the camera gets contradicted by the next frame.

So the buffer keeps the best K observations by quality and the search runs
against those, not against whatever arrived last.

What it deliberately does not do
--------------------------------
It does not stitch partial faces together. A left-profile frame contains no
information about the side turned away from it, so composing one with a
right-profile frame is not reconstruction, it is averaging two views of
disjoint evidence. On the ArcFace hypersphere those vectors sit far apart and
their mean matches everyone slightly. Finding the frontal moment in the track
is strictly better and is what this does instead.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from campus.imaging.align import l2_normalize

# Buffers are small on purpose. K=8 spans about two seconds at the analysis
# rates in use, which is enough to contain several good frames and few enough
# that a long track cannot be dominated by its first minute.
DEFAULT_CAPACITY = 8


@dataclass(frozen=True, slots=True)
class Observation:
    """One quality-passing frame's contribution to a track."""

    timestamp: float
    embedding: npt.NDArray[np.float32]
    face_px: int
    blur: float
    yaw: float
    pitch: float
    roll: float
    brightness: float
    contrast: float
    det_score: float

    def score(self) -> float:
        """How usable this frame is as identification evidence, in [0, 1].

        Pose dominates on purpose. A pin-sharp 60-degree profile is worth less
        than a slightly soft frontal view, because profile views embed into a
        region the model was never trained to discriminate and the resulting
        vector matches strangers about as well as the student. Weighting
        sharpness first is the classic mistake here.

        Size saturates: past roughly 100px more pixels buy nothing, so the
        term rewards going from 20px to 60px and then flattens. That keeps a
        huge blurry face from outranking a smaller sharp one.
        """
        if self.face_px <= 0:
            return 0.0

        frontal = (
            max(0.0, 1.0 - abs(self.yaw) / 60.0)
            * max(0.0, 1.0 - abs(self.pitch) / 45.0)
            * max(0.0, 1.0 - abs(self.roll) / 50.0)
        )
        # Hard knee at 100px, not a soft asymptote. A saturating ratio like
        # px/(px+40) never actually reaches full marks, so a 2000px face still
        # edges out a 100px one for reasons that carry no signal. Past the knee
        # extra pixels are not better evidence, they are just more pixels.
        size = min(1.0, self.face_px / 100.0)
        # Sharp enough saturates at ~300 variance-of-Laplacian; below that it
        # scales linearly.
        sharp = min(1.0, self.blur / 300.0)
        # Exposure: full marks in the middle of the range, falling off toward
        # both ends, so a blown-out face does not outrank a correctly exposed one.
        exposure = _exposure_factor(self.brightness, self.contrast)

        return frontal * size * sharp * exposure * min(1.0, self.det_score / 0.8)


def _exposure_factor(brightness: float, contrast: float) -> float:
    if brightness < 30.0:
        return max(0.0, 1.0 - (30.0 - brightness) / 30.0)
    if brightness > 225.0:
        return max(0.0, 1.0 - (brightness - 225.0) / 30.0)
    if contrast < 25.0:
        return max(0.0, contrast / 25.0)
    return 1.0


@dataclass(slots=True)
class TrackEvidence:
    """Bounded, ranked store of the best observations for one track.

    Insertion is O(K) rather than a heap, which is deliberate: K is 8, so a
    heap's asymptotics buy nothing and cost readability.
    """

    capacity: int = DEFAULT_CAPACITY
    _items: deque[Observation] = field(default_factory=deque, repr=False)
    _seen: int = 0

    def add(self, obs: Observation) -> bool:
        """Insert, keeping only the best `capacity`. Returns True if kept."""
        self._seen += 1
        if len(self._items) >= self.capacity and obs.score() <= self.worst_score():
            return False
        self._items.append(obs)
        self._trim()
        return True

    def _trim(self) -> None:
        while len(self._items) > self.capacity:
            ranked = sorted(self._items, key=lambda o: o.score(), reverse=True)
            self._items.clear()
            self._items.extend(ranked[: self.capacity])

    def worst_score(self) -> float:
        return min((o.score() for o in self._items), default=0.0)

    def best(self) -> Observation | None:
        if not self._items:
            return None
        return max(self._items, key=lambda o: o.score())

    def best_n(self, n: int = 3) -> list[Observation]:
        return sorted(self._items, key=lambda o: o.score(), reverse=True)[:n]

    def centroid(self, n: int = 3) -> npt.NDArray[np.float32] | None:
        """Quality-weighted mean of the best N embeddings.

        Only ever called with frames of similar pose — the caller is expected
        to pass a pose window. Averaging across a 90-degree yaw swing is the
        thing this module exists to avoid, so the weighting is sharp enough
        that a well-ranked outlier cannot drag the centroid.
        """
        picks = self.best_n(n)
        if not picks:
            return None
        if len(picks) == 1:
            return picks[0].embedding
        weights = np.array([o.score() for o in picks], dtype=np.float32)
        total = float(weights.sum())
        if total <= 1e-9:
            return picks[0].embedding
        weights /= total
        stack = np.stack([o.embedding for o in picks])
        return l2_normalize((stack * weights[:, None]).sum(axis=0).astype(np.float32))

    def pose_spread(self) -> float:
        """Largest absolute yaw among buffered frames.

        If this is large the buffer holds views of quite different geometry and
        a centroid across them is not trustworthy; the caller should search with
        the single best frame instead.
        """
        if not self._items:
            return 0.0
        yaws = [o.yaw for o in self._items]
        return max(yaws) - min(yaws)

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def seen(self) -> int:
        return self._seen

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float | int]:
        best = self.best()
        return {
            "kept": len(self._items),
            "seen": self._seen,
            "best_score": round(best.score(), 4) if best else 0.0,
            "best_px": best.face_px if best else 0,
            "best_yaw": round(best.yaw, 1) if best else 0.0,
            "pose_spread": round(self.pose_spread(), 1),
        }

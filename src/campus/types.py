"""Domain types shared across the capture, inference, and identity planes.

Two planes, deliberately separated:

* **Capture** — one worker owns a disjoint slice of cameras. It never talks to
  the gallery. It emits observations (a face, an embedding, a track) and
  nothing else.
* **Identity** — a separate service owns the gallery and is the only component
  that can turn an observation into a named student.

Keeping the boundary here rather than in the transport is what makes the
horizontal scale-out safe: a worker crash loses camera coverage, never
identity integrity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NewType

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

CameraId = NewType("CameraId", str)
StudentId = NewType("StudentId", str)
TrackId = NewType("TrackId", str)
ClusterId = NewType("ClusterId", str)
# Proves an observation came from a worker we control. Unforgeable in practice
# because only workers hold the signing key.
WorkerToken = NewType("WorkerToken", str)


class TrackState(StrEnum):
    """Lifecycle of a single person crossing one camera's field of view.

    A track is scoped to a single camera. It is deliberately *not* a global
    identity — that is the whole point of the temporal verifier downstream.
    """

    NEW = "new"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


class ObservationOutcome(StrEnum):
    """Result of resolving one face observation against the gallery."""

    IDENTIFIED = "identified"
    """Same identity as the track's current commitment, with fresh evidence."""

    AMBIGUOUS = "ambiguous"
    """Several candidates in the same band; insufficient to commit either way."""

    UNKNOWN = "unknown"
    """No candidate cleared the floor. Face present, person not in gallery."""

    CONSENT_WITHHELD = "consent_withheld"
    """Student matched, but their consent policy forbids this camera/purpose."""

    IDENTITY_CHANGED = "identity_changed"
    """A sustained challenger took the track over from its committed subject.

    Emitted on the single frame the switch is detected, for alerting and
    telemetry. It does *not* produce a second presence event: presence rows are
    keyed on track_id, so a second commit would be a duplicate rather than a
    correction, and a duplicated attendance record is worse than one that is
    slightly stale."""


@dataclass(frozen=True, slots=True)
class FaceBox:
    """One detected face, in *source frame* pixel coordinates.

    Always source-frame, never tile-local. Tile-local coordinates are an
    implementation detail of the detector and must be projected back before a
    FaceBox escapes the capture worker.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    landmarks: IntArray | None = None
    """(5, 2) int array: left eye, right eye, nose, left mouth, right mouth."""

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def as_array(self) -> FloatArray:
        return np.array([self.x0, self.y0, self.x1, self.y1], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Why a face was kept or dropped, and by how much.

    Thresholds are deliberately visible on the report so an operator can
    explain a miss instead of staring at a boolean.
    """

    face_px: int
    blur_score: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    brightness: float
    contrast: float
    occlusion_ratio: float
    passed: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_px": self.face_px,
            "blur_score": round(self.blur_score, 2),
            "yaw_deg": round(self.yaw_deg, 1),
            "pitch_deg": round(self.pitch_deg, 1),
            "roll_deg": round(self.roll_deg, 1),
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "occlusion_ratio": round(self.occlusion_ratio, 3),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """A quality-passing face with its embedding, awaiting identity resolution.

    `embedding` is always L2-normalised (float32, shape (D,)), which makes
    cosine similarity a plain dot product and lets the index use inner-product
    search without a normalisation pass at query time.
    """

    camera_id: CameraId
    frame_seq: int
    timestamp: float
    box: FaceBox
    embedding: FloatArray
    quality: QualityReport
    track_id: TrackId | None = None
    worker_id: str = ""
    tile_count: int = 1
    """How many tiles produced this box. >1 means it sat on a tile seam."""


@dataclass(frozen=True, slots=True)
class GalleryCandidate:
    student_id: StudentId
    score: float
    """Cosine similarity in [-1, 1]. ArcFace siblings typically land 0.3-0.6,
    unrelated pairs 0.0-0.2. Single-frame thresholds must sit well below the
    sibling band, because a single frame is genuinely ambiguous."""

    rank: int
    enrolled_photos: int = 1
    """Gallery records built from more than one enrollment photo are more
    reliable; the resolver uses this to break near-ties."""


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """The system's answer for one observation, with its full evidence chain."""

    outcome: ObservationOutcome
    camera_id: CameraId
    frame_seq: int
    timestamp: float
    track_id: TrackId | None
    student_id: StudentId | None
    score: float
    candidates: tuple[GalleryCandidate, ...]
    quality: QualityReport
    consent_id: str | None = None
    reason: str = ""

    @property
    def is_identified(self) -> bool:
        return self.outcome in (ObservationOutcome.IDENTIFIED, ObservationOutcome.CONSENT_WITHHELD)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "camera_id": self.camera_id,
            "frame_seq": self.frame_seq,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "student_id": self.student_id,
            "score": round(self.score, 4),
            "candidates": [
                {"student_id": c.student_id, "score": round(c.score, 4), "rank": c.rank}
                for c in self.candidates
            ],
            "quality": self.quality.to_dict(),
            "consent_id": self.consent_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TrackCommit:
    """A decision the temporal verifier is willing to hold for a whole track.

    Emitted once per track, when evidence has accumulated past the commit
    threshold. Downstream consumers should treat this — not per-frame
    decisions — as the fact that a student was present.
    """

    track_id: TrackId
    camera_id: CameraId
    student_id: StudentId
    first_seen: float
    last_seen: float
    committed_at: float
    evidence_frames: int
    window_frames: int
    median_score: float
    min_score: float
    mean_score: float
    score_std: float
    runner_up_id: StudentId | None
    runner_up_score: float
    margin: float
    duration_s: float
    quality_frames: int
    consent_id: str | None = None
    suppressed: bool = False
    """True when policy requires the event to be logged without a student_id."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "student_id": self.student_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "committed_at": self.committed_at,
            "evidence_frames": self.evidence_frames,
            "window_frames": self.window_frames,
            "median_score": round(self.median_score, 4),
            "min_score": round(self.min_score, 4),
            "score_std": round(self.score_std, 4),
            "runner_up_id": self.runner_up_id,
            "runner_up_score": round(self.runner_up_score, 4),
            "margin": round(self.margin, 4),
            "duration_s": round(self.duration_s, 3),
            "quality_frames": self.quality_frames,
            "consent_id": self.consent_id,
            "suppressed": self.suppressed,
        }


@dataclass(frozen=True, slots=True)
class EnrollmentRecord:
    """One student's gallery entry.

    A student may hold several photos (front, slight left, slight right); they
    are stored as separate records sharing `student_id` and are averaged at
    index build time into a single centroid, with the originals kept for
    re-scoring and debugging.
    """

    student_id: StudentId
    embedding: FloatArray
    photo_id: str
    angle: str = "front"
    source: str = "id_card"
    enrolled_at: float = field(default_factory=time.time)
    quality: QualityReport | None = None


@dataclass(frozen=True, slots=True)
class ConsentDecision:
    """Result of evaluating DPDP consent for a would-be identification."""

    permitted: bool
    consent_id: str | None
    purposes: frozenset[str]
    reason: str = ""

    def allows(self, purpose: str) -> bool:
        return self.permitted and purpose in self.purposes


@dataclass(slots=True)
class PresenceEvent:
    """The fact that a student was seen at a camera at a time.

    This is the system's primary output and the record the university would
    need to produce on a DPDP erasure request, so it carries `event_id` for
    exactly-once deduplication and `retained_until` for automatic expiry.
    """

    event_id: str
    student_id: StudentId
    camera_id: CameraId
    zone: str
    first_seen: float
    last_seen: float
    track_id: TrackId
    confidence: float
    purposes: tuple[str, ...] = ()
    redacted: bool = False
    retained_until: float = 0.0

    @classmethod
    def new_id(cls) -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "student_id": self.student_id,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
            "purposes": list(self.purposes),
            "redacted": self.redacted,
            "retained_until": self.retained_until,
        }

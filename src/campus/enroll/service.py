"""Enrollment: turning photos into a gallery.

The single biggest accuracy lever in the whole system. A student enrolled
from a poor ID-card photo will have a permanently weak match rate, and no
amount of downstream tuning recovers it — the gallery entry is the ceiling.

So enrollment is the most gated path in the codebase: detect with the *large*
SCRFD variant, align, check against a much stricter quality bar than live
detection, and keep the original photo. A re-enrollment station that captures
front / left / right under controlled lighting roughly halves the false-match
rate versus a single ID-card photo, and it is cheap compared to a semester of
attendance errors.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from campus.capture.source import frame_gray
from campus.imaging.align import align
from campus.imaging.quality import assess
from campus.models.arcface import ArcFaceEmbedder, EnrollmentQuality
from campus.models.scrfd import InsufficientFacesError, ScrfdDetector, single_face
from campus.types import EnrollmentRecord, QualityReport, StudentId

log = logging.getLogger("campus.enrollment")

VALID_ANGLES = ("front", "left", "right", "up", "down")


@dataclass(slots=True)
class RejectedEnrollment:
    student_id: str
    photo_path: str
    reasons: list[str]
    quality: QualityReport | None = None


@dataclass(slots=True)
class EnrollmentReport:
    accepted: list[EnrollmentRecord] = field(default_factory=list)
    rejected: list[RejectedEnrollment] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def summary(self) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "rejections": [
                {"photo": r.photo_path, "reasons": r.reasons} for r in self.rejected
            ],
        }


class EnrollError(RuntimeError):
    pass


class EnrollmentService:
    """Builds gallery records from photos.

    Uses a separate detector instance from the live pipeline: the 10G model is
    ~3x slower and materially better on the small, off-centre, badly-lit faces
    that ID-card photos tend to be. Enrollment runs in seconds per student;
    live detection runs 3200 times a second.
    """

    def __init__(
        self,
        detector: ScrfdDetector,
        embedder: ArcFaceEmbedder,
        quality: EnrollmentQuality | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.quality = quality or EnrollmentQuality()

    def enroll_photo(
        self,
        student_id: str,
        image: npt.NDArray[np.uint8],
        angle: str = "front",
        source: str = "id_card",
    ) -> EnrollmentRecord:
        """Enroll a single photo. Raises on any quality failure.

        Raises rather than returning a partial result: a student being
        enrolled must either land in the gallery correctly or not at all. A
        silently-degraded enrollment is a false-match source that surfaces
        weeks later as an attendance error nobody can explain.
        """
        if angle not in VALID_ANGLES:
            raise EnrollError(f"invalid angle {angle!r}, expected one of {VALID_ANGLES}")

        detections = self.detector.detect(image)
        try:
            box = single_face(detections, min_score=0.6)
        except InsufficientFacesError as exc:
            raise EnrollError(
                f"{student_id} {angle}: {exc}. Photo must contain exactly one clear face."
            ) from exc

        report = assess(frame_gray(image), box)
        passed, reasons = self.quality.check(report)
        if not passed:
            raise EnrollError(f"{student_id} {angle}: {', '.join(reasons)}")

        aligned = align(image, box, self.embedder.config.input_size)
        if aligned is None:
            raise EnrollError(
                f"{student_id} {angle}: alignment failed. The face was too oblique "
                f"or the landmarks were degenerate. Ask for a more frontal photo."
            )

        vector = self.embedder.embed([aligned])[0]
        return EnrollmentRecord(
            student_id=StudentId(student_id),
            embedding=vector,
            photo_id=str(uuid.uuid4()),
            angle=angle,
            source=source,
            enrolled_at=time.time(),
            quality=report,
        )

    def enroll_directory(
        self,
        student_id: str,
        directory: str | Path,
        angles: Sequence[str] = ("front", "left", "right"),
        source: str = "enrollment_station",
    ) -> EnrollmentReport:
        """Enroll from a folder of photos, one per angle.

        Missing angles are skipped, not fatal — a single good front photo is
        far better than nothing, and the centroid degrades gracefully with two
        or three.
        """
        root = Path(directory)
        if not root.is_dir():
            raise EnrollError(f"not a directory: {root}")

        report = EnrollmentReport()
        for angle in angles:
            candidates = sorted(
                [*root.glob(f"{angle}.*"), *root.glob(f"{angle}_*.*")]
            )
            if not candidates:
                log.info("%s: no %s photo, skipping", student_id, angle)
                continue
            path = candidates[0]
            image = cv2.imread(str(path))
            if image is None:
                report.rejected.append(
                    RejectedEnrollment(student_id, str(path), ["unreadable image"])
                )
                continue
            try:
                report.accepted.append(
                    self.enroll_photo(student_id, image, angle=angle, source=source)
                )
            except EnrollError as exc:
                report.rejected.append(
                    RejectedEnrollment(student_id, str(path), [str(exc)])
                )

        if not report.accepted and report.rejected:
            raise EnrollError(
                f"{student_id}: no usable photos. "
                + "; ".join(r.reasons[0] for r in report.rejected)
            )
        return report

    def enroll_from_camera(
        self,
        student_id: str,
        image: npt.NDArray[np.uint8],
        angle: str = "front",
    ) -> tuple[EnrollmentRecord, npt.NDArray[np.uint8]]:
        """Enroll from an enrollment-station camera frame.

        Returns the record *and* the aligned crop, so the station can show the
        operator what was actually enrolled. Without that preview, subjects
        routinely capture a frame where they blinked or turned and do not
        notice until the match rate is bad weeks later.
        """
        detections = self.detector.detect(image)
        box = single_face(detections, min_score=0.6)
        report = assess(frame_gray(image), box)
        passed, reasons = self.quality.check(report)
        if not passed:
            raise EnrollError(f"{student_id}: {', '.join(reasons)}")
        aligned = align(image, box, self.embedder.config.input_size)
        if aligned is None:
            raise EnrollError(f"{student_id}: alignment failed")
        vector = self.embedder.embed([aligned])[0]
        return (
            EnrollmentRecord(
                student_id=StudentId(student_id),
                embedding=vector,
                photo_id=str(uuid.uuid4()),
                angle=angle,
                source="enrollment_station",
                enrolled_at=time.time(),
                quality=report,
            ),
            aligned,
        )

    @staticmethod
    def verify_consistency(
        records: Sequence[EnrollmentRecord], min_pairwise: float = 0.25
    ) -> list[str]:
        """Warn when a student's own photos disagree.

        If the front, left and right photos of one person do not match each
        other, the cause is almost always capture (bad lighting, wrong subject,
        hair covering the face) rather than the model. Surfacing it at
        enrollment is the only chance to fix it.
        """
        warnings: list[str] = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                sim = float(np.dot(a.embedding, b.embedding))
                if sim < min_pairwise:
                    warnings.append(
                        f"{a.student_id}: {a.angle} vs {b.angle} similarity "
                        f"{sim:.3f} < {min_pairwise}. Re-capture one of them."
                    )
        return warnings

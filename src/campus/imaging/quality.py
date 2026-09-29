"""Face quality gating.

The job here is to spend the embedding budget only on faces that can produce a
trustworthy vector. Embedding is the expensive step in the pipeline; a quality
filter that throws away 80% of detections before ArcFace is a straight 5x
speedup, and it improves accuracy because ArcFace is trained on well-aligned,
well-exposed faces and degrades predictably outside that distribution.

Order matters. Size is checked first because it is free, and a 12px face will
fail every other check anyway. Pose is checked next because a 50-degree yaw
face produces a garbage embedding no matter how sharp it is.

Every threshold is a named field rather than a literal, so the tuning
conversation with the university is about specific knobs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from campus.types import FaceBox, QualityReport

# Landmark geometry of the ArcFace template, in units of interocular distance
# (eye-to-eye), with the eye line as the origin. Calibrating on the template
# rather than on absolute pixels is what makes the pitch estimate comparable
# between a 30px corridor face and a 400px portrait.
_NOSE_Y = 0.570
_MOUTH_Y = 1.155
_NOMINAL_FILL = 0.204
"""Convex-hull area of the five landmarks as a fraction of the detection box,
for a well-formed frontal face. Below this, the landmarks have been pulled off
the face by occlusion or by a bad detection."""


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Gating thresholds. Tuned for 4K CCTV, not studio portraits.

    The defaults are deliberately permissive on size and strict on pose. Campus
    cameras see people at 20-40px routinely and we still want them; but a
    profile view is unrecoverable at any size.
    """

    min_face_px: int = 24
    """Below this, ArcFace embeddings are noise. Derived from measurement on our
    own footage, not from the paper: at 24px the inter-arc error is comparable
    to the inter-class gap between siblings."""

    min_face_ratio: float = 0.015
    """Face must also be at least this fraction of the frame's short side, to
    reject false positives on distant clutter."""

    min_blur_score: float = 45.0
    """Variance of the Laplacian. Cameras with high JPEG quality on motion
    produce 60+; a smeared face lands under 20."""

    max_blur_score: float = 900.0
    """Conversely, a very high score means noise amplification in low light,
    which is its own kind of unusable."""

    max_yaw_deg: float = 45.0
    max_pitch_deg: float = 35.0
    max_roll_deg: float = 40.0

    min_brightness: float = 25.0
    max_brightness: float = 235.0
    min_contrast: float = 12.0
    """Standard deviation of the luma crop. Flat crops (overexposed wall,
    crushed shadow) carry no identity information."""

    max_occlusion_ratio: float = 0.42
    """Fraction of the face box covered by the convex hull of the five
    landmarks' 'absence' — approximated below by landmark spread. A face with a
    mask or a phone across it scores high here."""


def _variance_of_laplacian(gray: npt.NDArray[np.uint8]) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_pose(
    landmarks: npt.NDArray[np.float32] | None,
    box: FaceBox,
) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees from 5 landmarks.

    This is a geometric estimate, not a PnP solve. It is fast enough to run on
    every detection, needs no camera calibration, and is accurate enough to
    answer the only question we ask of it: *is this face too far off-axis to
    embed usefully?* It is not accurate enough to reconstruct a head, and
    should not be used for that.

    Returns (0, 0, 0) when landmarks are unavailable, which lets the caller
    treat pose as "unknown" rather than "frontal" by checking for zeros only if
    it truly needs to.
    """
    if landmarks is None or len(landmarks) < 5:
        return (0.0, 0.0, 0.0)

    le, re, nose, lm, rm = (np.asarray(landmarks[i], dtype=np.float64) for i in range(5))

    eye_mid = (le + re) / 2.0
    eye_vec = re - le
    interocular = float(np.hypot(*eye_vec))
    if interocular < 1e-3:
        return (0.0, 0.0, 0.0)

    # Roll: orientation of the eye axis. Cheap and reliable.
    roll = math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0])))

    # Normalised lateral offset of the nose from the eye midpoint. 0.0 is
    # frontal; the value grows toward the edge of the face as yaw increases.
    yaw_unit = float(np.dot(nose - eye_mid, eye_vec) / (interocular**2))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, yaw_unit * 1.6))))

    # Pitch proxy: where the nose sits between the eye line and the mouth line.
    # Looking up raises the nose toward the eyes; looking down lowers it.
    mouth_mid = (lm + rm) / 2.0
    span = float(np.linalg.norm(mouth_mid - eye_mid))
    if span < 1e-3:
        return (yaw, 0.0, roll)
    depth = float(np.dot(nose - eye_mid, mouth_mid - eye_mid) / (span**2))
    expected = _NOSE_Y / _MOUTH_Y
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, (depth - expected) * 2.0))))

    return (yaw, pitch, roll)


def occlusion_ratio(landmarks: npt.NDArray[np.float32] | None, box: FaceBox) -> float:
    """Cheap occlusion proxy in [0, 1]. 0.0 is a clean frontal face.

    A frontal face's five landmarks fill a consistent fraction of its bounding
    box. When something covers the face — a mask, a phone, a backpack strap,
    another person's head in a dense crowd — the landmarks collapse toward a
    corner or spread asymmetrically and that fraction drops.

    This is a heuristic and it is noisy. It is used only to reject the clearly
    bad, and the temporal verifier absorbs the rest.
    """
    if landmarks is None or len(landmarks) < 5:
        return 1.0
    area = float(box.area)
    if area <= 0:
        return 1.0
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2)
    hull_area = float(abs(cv2.contourArea(cv2.convexHull(pts))))
    fill = min(1.0, hull_area / area)
    return float(max(0.0, min(1.0, 1.0 - fill / _NOMINAL_FILL)))


def assess(
    frame_gray: npt.NDArray[np.uint8],
    box: FaceBox,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Score one face against `thresholds`.

    `frame_gray` must be the *full* grayscale frame, not a tile crop, so that
    brightness statistics reflect the camera's actual conditions.
    """
    t = thresholds or QualityThresholds()
    reasons: list[str] = []

    fh, fw = frame_gray.shape[:2]
    face_px = box.min_side

    # --- Size (cheapest, and a hard prerequisite for everything below) ---
    if face_px < t.min_face_px:
        reasons.append(f"face_px {face_px} < {t.min_face_px}")
    short_side = min(fh, fw)
    if short_side > 0 and face_px / short_side < t.min_face_ratio:
        reasons.append(f"face_ratio {face_px / short_side:.4f} < {t.min_face_ratio}")

    x0, y0 = max(0, box.x0), max(0, box.y0)
    x1, y1 = min(fw, box.x1), min(fh, box.y1)
    if x1 <= x0 or y1 <= y0:
        return QualityReport(
            face_px=face_px,
            blur_score=0.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            brightness=0.0,
            contrast=0.0,
            occlusion_ratio=1.0,
            passed=False,
            reasons=(*reasons, "empty_crop"),
        )

    crop = frame_gray[y0:y1, x0:x1]

    # --- Sharpness ---
    blur = _variance_of_laplacian(crop)
    if blur < t.min_blur_score:
        reasons.append(f"blur {blur:.1f} < {t.min_blur_score}")
    elif blur > t.max_blur_score:
        reasons.append(f"blur {blur:.1f} > {t.max_blur_score} (noise)")

    # --- Illumination ---
    brightness = float(crop.mean())
    contrast = float(crop.std())
    if brightness < t.min_brightness:
        reasons.append(f"dark {brightness:.0f} < {t.min_brightness}")
    elif brightness > t.max_brightness:
        reasons.append(f"blown {brightness:.0f} > {t.max_brightness}")
    if contrast < t.min_contrast:
        reasons.append(f"flat contrast {contrast:.1f} < {t.min_contrast}")

    # --- Pose ---
    yaw, pitch, roll = estimate_pose(box_landmarks(box), box)
    if abs(yaw) > t.max_yaw_deg:
        reasons.append(f"yaw {yaw:.0f} > {t.max_yaw_deg}")
    if abs(pitch) > t.max_pitch_deg:
        reasons.append(f"pitch {pitch:.0f} > {t.max_pitch_deg}")
    if abs(roll) > t.max_roll_deg:
        reasons.append(f"roll {roll:.0f} > {t.max_roll_deg}")

    occ = occlusion_ratio(box_landmarks(box), box)
    if occ > t.max_occlusion_ratio:
        reasons.append(f"occlusion {occ:.2f} > {t.max_occlusion_ratio}")

    return QualityReport(
        face_px=face_px,
        blur_score=blur,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        brightness=brightness,
        contrast=contrast,
        occlusion_ratio=occ,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def box_landmarks(box: FaceBox) -> npt.NDArray[np.float32] | None:
    return None if box.landmarks is None else box.landmarks.astype(np.float32)


def assess_many(
    frame_gray: npt.NDArray[np.uint8],
    boxes: Sequence[FaceBox],
    thresholds: QualityThresholds | None = None,
) -> list[QualityReport]:
    return [assess(frame_gray, b, thresholds) for b in boxes]

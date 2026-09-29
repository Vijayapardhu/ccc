"""Shared fixtures and synthetic-data builders.

Tests must not require a GPU, a camera, or model weights, so every fixture
here is synthetic. The one deliberate exception is `realistic_frame`, which
draws a schematic dense crowd — enough to exercise the tiling and quality
paths end to end without pretending to be a face-recognition benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from campus.types import FaceBox, GalleryCandidate, StudentId


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260929)


def frontal_landmarks(cx: float, cy: float, size: float) -> np.ndarray:
    """Landmarks for a frontal face, matching the ArcFace template's geometry.

    The eye line sits at ``cy`` and the interocular distance is 0.44 * size.
    Every offset is expressed in interocular units taken from
    ``ARCFACE_TEMPLATE``, so the pose estimator's calibration (which is
    derived from that same template) reads a synthetic frontal face as exactly
    zero yaw, zero pitch, zero roll. If the two drift apart, every pose
    assertion in the suite becomes meaningless.

    Returns left eye, right eye, nose, left mouth, right mouth.
    """
    s = size
    d = 0.44 * s
    return np.array(
        [
            [cx - 0.500 * d, cy],
            [cx + 0.500 * d, cy],
            [cx, cy + 0.570 * d],
            [cx - 0.414 * d, cy + 1.155 * d],
            [cx + 0.414 * d, cy + 1.155 * d],
        ],
        dtype=np.int32,
    )


def face_box(cx: float, cy: float, size: float, score: float = 0.9) -> FaceBox:
    """A FaceBox with frontal landmarks, the common case in tests."""
    half = size / 2.0
    return FaceBox(
        x0=int(cx - half),
        y0=int(cy - half),
        x1=int(cx + half),
        y1=int(cy + half),
        score=score,
        landmarks=frontal_landmarks(cx, cy, size),
    )


def crowd_frame(width: int, height: int, count: int, min_px: int = 28, max_px: int = 60):
    """A blank frame plus a grid of synthetic face boxes.

    Positions are deterministic and grid-aligned so a test can assert an exact
    count through the tile/NMS path.
    """
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cols = max(1, int(np.ceil(np.sqrt(count * width / height))))
    rows = max(1, int(np.ceil(count / cols)))
    boxes: list[FaceBox] = []
    for i in range(count):
        r, c = divmod(i, cols)
        size = min_px + (i % 5) * (max_px - min_px) // 4
        cx = (c + 0.5) * width / cols
        cy = (r + 0.5) * height / rows
        if cx - size / 2 < 0 or cx + size / 2 > width:
            continue
        if cy - size / 2 < 0 or cy + size / 2 > height:
            continue
        boxes.append(face_box(cx, cy, size, score=0.6 + 0.05 * (i % 8)))
    return frame, boxes


def _structured_field(width: int, height: int, seed: int) -> np.ndarray:
    """Mid-frequency luminance field with structure at roughly face-feature scale.

    Deliberately not white noise. Uniform random pixels have an enormous
    variance-of-Laplacian, which the quality gate correctly rejects as
    noise-amplified low light — so a noise-based "sharp" fixture would fail
    the gate it is supposed to pass, and the sharpness tests would be testing
    the wrong thing.
    """
    import cv2

    rng = np.random.default_rng(seed)
    block = 4
    small = rng.integers(40, 200, size=(height // block + 2, width // block + 2)).astype(np.float32)
    field = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = field * 0.75 + 35.0 + 42.0 * (xx / max(1, width - 1))
    return np.clip(field, 0, 255).astype(np.uint8)


def sharp_image(width: int, height: int, seed: int = 7) -> np.ndarray:
    """A well-exposed, in-focus BGR frame. Passes every quality gate."""
    import cv2

    gray = _structured_field(width, height, seed)
    return cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2BGR)


def blurry_image(width: int, height: int, seed: int = 7) -> np.ndarray:
    """The same scene defocused. Rejected by the blur gate, as a real one would be."""
    import cv2

    gray = _structured_field(width, height, seed)
    return cv2.cvtColor(
        np.ascontiguousarray(cv2.GaussianBlur(gray, (0, 0), 4.0)),
        cv2.COLOR_GRAY2BGR,
    )


def candidates(pairs: list[tuple[str, float]]) -> list[GalleryCandidate]:
    """Build a ranked candidate list from ``(student_id, score)`` pairs."""
    return [
        GalleryCandidate(student_id=StudentId(sid), score=score, rank=i)
        for i, (sid, score) in enumerate(pairs, start=1)
    ]


def consistent_candidates(
    student: str, score: float = 0.55, rival: str = "9999", rival_score: float = 0.30
) -> list[GalleryCandidate]:
    return candidates([(student, score), (rival, rival_score)])

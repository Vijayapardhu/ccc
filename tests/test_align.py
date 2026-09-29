"""Alignment to the ArcFace template."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from campus.imaging.align import (
    ARCFACE_TEMPLATE,
    align,
    align_batch,
    estimate_arc,
    is_mirrored,
    l2_normalize,
    umeyama_similarity,
)
from campus.types import FaceBox

from .conftest import frontal_landmarks


def _unconstrained_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama *without* the determinant sign correction, for fixture design.

    Exists so a test can assert that its input genuinely exercises the guard
    rather than silently passing because the correction was never needed.
    """
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    rot = u @ vt
    var_src = float((src_c**2).sum() / len(src))
    scale = 1.0 if var_src < 1e-12 else float(s.sum() / var_src)
    return scale * rot


def box_at(cx: float, cy: float, size: float) -> FaceBox:
    return FaceBox(
        x0=int(cx - size / 2), y0=int(cy - size / 2),
        x1=int(cx + size / 2), y1=int(cy + size / 2),
        score=0.9, landmarks=frontal_landmarks(cx, cy, size),
    )


class TestUmeyama:
    def test_recovers_known_similarity_transform(self):
        """The whole method: given a known rotation+scale+translation, the
        solver must invert it."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=np.float64)
        theta = np.radians(20)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dst = (2.5 * (src @ rot.T)) + np.array([37.0, 11.0])
        m = umeyama_similarity(src, dst)
        recovered = (src @ m[:, :2].T) + m[:, 2]
        assert np.allclose(recovered, dst, atol=1e-6)

    def test_does_not_reflect(self):
        """The SVD sign fix keeps the solver from returning a mirror.

        A mirrored transform produces embeddings that match nothing while
        still looking like valid numbers, so the guard is asserted directly on
        the solver's output for a configuration whose best *unconstrained* fit
        is a reflection. The fixture is a near-mirror (one axis compressed), so
        the corrected solution still has positive scale — for an *exact* mirror
        the correction correctly collapses the scale to zero, and that
        degeneracy is covered separately by `test_mirrored_landmarks_are_rejected`.
        """
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=np.float64)
        near_mirror = src * np.array([-0.5, 1.0])
        assert np.linalg.det(_unconstrained_fit(src, near_mirror)) < 0, (
            "fixture no longer exercises the guard"
        )
        m = umeyama_similarity(src, near_mirror)
        assert np.linalg.det(m[:, :2]) > 0, "solver returned a reflection"
        assert float(np.hypot(m[0, 0], m[1, 0])) > 1e-6, "guard collapsed the scale"

    def test_mirrored_landmarks_are_rejected(self):
        """A mirror is a congruence, not a similarity, so no valid transform
        exists — the solver would return a plausible-looking wrong one.
        Detected geometrically instead, before solving."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = frontal_landmarks(320, 240, 120)
        pts[:, 0] = 640 - pts[:, 0]
        assert is_mirrored(pts)
        box = FaceBox(260, 180, 380, 300, 0.9, pts)
        assert align(frame, box) is None

    def test_correctly_ordered_landmarks_pass_the_mirror_check(self):
        assert not is_mirrored(frontal_landmarks(320, 240, 120))

    def test_shape_is_2x3(self):
        src = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float64)
        assert umeyama_similarity(src, src * 2).shape == (2, 3)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            umeyama_similarity(np.zeros((3, 2)), np.zeros((4, 2)))
        with pytest.raises(ValueError):
            umeyama_similarity(np.zeros((1, 2)), np.zeros((1, 2)))
        with pytest.raises(ValueError):
            umeyama_similarity(np.zeros((3, 3)), np.zeros((3, 3)))


class TestAlign:
    def test_produces_112x112_bgr(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        out = align(frame, box_at(1920, 1080, 200))
        assert out is not None
        assert out.shape == (112, 112, 3)
        assert out.dtype == np.uint8

    def test_lands_the_face_in_the_template(self):
        """A white disc drawn at the face centre must land at the matching
        point of the ArcFace template after warping. This is the actual
        correctness criterion — a crop that is merely the right *size* proves
        nothing, and a crop from the wrong part of the frame is exactly the
        failure that produces valid-looking embeddings matching nobody.

        The synthetic landmarks put the eye line at ``cy``, so the disc centre
        maps to the template's eye-line midpoint, (55.9, 51.6).
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cx, cy, size = 320, 240, 120
        cv2.circle(frame, (int(cx), int(cy)), int(size * 0.2), (255, 255, 255), -1)
        out = align(frame, box_at(cx, cy, size))
        assert out is not None

        ys, xs = np.where(out[:, :, 0] > 128)
        assert len(xs) > 100, "the face was warped out of the template"
        assert abs(xs.mean() - 55.9) < 2.0, f"x centroid {xs.mean():.1f} != 55.9"
        assert abs(ys.mean() - 51.6) < 2.0, f"y centroid {ys.mean():.1f} != 51.6"
        # The template corners are background; the warp must not fill them.
        assert out[4, 4].mean() < 40

    def test_translation_invariance(self):
        """The same face at two frame positions must produce the same crop.
        A camera-height change must not change the embedding."""
        a = np.zeros((480, 640, 3), dtype=np.uint8)
        b = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(a, (200, 200), 30, (255, 255, 255), -1)
        cv2.circle(b, (900, 700), 30, (255, 255, 255), -1)
        out_a = align(a, box_at(200, 200, 120))
        out_b = align(b, box_at(900, 700, 120))
        assert out_a is not None and out_b is not None
        assert np.abs(out_a.astype(int) - out_b.astype(int)).mean() < 2.0

    def test_scale_invariance(self):
        """A face at 60px and the same face at 240px should align to nearly
        the same template. This is what makes a 20px corridor face match a
        400px portrait enrollment."""
        small = np.zeros((480, 640, 3), dtype=np.uint8)
        large = np.zeros((960, 1280, 3), dtype=np.uint8)
        cv2.circle(small, (320, 240), 6, (255, 255, 255), -1)
        cv2.circle(large, (640, 480), 24, (255, 255, 255), -1)
        out_s = align(small, box_at(320, 240, 60))
        out_l = align(large, box_at(640, 480, 240))
        assert out_s is not None and out_l is not None
        assert np.abs(out_s.astype(int) - out_l.astype(int)).mean() < 2.0

    def test_missing_landmarks_returns_none(self):
        """Not a fallback crop: an unaligned crop embeds into a region ArcFace
        never saw, and the resulting match scores look plausible while wrong."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = FaceBox(260, 180, 380, 300, 0.9, None)
        assert align(frame, box) is None

    def test_exact_mirror_collapses_the_transform(self):
        """No valid similarity exists for an exact mirror, so the corrected fit
        degenerates to scale 0. `align` must reject that rather than warp the
        frame flat and embed a constant image."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=np.float64)
        m = umeyama_similarity(src, src * np.array([-1.0, 1.0]))
        assert float(np.hypot(m[0, 0], m[1, 0])) < 1e-6

    def test_degenerate_landmarks_return_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = FaceBox(0, 0, 100, 100, 0.9, np.zeros((5, 2), dtype=np.int32))
        assert align(frame, box) is None

    def test_custom_output_size(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = align(frame, box_at(320, 240, 120), output_size=224)
        assert out is not None
        assert out.shape == (224, 224, 3)


class TestAlignBatch:
    def test_reports_which_faces_succeeded(self):
        """Index alignment matters: a silent pad would shift every row against
        its box and embed the wrong person's face."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [
            box_at(200, 200, 80),
            FaceBox(0, 0, 10, 10, 0.9, None),  # will fail
            box_at(400, 300, 80),
        ]
        crops, indices = align_batch(frame, boxes)
        assert len(crops) == 2
        assert indices == [0, 2]

    def test_all_fail_returns_empty(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crops, indices = align_batch(frame, [FaceBox(0, 0, 10, 10, 0.9, None)])
        assert crops == []
        assert indices == []


class TestNormalise:
    def test_produces_unit_vectors(self):
        v = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
        assert float(np.linalg.norm(v[0])) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_stays_zero(self):
        """Not NaN. A NaN in a gallery index silently corrupts neighbours."""
        v = l2_normalize(np.zeros((1, 4), dtype=np.float32))
        assert np.isfinite(v).all()
        assert np.all(v == 0)

    def test_batch_with_one_zero_row(self):
        v = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        assert float(np.linalg.norm(v[0])) == pytest.approx(1.0, abs=1e-6)
        assert np.all(v[1] == 0)


class TestArc:
    def test_identical_is_zero_degrees(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert estimate_arc(v, v) == pytest.approx(0.0, abs=1e-3)

    def test_orthogonal_is_90_degrees(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert estimate_arc(a, b) == pytest.approx(90.0, abs=1e-2)

    def test_opposite_is_180_degrees(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert estimate_arc(a, b) == pytest.approx(180.0, abs=1e-2)

    def test_no_nan_at_the_boundary(self):
        """arccos of a dot product very slightly over 1.0 must clamp, not NaN —
        exactly where the interesting decisions sit."""
        a = np.array([1.0, 1e-8, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.isfinite(estimate_arc(a, b))

    def test_zero_vector_is_maximum_distance(self):
        z = np.zeros(3, dtype=np.float32)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert estimate_arc(z, v) == 180.0

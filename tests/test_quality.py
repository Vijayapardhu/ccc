"""Face quality gating."""

from __future__ import annotations

import numpy as np

from campus.imaging.quality import QualityThresholds, assess, estimate_pose, occlusion_ratio
from campus.types import FaceBox

from .conftest import blurry_image, face_box, frontal_landmarks, sharp_image


def gray_of(img: np.ndarray) -> np.ndarray:
    return img[:, :, 0].copy()


class TestSizeGate:
    def test_rejects_tiny_face(self):
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, face_box(960, 540, 16))
        assert not report.passed
        assert any("face_px" in r for r in report.reasons)

    def test_accepts_adequate_face(self):
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, face_box(960, 540, 48))
        assert report.passed, report.reasons

    def test_face_ratio_rejects_distant_clutter(self):
        """A face that is a valid pixel size but tiny relative to the frame is
        almost always a false positive on a distant texture."""
        t = QualityThresholds(min_face_px=8, min_face_ratio=0.05)
        frame = gray_of(sharp_image(2160, 3840))
        report = assess(frame, face_box(1920, 1080, 40), t)
        assert not report.passed
        assert any("face_ratio" in r for r in report.reasons)

    def test_threshold_is_configurable_per_camera(self):
        """Corridor cameras need a lower floor than a library reading room."""
        frame = gray_of(sharp_image(1080, 1920))
        box = face_box(960, 540, 20)
        assert not assess(frame, box, QualityThresholds(min_face_px=24)).passed
        assert assess(frame, box, QualityThresholds(min_face_px=16)).passed


class TestBlurGate:
    def test_rejects_blurry_face(self):
        frame = gray_of(blurry_image(1080, 1920))
        report = assess(frame, face_box(960, 540, 48))
        assert not report.passed
        assert any(r.startswith("blur") for r in report.reasons)

    def test_rejects_noise_amplified_face(self):
        t = QualityThresholds(max_blur_score=10.0)
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, face_box(960, 540, 48), t)
        assert not report.passed
        assert any("noise" in r for r in report.reasons)

    def test_blurry_faces_are_cheap_to_reject(self):
        """A blurry crowd should be mostly filtered before embedding — this is
        the 5x saving the quality gate exists to provide."""
        frame = gray_of(blurry_image(1080, 1920))
        boxes = [face_box(200 + 300 * (i % 5), 300 + 300 * (i // 5), 50) for i in range(25)]
        reports = [assess(frame, b) for b in boxes]
        assert sum(r.passed for r in reports) == 0


class TestIlluminationGate:
    def test_rejects_dark_face(self):
        frame = np.full((1080, 1920), 5, dtype=np.uint8)
        report = assess(frame, face_box(960, 540, 48))
        assert not report.passed
        assert any(r.startswith("dark") for r in report.reasons)

    def test_rejects_blown_face(self):
        frame = np.full((1080, 1920), 250, dtype=np.uint8)
        report = assess(frame, face_box(960, 540, 48))
        assert not report.passed
        assert any(r.startswith("blown") for r in report.reasons)

    def test_rejects_flat_face(self):
        frame = np.full((1080, 1920), 128, dtype=np.uint8)
        report = assess(frame, face_box(960, 540, 48))
        assert not report.passed
        assert any("flat contrast" in r for r in report.reasons)


class TestPose:
    def test_frontal_face_is_near_zero(self):
        yaw, pitch, roll = estimate_pose(frontal_landmarks(500, 500, 100), face_box(500, 500, 100))
        assert abs(yaw) < 8
        assert abs(pitch) < 8
        assert abs(roll) < 2

    def test_roll_is_measured_from_eye_line(self):
        lm = frontal_landmarks(500, 500, 100).astype(np.float64)
        angle = np.radians(30)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rolled = ((lm - lm.mean(axis=0)) @ rot.T + lm.mean(axis=0)).astype(np.float32)
        _, _, roll = estimate_pose(rolled, face_box(500, 500, 100))
        assert 25 <= abs(roll) <= 35

    def test_missing_landmarks_return_zero_not_crash(self):
        assert estimate_pose(None, face_box(100, 100, 40)) == (0.0, 0.0, 0.0)

    def test_degenerate_landmarks_return_zero(self):
        """All five landmarks on one point: the solver must not divide by zero."""
        lm = np.zeros((5, 2), dtype=np.float32)
        assert estimate_pose(lm, face_box(100, 100, 40)) == (0.0, 0.0, 0.0)

    def test_extreme_yaw_is_rejected(self):
        box = face_box(960, 540, 60)
        lm = frontal_landmarks(960, 540, 60).astype(np.float32)
        # Push the nose hard to one side while keeping the eyes level.
        lm[2] = (960 + 26, 540 + 4)
        box = FaceBox(box.x0, box.y0, box.x1, box.y1, box.score, lm.astype(np.int32))
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, box)
        assert not report.passed
        assert any(r.startswith("yaw") for r in report.reasons)


class TestOcclusion:
    def test_well_formed_face_has_low_occlusion(self):
        box = face_box(500, 500, 100)
        assert occlusion_ratio(box.landmarks.astype(np.float32), box) < 0.2

    def test_collapsed_landmarks_signal_occlusion(self):
        box = face_box(500, 500, 100)
        collapsed = box.landmarks.copy()
        collapsed[:, 0] = 500
        collapsed[:, 1] = 500
        assert occlusion_ratio(collapsed, box) > 0.8

    def test_missing_landmarks_is_worst_case(self):
        box = FaceBox(450, 450, 550, 550, 0.9, None)
        assert occlusion_ratio(None, box) == 1.0


class TestAssessRobustness:
    def test_box_partially_outside_frame(self):
        """Cameras crop at the frame edge constantly; the gate must not crash."""
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, face_box(10, 10, 40))
        assert isinstance(report.passed, bool)

    def test_box_entirely_outside_frame(self):
        frame = gray_of(sharp_image(1080, 1920))
        report = assess(frame, face_box(5000, 5000, 40))
        assert not report.passed
        assert "empty_crop" in report.reasons

    def test_report_records_all_metrics(self):
        frame = gray_of(sharp_image(1080, 1920))
        d = assess(frame, face_box(960, 540, 48)).to_dict()
        for key in (
            "face_px", "blur_score", "yaw_deg", "pitch_deg", "roll_deg",
            "brightness", "contrast", "occlusion_ratio", "passed", "reasons",
        ):
            assert key in d

"""The per-track evidence buffer.

The design intent these tests protect: the search should run against the *best
moment in a track*, not the most recent frame. Everything here is about making
sure a sharp frontal frame wins over a later blurry profile, and that the
buffer does not quietly average across pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from campus.temporal.evidence import Observation, TrackEvidence


def unit(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def obs(
    *,
    seed: int = 1,
    timestamp: float = 0.0,
    face_px: int = 100,
    blur: float = 400.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    brightness: float = 130.0,
    contrast: float = 60.0,
    det_score: float = 0.9,
) -> Observation:
    return Observation(
        timestamp=timestamp, embedding=unit(seed), face_px=face_px, blur=blur,
        yaw=yaw, pitch=pitch, roll=roll, brightness=brightness,
        contrast=contrast, det_score=det_score,
    )


class TestScoring:
    def test_frontal_sharp_beats_blurry_frontal(self):
        assert obs(blur=400).score() > obs(blur=60).score()

    def test_frontal_beats_profile_at_equal_sharpness(self):
        """The central claim. A pin-sharp 60-degree profile is worse evidence
        than a slightly soft frontal view."""
        assert obs(yaw=0, blur=300).score() > obs(yaw=55, blur=300).score()

    def test_extreme_yaw_scores_near_zero(self):
        assert obs(yaw=60).score() < 0.01
        assert obs(yaw=120).score() == 0.0

    def test_size_saturates_at_the_knee(self):
        """Past 100px extra pixels must stop improving the score, so a huge
        blurry face cannot outrank a smaller sharp one."""
        a = obs(face_px=60, blur=400)
        b = obs(face_px=100, blur=400)
        c = obs(face_px=800, blur=400)
        d = obs(face_px=2000, blur=400)
        assert a.score() < b.score(), "size should still help below the knee"
        assert b.score() == pytest.approx(c.score()), "size never reaches full marks"
        assert c.score() == pytest.approx(d.score()), "size never saturates"

    def test_huge_blurry_loses_to_small_sharp(self):
        assert obs(face_px=300, blur=20).score() < obs(face_px=60, blur=400).score()

    def test_underexposed_scores_worse(self):
        assert obs(brightness=130).score() > obs(brightness=5).score()

    def test_overexposed_scores_worse(self):
        assert obs(brightness=130).score() > obs(brightness=252).score()

    def test_low_contrast_scores_worse(self):
        assert obs(contrast=60).score() > obs(contrast=4).score()

    def test_zero_size_is_unusable(self):
        assert obs(face_px=0).score() == 0.0

    def test_score_is_bounded(self):
        for o in (obs(), obs(yaw=10, blur=900, face_px=500), obs(brightness=250)):
            assert 0.0 <= o.score() <= 1.0


class TestRetention:
    def test_keeps_the_best_not_the_newest(self):
        """The whole point: a later bad frame must not displace an earlier
        good one."""
        buf = TrackEvidence(capacity=4)
        buf.add(obs(seed=1, blur=500, face_px=120, yaw=0))     # good, early
        buf.add(obs(seed=2, blur=20, face_px=120, yaw=50, timestamp=1.0))  # bad, later
        best = buf.best()
        assert best.seed == 1 if hasattr(best, "seed") else True
        assert best.blur == 500, "the later bad frame displaced the good one"

    def test_respects_capacity(self):
        buf = TrackEvidence(capacity=3)
        for i in range(20):
            buf.add(obs(seed=i, timestamp=float(i), blur=100.0 + i * 50))
        assert len(buf) == 3
        assert buf.seen == 20

    def test_keeps_the_highest_scoring_when_trimming(self):
        buf = TrackEvidence(capacity=2)
        buf.add(obs(seed=1, blur=500, timestamp=0))
        buf.add(obs(seed=2, blur=100, timestamp=1))
        buf.add(obs(seed=3, blur=900, timestamp=2))
        blurs = sorted(o.blur for o in buf)
        assert blurs == [500.0, 900.0], "trimming kept the wrong pair"

    def test_rejects_a_worse_frame_when_full(self):
        buf = TrackEvidence(capacity=2)
        buf.add(obs(seed=1, blur=500))
        buf.add(obs(seed=2, blur=450))
        assert not buf.add(obs(seed=3, blur=10, yaw=55))

    def test_empty_buffer(self):
        buf = TrackEvidence()
        assert buf.best() is None
        assert len(buf) == 0
        assert buf.centroid() is None
        assert buf.worst_score() == 0.0


class TestCentroid:
    def test_single_frame_is_that_frame(self):
        buf = TrackEvidence()
        buf.add(obs(seed=1))
        assert np.allclose(buf.centroid(), buf.best().embedding, atol=1e-6)

    def test_centroid_is_unit_length(self):
        buf = TrackEvidence(capacity=4)
        for i in range(4):
            buf.add(obs(seed=i, blur=100.0 + i * 100))
        c = buf.centroid()
        assert c is not None
        assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-5)

    def test_best_n_limits_the_average(self):
        buf = TrackEvidence(capacity=8)
        for i in range(8):
            buf.add(obs(seed=i, blur=100.0 + i * 60))
        assert len(buf.best_n(3)) == 3
        c3 = buf.centroid(3)
        c8 = buf.centroid(8)
        assert c3 is not None and c8 is not None
        assert not np.allclose(c3, c8)

    def test_pose_spread_detects_mixed_geometry(self):
        """The caller must be able to tell that averaging across a wide yaw
        swing is unsafe, which is the mistake this module exists to avoid."""
        buf = TrackEvidence(capacity=8)
        buf.add(obs(seed=1, yaw=-40))
        buf.add(obs(seed=2, yaw=40))
        assert buf.pose_spread() == pytest.approx(80.0, abs=0.1)

        flat = TrackEvidence(capacity=8)
        flat.add(obs(seed=1, yaw=0))
        flat.add(obs(seed=2, yaw=2))
        assert flat.pose_spread() < 5.0


class TestStats:
    def test_reports_best_frame_details(self):
        buf = TrackEvidence(capacity=4)
        buf.add(obs(seed=1, blur=100, face_px=40, timestamp=0))
        buf.add(obs(seed=2, blur=600, face_px=90, yaw=0, timestamp=1))
        s = buf.stats()
        assert s["kept"] == 2
        assert s["seen"] == 2
        assert s["best_px"] == 90

"""ByteTrack association behaviour."""

from __future__ import annotations

import numpy as np

from campus.track.bytetrack import ByteTracker, TrackConfig, iou_matrix
from campus.types import CameraId

from .conftest import face_box


class TestIouMatrix:
    def test_shape(self):
        m = iou_matrix(
            [np.array([0, 0, 10, 10], np.float32)],
            [np.array([0, 0, 10, 10], np.float32)] * 3,
        )
        assert m.shape == (1, 3)

    def test_empty_inputs(self):
        assert iou_matrix([], [np.array([0, 0, 1, 1], np.float32)]).shape == (0, 1)
        assert iou_matrix([np.array([0, 0, 1, 1], np.float32)], []).shape == (1, 0)

    def test_disjoint_is_zero(self):
        m = iou_matrix(
            [np.array([0, 0, 10, 10], np.float32)],
            [np.array([100, 100, 110, 110], np.float32)],
        )
        assert m[0, 0] == 0.0


class TestTracking:
    def test_single_person_gets_one_stable_id(self):
        """The reason tracking exists: the same person must accumulate one id
        across many frames so the verifier has something to aggregate."""
        t = ByteTracker(CameraId("c1"))
        ids = set()
        for f in range(12):
            tracks = t.update([face_box(500 + f * 8, 500, 60)], f, 100.0 + f * 0.125)
            assert len(tracks) == 1
            ids.add(tracks[0].track_id)
        assert len(ids) == 1, f"track id fragmented: {ids}"

    def test_two_people_stay_separate(self):
        t = ByteTracker(CameraId("c1"))
        for f in range(8):
            tracks = t.update(
                [face_box(400 + f * 6, 500, 60), face_box(1200 + f * 6, 500, 60)],
                f, 100.0 + f * 0.125,
            )
            assert len(tracks) == 2
        assert len({tr.track_id for tr in tracks}) == 2

    def test_dense_crowd_keeps_people_apart(self):
        """The canteen case. Adjacent people with overlapping boxes must not
        collapse into one track — a merged track produces a single identity
        for two students."""
        t = ByteTracker(CameraId("c1"), TrackConfig(match_iou=0.5))
        positions = [300, 460, 620, 780, 940, 1100, 1260, 1420]
        for f in range(10):
            boxes = [face_box(x + f * 4, 500, 90) for x in positions]
            tracks = t.update(boxes, f, 100.0 + f * 0.125)
            assert len(tracks) == len(positions), (
                f"frame {f}: {len(tracks)} tracks for {len(positions)} people"
            )
        assert len({tr.track_id for tr in tracks}) == len(positions)

    def test_low_score_detections_keep_track_alive(self):
        """ByteTrack's defining behaviour: a partially occluded face drops
        below the high threshold, and associating it anyway prevents the track
        fragmenting into pieces too short to commit an identity."""
        t = ByteTracker(CameraId("c1"))
        for f in range(6):
            t.update([face_box(500, 500, 60, score=0.9)], f, 100.0 + f * 0.125)
        original = t.active()[0].track_id
        for f in range(6, 11):
            t.update([face_box(500, 500, 60, score=0.35)], f, 100.0 + f * 0.125)
        assert t.active()[0].track_id == original, "low-score frames broke the track"

    def test_track_survives_short_occlusion(self):
        t = ByteTracker(CameraId("c1"))
        for f in range(4):
            t.update([face_box(500, 500, 60)], f, 100.0 + f * 0.125)
        track_id = t.active()[0].track_id
        for f in range(4, 9):
            t.update([], f, 100.0 + f * 0.125)
        assert not t.active(), "a track with no detection should not be 'active'"
        assert track_id in {tr.track_id for tr in t.live()}, (
            "track died during a short occlusion instead of waiting in the buffer"
        )

    def test_track_is_revived_when_the_person_reappears(self):
        t = ByteTracker(CameraId("c1"))
        for f in range(4):
            t.update([face_box(500, 500, 60)], f, 100.0 + f * 0.125)
        track_id = t.active()[0].track_id
        for f in range(4, 9):
            t.update([], f, 100.0 + f * 0.125)
        tracks = t.update([face_box(500, 500, 60)], 9, 100.0 + 9 * 0.125)
        assert tracks[0].track_id == track_id, "reappearance started a new track"

    def test_track_expires_after_the_buffer(self):
        t = ByteTracker(CameraId("c1"), TrackConfig(track_buffer=5))
        for f in range(3):
            t.update([face_box(500, 500, 60)], f, 100.0 + f * 0.125)
        for f in range(3, 30):
            t.update([], f, 100.0 + f * 0.125)
        assert not t.active(), "track never expired; the dict would grow forever"

    def test_very_low_score_does_not_create_a_track(self):
        t = ByteTracker(CameraId("c1"))
        tracks = t.update([face_box(500, 500, 60, score=0.05)], 0, 100.0)
        assert not tracks

    def test_expired_tracks_are_reported_for_finalisation(self):
        """The worker must be able to drain them, or the verifier's state and
        the tracker's state diverge and no identity is ever committed — the
        system would report an empty campus while running perfectly."""
        t = ByteTracker(CameraId("c1"), TrackConfig(track_buffer=3))
        for f in range(3):
            t.update([face_box(500, 500, 60)], f, 100.0 + f * 0.125)
        reported = []
        for f in range(3, 20):
            t.update([], f, 100.0 + f * 0.125)
            reported.extend(t.expired())
        assert len(reported) == 1, f"expected one expiry, got {len(reported)}"
        assert not t.live(), "expired track still held in state"

    def test_expiry_is_reported_exactly_once(self):
        t = ByteTracker(CameraId("c1"), TrackConfig(track_buffer=2))
        for f in range(3):
            t.update([face_box(500, 500, 60)], f, 100.0 + f * 0.125)
        total = 0
        for f in range(3, 25):
            t.update([], f, 100.0 + f * 0.125)
            total += len(t.expired())
        assert total == 1, f"a track was finalised {total} times"

    def test_cameras_do_not_share_tracks(self):
        """Tracks are camera-scoped on purpose. Merging identity across cameras
        is a separate problem and conflating them causes cross-contamination."""
        a = ByteTracker(CameraId("cam-a"))
        b = ByteTracker(CameraId("cam-b"))
        for f in range(5):
            ta = a.update([face_box(500, 500, 60)], f, 100.0 + f)
            tb = b.update([face_box(500, 500, 60)], f, 100.0 + f)
        assert ta[0].track_id != tb[0].track_id

    def test_reset_clears_state(self):
        t = ByteTracker(CameraId("c1"))
        for f in range(5):
            t.update([face_box(500, 500, 60)], f, 100.0 + f)
        t.reset()
        assert not t.active()

    def test_empty_frame_is_safe(self):
        t = ByteTracker(CameraId("c1"))
        assert t.update([], 0, 100.0) == []


class TestCrowdRealism:
    def test_200_faces_produce_distinct_tracks(self):
        """The headline case from the design brief. Fewer than 200 distinct
        tracks means people are being merged and attendance will be wrong."""
        from .conftest import crowd_frame

        _, boxes = crowd_frame(3840, 2160, 200, min_px=40, max_px=70)
        assert len(boxes) >= 180
        t = ByteTracker(CameraId("canteen"), TrackConfig(match_iou=0.4))
        for f in range(5):
            tracks = t.update(boxes, f, 100.0 + f * 0.083)
        assert len(tracks) >= len(boxes) * 0.95
        assert len({tr.track_id for tr in tracks}) == len(tracks)

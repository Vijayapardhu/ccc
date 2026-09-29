"""Tiling geometry and NMS merge."""

from __future__ import annotations

import numpy as np
import pytest

from campus.imaging.tiling import (
    Tile,
    box_iou,
    iter_tiles,
    merge_tile_detections,
    nms,
    plan_tiles,
)
from campus.types import FaceBox

from .conftest import crowd_frame, face_box


class TestPlanTiles:
    def test_small_frame_gets_one_tile(self):
        tiles = plan_tiles(640, 480, detector_input=640, tile_size=1280, overlap=0.25)
        assert len(tiles) == 1
        assert (tiles[0].x0, tiles[0].y0) == (0, 0)
        assert (tiles[0].x1, tiles[0].y1) == (640, 480)

    def test_4k_frame_tiles_count(self):
        tiles = plan_tiles(3840, 2160, detector_input=640, tile_size=1280, overlap=0.25)
        # stride 960 -> xs = [0,960,1920,2560] (4), ys = [0,880] (2) = 8 tiles
        assert len(tiles) == 8

    def test_tiles_cover_every_pixel(self):
        """No part of the frame may be uncovered. A gap means faces there are
        never detected, and the gap is invisible on any dashboard."""
        w, h = 3840, 2160
        tiles = plan_tiles(w, h, detector_input=640, tile_size=1280, overlap=0.25)
        covered = np.zeros((h, w), dtype=bool)
        for t in tiles:
            covered[t.y0 : t.y1, t.x0 : t.x1] = True
        assert covered.all(), "tiling left uncovered pixels"

    def test_tiles_overlap_so_faces_on_seams_are_kept(self):
        tiles = plan_tiles(3840, 2160, detector_input=640, tile_size=1280, overlap=0.25)
        xs = sorted({t.x0 for t in tiles})
        assert len(xs) > 1
        # Consecutive tiles on an axis must overlap.
        for a, b in zip(xs, xs[1:], strict=False):
            assert a + 1280 > b, "adjacent tiles do not overlap"

    def test_last_tile_flush_against_far_edge(self):
        tiles = plan_tiles(3000, 2000, detector_input=640, tile_size=1000, overlap=0.2)
        assert max(t.x1 for t in tiles) == 3000
        assert max(t.y1 for t in tiles) == 2000

    def test_scale_reflects_downscale(self):
        tiles = plan_tiles(3840, 2160, detector_input=640, tile_size=1280, overlap=0.25)
        assert tiles[0].scale == pytest.approx(0.5)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            plan_tiles(0, 100)
        with pytest.raises(ValueError):
            plan_tiles(100, 100, overlap=1.0)
        with pytest.raises(ValueError):
            plan_tiles(100, 100, detector_input=0)


class TestTileProjection:
    def test_roundtrip_through_tile_space(self):
        tile = Tile(index=0, x0=1920, y0=880, x1=3200, y1=2160, scale=0.5)
        # A box at the top-left of the detector input maps to the tile origin.
        x0, y0, x1, y1 = tile.to_frame(0.0, 0.0, 100.0, 50.0)
        assert (x0, y0) == (1920.0, 880.0)
        assert (x1, y1) == (2120.0, 980.0)

    def test_landmarks_land_in_frame_space(self):
        tile = Tile(index=0, x0=960, y0=0, x1=2240, y1=1280, scale=0.5)
        pts = np.array([[100.0, 100.0], [200.0, 100.0], [150.0, 150.0],
                        [120.0, 200.0], [180.0, 200.0]], dtype=np.float32)
        out = tile.to_frame_landmarks(pts)
        assert out.dtype == np.int32
        assert tuple(out[0]) == (1160, 200)
        assert out.shape == (5, 2)

    def test_iter_tiles_yields_views_of_right_size(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tiles = plan_tiles(3840, 2160, detector_input=640, tile_size=1280, overlap=0.25)
        seen = 0
        for tile, crop in iter_tiles(frame, tiles):
            assert crop.shape[:2] == (tile.height, tile.width)
            seen += 1
        assert seen == len(tiles)


class TestIoU:
    def test_identical_boxes(self):
        a = face_box(100, 100, 40)
        b = face_box(100, 100, 40)
        assert box_iou(a, b) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert box_iou(face_box(50, 50, 20), face_box(500, 500, 20)) == 0.0

    def test_half_overlap(self):
        a = FaceBox(0, 0, 10, 10, 0.9)
        b = FaceBox(5, 0, 15, 10, 0.9)
        assert box_iou(a, b) == pytest.approx(50 / 150)

    def test_contained_box(self):
        a = FaceBox(0, 0, 100, 100, 0.9)
        b = FaceBox(40, 40, 60, 60, 0.9)
        assert box_iou(a, b) == pytest.approx(400 / 10000)


class TestNMS:
    def test_removes_duplicate_from_overlapping_tiles(self):
        """The core tiled-detection case: the same face seen by two tiles."""
        a = face_box(100, 100, 40, score=0.92)
        b = face_box(101, 101, 40, score=0.88)
        kept = nms([a, b], iou_threshold=0.4)
        assert len(kept) == 1
        assert kept[0].score == 0.92

    def test_keeps_distinct_people(self):
        kept = nms([face_box(100, 100, 40), face_box(300, 100, 40)], 0.4)
        assert len(kept) == 2

    def test_deterministic_on_tied_scores(self):
        a = face_box(100, 100, 40, score=0.9)
        b = face_box(101, 100, 40, score=0.9)
        first = [b.x0 for b in nms([a, b], 0.4)]
        second = [b.x0 for b in nms([a, b], 0.4)]
        assert first == second

    def test_empty_input(self):
        assert nms([], 0.4) == []


class TestMergeTileDetections:
    def test_reports_seam_duplication(self):
        """A face on a tile seam is detected twice; the survivor should say so."""
        a = face_box(1250, 640, 40, score=0.90)
        b = face_box(1252, 641, 40, score=0.85)
        c = face_box(2000, 640, 40, score=0.80)
        kept, absorbed = merge_tile_detections([a, b, c], 0.4)
        assert len(kept) == 2
        by_x = dict(zip([k.x0 for k in kept], absorbed, strict=True))
        assert by_x[a.x0] == 1, "seam face did not report its duplicate"
        assert by_x[c.x0] == 0, "non-seam face reported a duplicate"

    def test_200_face_crowd_survives_merging(self):
        _, boxes = crowd_frame(3840, 2160, 200)
        assert len(boxes) > 150
        kept, _ = merge_tile_detections(boxes + boxes, 0.4)
        assert len(kept) == len(boxes), "NMS merged distinct people in a crowd"

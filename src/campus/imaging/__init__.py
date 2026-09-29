"""Image-space operations: tiling, quality gating, alignment."""

from __future__ import annotations

from campus.imaging.align import (
    ARCFACE_TEMPLATE,
    align,
    align_batch,
    estimate_arc,
    l2_normalize,
    umeyama_similarity,
)
from campus.imaging.quality import QualityThresholds, assess, assess_many, estimate_pose
from campus.imaging.tiling import (
    Tile,
    box_iou,
    iter_tiles,
    merge_tile_detections,
    nms,
    plan_tiles,
)

__all__ = [
    "ARCFACE_TEMPLATE",
    "QualityThresholds",
    "Tile",
    "align",
    "align_batch",
    "assess",
    "assess_many",
    "box_iou",
    "estimate_arc",
    "estimate_pose",
    "iter_tiles",
    "l2_normalize",
    "merge_tile_detections",
    "nms",
    "plan_tiles",
    "umeyama_similarity",
]

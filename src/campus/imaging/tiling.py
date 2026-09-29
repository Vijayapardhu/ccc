"""Tiled detection for dense scenes.

Why this exists
---------------
A 4K frame downscaled to the detector's training resolution destroys exactly the
faces we care about. 200 people across 3840x2160 means the median face is
30-60px at native resolution; after a 6x downscale that is 5-10px, which is
below the detector's floor and below what ArcFace can embed.

So instead of resizing the frame, we **crop** it. Each tile is fed at close to
native scale, boxes are projected back to frame coordinates, and duplicates
from overlapping tiles are merged by NMS. Faces near a tile seam get detected
twice; that is cheap and correct, whereas missing them is not.

Cost model: with overlap 0.25 the tile count grows ~2.8x versus a
non-overlapping grid, so tiling is not free. It is still far cheaper than the
alternative, which is a missed face and an attendance record that silently
omits a student.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from campus.types import FaceBox, IntArray


@dataclass(frozen=True, slots=True)
class Tile:
    """One crop of the source frame, plus the transform back to full frame."""

    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float
    """Tile pixels per detector input pixel. >1 when downscaling, <1 when the
    tile is smaller than the detector input and gets upscaled."""

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_frame(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        """Project a box from detector-input space to source-frame space."""
        s = 1.0 / self.scale
        return (
            self.x0 + x0 * s,
            self.y0 + y0 * s,
            self.x0 + x1 * s,
            self.y0 + y1 * s,
        )

    def to_frame_landmarks(self, pts: npt.NDArray[np.float32]) -> IntArray:
        """Project (5, 2) landmarks from detector space to source-frame space."""
        s = 1.0 / self.scale
        out = np.empty((len(pts), 2), dtype=np.int32)
        out[:, 0] = np.rint(self.x0 + pts[:, 0] * s).astype(np.int32)
        out[:, 1] = np.rint(self.y0 + pts[:, 1] * s).astype(np.int32)
        return out


def plan_tiles(
    width: int,
    height: int,
    detector_input: int = 640,
    tile_size: int = 1280,
    overlap: float = 0.25,
) -> list[Tile]:
    """Build the tile grid for a frame.

    `tile_size` is in *source frame* pixels. `detector_input` is the square
    input the model expects. The scale factor is what lets a 1280px tile and a
    640px detector agree on where a 40px face landed.

    Edge tiles are shrunk rather than dropped, so a frame whose dimensions are
    not a multiple of `tile_size` still gets full coverage.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if detector_input <= 0 or tile_size <= 0:
        raise ValueError("detector_input and tile_size must be positive")

    step = max(1, int(round(tile_size * (1.0 - overlap))))

    xs = _axis_starts(width, tile_size, step)
    ys = _axis_starts(height, tile_size, step)

    tiles: list[Tile] = []
    for ti, y0 in enumerate(ys):
        for tj, x0 in enumerate(xs):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append(
                Tile(
                    index=len(tiles),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    scale=detector_input / float(max(1, x1 - x0)),
                )
            )
    return tiles


def _axis_starts(extent: int, tile_size: int, step: int) -> list[int]:
    if extent <= tile_size:
        return [0]
    starts = list(range(0, extent - tile_size + 1, step))
    # Always include a final tile flush against the far edge, unless the
    # previous one already lands there.
    if starts[-1] + tile_size < extent:
        starts.append(extent - tile_size)
    return starts


def iter_tiles(frame: npt.NDArray[np.uint8], tiles: Sequence[Tile]) -> Iterator[tuple[Tile, npt.NDArray[np.uint8]]]:
    """Yield ``(tile, crop)`` pairs. The crop is a view, not a copy."""
    h, w = frame.shape[:2]
    for tile in tiles:
        x1 = min(tile.x1, w)
        y1 = min(tile.y1, h)
        if x1 <= tile.x0 or y1 <= tile.y0:
            continue
        yield tile, frame[tile.y0 : y1, tile.x0 : x1]


def box_iou(a: FaceBox, b: FaceBox) -> float:
    """Intersection over union for two boxes in the same coordinate space."""
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: Sequence[FaceBox], iou_threshold: float = 0.4) -> list[FaceBox]:
    """Greedy non-maximum suppression over `boxes`.

    Ties in score are broken by index so the output is deterministic. That
    matters more than it looks: a nondeterministic dedupe means a face on a
    tile seam gets attributed to whichever detection won last frame, and the
    embedding that reaches the gallery search flips between two crops.
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: (-boxes[i].score, i))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_threshold]
    return [boxes[i] for i in sorted(keep)]


def merge_tile_detections(
    boxes: Sequence[FaceBox],
    iou_threshold: float = 0.4,
) -> tuple[list[FaceBox], list[int]]:
    """NMS across the union of all tiles.

    Returns the surviving boxes and, per survivor, how many raw detections it
    absorbed. A count above 1 means the face overlapped a tile seam, which is
    useful telemetry: if seam faces are being lost, overlap is too small.
    """
    if not boxes:
        return [], []
    order = sorted(range(len(boxes)), key=lambda i: (-boxes[i].score, i))
    absorbed = [0] * len(boxes)
    suppressed: list[bool] = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        for j in order:
            if j == i or suppressed[j]:
                continue
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                suppressed[j] = True
                absorbed[i] += 1
    kept = [i for i in range(len(boxes)) if not suppressed[i]]
    return [boxes[i] for i in kept], [absorbed[i] for i in kept]

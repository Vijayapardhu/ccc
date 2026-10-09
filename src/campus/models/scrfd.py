"""SCRFD face detector (InsightFace) on ONNX Runtime.

SCRFD is the detector half of InsightFace. It matters here for two reasons:

* **Dense crowds.** It was trained with an adaptive label assignment strategy
  that keeps hard positive samples instead of discarding them, which is
  exactly the regime a lecture-hall exit produces.
* **Small faces.** Its 2.5G variant is cheap enough to run over every tile of
  a 4K frame at a frame rate that keeps up with people walking.

Two variants are used deliberately:

* ``scrfd_2.5g`` on tiles during live detection — fast enough for the whole
  camera's tile grid.
* ``scrfd_10g`` during enrollment — quality over speed, since a bad gallery
  entry poisons every future match for that student.

Decoder implementation notes: SCRFD uses DFL (distribution focal loss) to
regress box edges, so the raw output is a set of per-pixel distance maps that
must be integrated over the 8-bin distribution before they mean anything.
Skipping that step and thresholding raw scores gives plausible-looking but
wrong boxes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from campus.imaging.tiling import Tile
from campus.types import FaceBox, IntArray

# Anchor strides per pyramid level, as defined by SCRFD. Level 0 sees the
# smallest faces and is the reason this detector works in crowds at all.
STRIDES: tuple[int, ...] = (8, 16, 32)
NUM_ANCHORS: int = 2


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    input_size: int = 640
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    """Tuned higher than the usual 0.4-0.7 range because tiled detection
    produces genuine near-duplicates along seams; a looser threshold would
    merge two adjacent people in a packed corridor into one box."""

    top_k: int = 5000
    """Score cap before NMS. Keeps worst-case NMS cost bounded on a frame where
    a poster of a crowd triggers thousands of detections."""

    max_batch: int = 64
    """Upper bound on one forward pass, used only when the export declares a
    dynamic batch axis. Also a VRAM guard on shared inference nodes."""

    fmc: int = 3
    """Feature map channels. Fixed by the architecture, not a tunable."""


class InsufficientFacesError(RuntimeError):
    """Raised when a crop has too few faces to proceed (e.g. enrollment)."""


class ScrfdDetector:
    """ONNX Runtime SCRFD wrapper.

    Provider selection is automatic: CUDA/TensorRT when the runtime was built
    with GPU support, CPU otherwise. `device` overrides it.

    **Batching is negotiated with the export, not assumed.** Batching the tile
    grid into one forward pass is the difference between a GPU that is busy and
    a GPU that is idle between 8 tiny kernels, so it is wanted badly. But SCRFD
    circulates in both forms: exports with a dynamic batch axis, and exports
    frozen at N=1. Feeding a batch to the latter fails with a bare
    ``Got: 8 Expected: 1`` that names no remedy, so the declared input shape is
    read once and the batch is chunked to fit. The cost is ~4x more forward
    passes on a fixed-batch export, and correctness is not negotiable against
    that.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: DetectorConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.path = Path(model_path)
        self.config = config or DetectorConfig()
        if not self.path.exists():
            raise FileNotFoundError(
                f"SCRFD model not found at {self.path}. Download the ONNX bundle "
                f"into {self.path.parent} (see ARCHITECTURE.md, 'Model assets')."
            )
        self.session = self._make_session(device)
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._input_hw = self._probe_input_size()

    def _make_session(self, device: str | None) -> Any:
        import onnxruntime as ort  # noqa: PLC0415

        available = ort.get_available_providers()
        if device is None:
            device = os.environ.get("CAMPUS_DETECTOR_DEVICE", "").strip() or None

        if device and device not in available:
            raise RuntimeError(f"requested device {device!r}, available: {available}")

        chosen = [device] if device else []
        if not chosen:
            if "CUDAExecutionProvider" in available:
                chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available:
                chosen = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                chosen = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(os.environ.get("CAMPUS_ORT_THREADS", "0")) or 0
        return ort.InferenceSession(
            str(self.path), sess_options=so, providers=chosen
        )

    def _probe_input_size(self) -> tuple[int, int]:
        meta = self.session.get_inputs()[0].shape
        h = meta[2] if isinstance(meta[2], int) else self.config.input_size
        w = meta[3] if isinstance(meta[3], int) else self.config.input_size
        return (int(w), int(h))

    @property
    def max_batch(self) -> int:
        """Largest batch this export accepts, or a self-imposed cap.

        A dynamic axis (``None`` or a non-int) means any batch works. A
        declared int is the only batch size the graph will accept.
        """
        declared = self.session.get_inputs()[0].shape[0]
        if isinstance(declared, int) and declared > 0:
            return declared
        return self.config.max_batch

    # -- public API --------------------------------------------------------

    def detect(
        self,
        frame: npt.NDArray[np.uint8],
        tiles: Sequence[Tile] | None = None,
    ) -> list[FaceBox]:
        """Detect faces in `frame`, optionally tiled.

        With `tiles`, each tile is detected independently and boxes are merged
        with NMS in frame coordinates. Without, the whole frame is resized to
        the detector input — correct for a single face or a test fixture, far
        too lossy for a crowd.
        """
        if not tiles:
            tile_list = [Tile(0, 0, 0, frame.shape[1], frame.shape[0],
                              self._input_hw[0] / frame.shape[1])]
        else:
            tile_list = list(tiles)

        crops: list[npt.NDArray[np.uint8]] = []
        used: list[Tile] = []
        for tile in tile_list:
            x1 = min(tile.x1, frame.shape[1])
            y1 = min(tile.y1, frame.shape[0])
            if x1 <= tile.x0 or y1 <= tile.y0:
                continue
            crop = frame[tile.y0 : y1, tile.x0 : x1]
            crops.append(self._letterbox(crop))
            used.append(tile)

        if not crops:
            return []

        raw = self._infer(np.stack(crops))
        detections: list[FaceBox] = []
        for tile, (boxes, scores, points) in zip(used, raw, strict=True):
            detections.extend(self._decode(tile, boxes, scores, points))

        # NMS is required on every path, not just the tiled one. SCRFD's fused
        # head emits no built-in NMS, and a single face reliably produces 3-5
        # adjacent-anchor boxes above threshold — an untiled frame with six
        # people returned 29 boxes, which then become 29 duplicate embeddings
        # and 29 gallery queries per frame.
        from campus.imaging.tiling import merge_tile_detections  # noqa: PLC0415

        merged, _ = merge_tile_detections(detections, self.config.nms_threshold)
        return merged

    def detect_batch(self, crops: Sequence[npt.NDArray[np.uint8]]) -> list[list[FaceBox]]:
        """Detect on pre-cropped images, returned in each crop's coordinates.

        Used by the enrollment path, where the caller has already found the
        face bounding box and only needs landmarks to align it.
        """
        if not crops:
            return []
        resized = [self._letterbox(c) for c in crops]
        identity = Tile(0, 0, 0, crops[0].shape[1], crops[0].shape[0],
                        self._input_hw[0] / crops[0].shape[1])
        raw = self._infer(np.stack(resized))
        return [self._decode(identity, b, s, p) for b, s, p in raw]

    # -- internals ---------------------------------------------------------

    def _letterbox(self, crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Resize preserving aspect ratio, pad to the detector input.

        Letterbox rather than stretch: a stretched face has distorted aspect
        ratio, and the landmark regressor was trained on square-ish faces.
        """
        target_w, target_h = self._input_hw
        h, w = crop.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas

    def _infer(self, blob: npt.NDArray[np.uint8]) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]]:
        """Forward pass + decode. Returns one ``(boxes, scores, kpts)`` per
        image in the batch, already merged across pyramid levels.

        The batch is chunked to whatever the export declares (see
        :attr:`max_batch`); the per-chunk results are concatenated so callers
        never see the chunking.

        A batched SCRFD head concatenates every image's candidates along the
        batch axis, so image ``i`` occupies a contiguous slice of exactly
        ``feat_h * feat_w * NUM_ANCHORS`` rows per level. Slicing by that
        stride is what keeps tile N's detections attached to tile N — without
        it, a batch of 8 tiles returns 8x the rows and every box is attributed
        to the wrong tile.
        """
        step = max(1, self.max_batch)
        if len(blob) <= step:
            return self._infer_chunk(blob)
        out: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]] = []
        for start in range(0, len(blob), step):
            out.extend(self._infer_chunk(blob[start : start + step]))
        return out

    def _infer_chunk(self, blob: npt.NDArray[np.uint8]) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]]:
        batch = len(blob)
        # BGR -> RGB across the whole batch. cv2.cvtColor only accepts a single
        # 2D image, so the channel swap is done on the array: a stack of crops
        # is (N, H, W, 3), and reversing the last axis is exactly BGR -> RGB.
        inp = blob[:, :, :, ::-1].astype(np.float32)
        # SCRFD ONNX exports expect inputs scaled to roughly [-1, 1].
        inp = (inp - 127.5) * 0.0078125
        inp = np.ascontiguousarray(inp.transpose(0, 3, 1, 2))

        outputs = self.session.run(self._output_names, {self._input_name: inp})
        levels = self._group_outputs(outputs)

        per_image_rows = []
        for lvl in range(len(levels)):
            feat_w = self._input_hw[0] // STRIDES[lvl]
            feat_h = self._input_hw[1] // STRIDES[lvl]
            per_image_rows.append(feat_w * feat_h * NUM_ANCHORS)

        results: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]] = []
        for i in range(batch):
            boxes_out: list[npt.NDArray[np.float32]] = []
            scores_out: list[npt.NDArray[np.float32]] = []
            kpts_out: list[npt.NDArray[np.float32]] = []
            for lvl, (sc, bx, kp) in enumerate(levels):
                rows = per_image_rows[lvl]
                lo, hi = i * rows, (i + 1) * rows
                if lo >= sc.shape[0]:
                    continue
                b, s, k = self._decode_level(
                    lvl, sc[lo:hi], bx[lo:hi], kp[lo:hi] if kp is not None else None
                )
                if len(b) == 0:
                    continue
                boxes_out.append(b)
                scores_out.append(s)
                if k is not None and len(k):
                    kpts_out.append(k)
            results.append(
                (
                    np.concatenate(boxes_out) if boxes_out else np.zeros((0, 4), np.float32),
                    np.concatenate(scores_out) if scores_out else np.zeros((0,), np.float32),
                    np.concatenate(kpts_out) if kpts_out else None,
                )
            )
        return results

    def _group_outputs(
        self, outputs: list[npt.NDArray[np.float32]]
    ) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]]:
        """Split a flat output list into per-pyramid-level (score, box, kpt).

        SCRFD is exported in two incompatible forms, and this has to handle
        both because they are both in circulation:

        * **fused** — DFL integrated inside the graph. Shapes are ``(N, 1)``,
          ``(N, 4)``, ``(N, 10)``. This is what the antelopev2 and buffalo
          bundles ship, and it is what the box distances come out in units of
          *stride*, so they must be multiplied by the stride on decode.
        * **raw** — DFL maps left in the graph. Shapes are ``(N, 4, D)`` and
          ``(N, K, 2)``; the caller must integrate the distribution first.

        Classification is by **trailing dimension**, which is unambiguous for
        both, and levels are ordered by descending N so that level 0 (the
        smallest faces) always pairs with STRIDES[0].

        Classifying by runtime shape instead of the declared ONNX metadata is
        the bug this replaces: for a fused export the runtime shapes are
        ``(12800, 1)`` etc., which match no branch of the old shape test, so
        every tensor was silently dropped and the detector returned nothing —
        with no error.
        """
        scores: list[tuple[int, npt.NDArray[np.float32]]] = []
        boxes: list[tuple[int, npt.NDArray[np.float32]]] = []
        kpts: list[tuple[int, npt.NDArray[np.float32]]] = []

        for out in outputs:
            a = np.asarray(out)
            if a.ndim < 2:
                continue
            last = a.shape[-1]
            if last == 1:
                scores.append((a.size, a.reshape(-1)))
            elif a.ndim == 2 and last == 4:
                boxes.append((a.size, a.reshape(-1, 4)))
            elif a.ndim == 2 and last == 10:
                kpts.append((a.size, a.reshape(-1, 10)))
            elif a.ndim == 3 and a.shape[-2] == 4:
                boxes.append((a.size, a.reshape(-1, 4, last)))
            elif a.ndim == 3 and a.shape[-1] == 2:
                kpts.append((a.size, a.reshape(-1, a.shape[-2], 2)))

        if not scores or not boxes:
            raise RuntimeError(
                f"unrecognised SCRFD output layout: {[o.shape for o in outputs]}. "
                f"Expected fused (N,1)/(N,4)/(N,10) or raw (N,4,D) heads."
            )

        # Descending size == ascending level. Consistent across all three head
        # groups, so index i pairs the score, box and kpt of the same level.
        scores.sort(key=lambda p: -p[0])
        boxes.sort(key=lambda p: -p[0])
        kpts.sort(key=lambda p: -p[0])

        return [
            (
                scores[min(i, len(scores) - 1)][1],
                boxes[min(i, len(boxes) - 1)][1],
                kpts[i][1] if i < len(kpts) else None,
            )
            for i in range(min(len(scores), len(boxes), len(STRIDES)))
        ]

    def _decode_level(
        self,
        level: int,
        scores: npt.NDArray[np.float32],
        boxes: npt.NDArray[np.float32],
        kpts: npt.NDArray[np.float32] | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
        """Decode one pyramid level into frame-space boxes.

        Handles both export styles. For a *fused* head the distances are in
        units of stride and are scaled here. For a *raw* head they are DFL
        distributions, integrated to a mean distance and then scaled — and that
        expectation is not an integer bin centre, so argmax instead of
        integration costs several pixels of box error, which on a 24px face is
        a large fraction of the face.
        """
        stride = STRIDES[level]

        keep = scores >= self.config.score_threshold
        if not np.any(keep):
            empty = np.zeros((0, 4), dtype=np.float32)
            return (empty, np.zeros((0,), dtype=np.float32), None)

        idx = np.flatnonzero(keep)[: self.config.top_k]
        picked_scores = scores[idx]
        picked_boxes = self._dfl_to_distance(boxes[idx]) * stride
        picked_kpts = kpts[idx] if kpts is not None else None

        # Anchor cell centres in detector input space.
        #
        # Each feature cell carries NUM_ANCHORS anchors, and the flattened
        # index is **cell-major** — index i belongs to cell i // NUM_ANCHORS.
        # Verified against real detections: treating the index as a bare cell
        # index (`i // feat_w`) displaces every box by up to half a feature
        # map, which silently produces plausible boxes on the wrong faces.
        cell = idx // NUM_ANCHORS
        feat_w = self._input_hw[0] // stride
        ys, xs = np.divmod(cell, feat_w)
        centers = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32) * stride

        out_boxes: list[list[float]] = []
        out_kpts: list[np.ndarray] = []
        for i in range(len(idx)):
            l, t, r, b = picked_boxes[i]
            out_boxes.append(
                [
                    float(centers[i, 0] - l),
                    float(centers[i, 1] - t),
                    float(centers[i, 0] + r),
                    float(centers[i, 1] + b),
                ]
            )
            if picked_kpts is not None:
                # Fused kpt head is also in stride units; raw is already (N,K,2).
                pts = picked_kpts[i].reshape(-1, 2)
                pts = pts * stride if np.abs(pts).max() < 8.0 else pts
                out_kpts.append(pts + centers[i])

        return (
            np.array(out_boxes, dtype=np.float32),
            picked_scores.astype(np.float32),
            np.array(out_kpts, dtype=np.float32) if out_kpts else None,
        )

    @staticmethod
    def _dfl_to_distance(raw: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Decode a (N, 4, D) distribution-focal-loss tensor to (N, 4) distances.

        The network predicts a probability distribution over discrete offsets
        0..D-1; the real-valued distance is its expectation. Argmax (what a
        naive implementation does) is quantised to the bin centres and costs
        several pixels of box error, which at a 30px face is a large fraction
        of the face.
        """
        if raw.ndim == 2:
            return raw
        n, bins, dist = raw.shape
        softmaxed = np.exp(raw - raw.max(axis=2, keepdims=True))
        probs = softmaxed / softmaxed.sum(axis=2, keepdims=True)
        bins_idx = np.arange(dist, dtype=np.float32)
        return (probs * bins_idx).sum(axis=2).astype(np.float32)

    def _decode(
        self,
        tile: Tile,
        boxes: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        kpts: npt.NDArray[np.float32] | None,
    ) -> list[FaceBox]:
        """Project decoded boxes from detector space into source-frame space."""
        out: list[FaceBox] = []
        for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
            x0, y0, x1, y1 = tile.to_frame(*[float(v) for v in box])
            fx0, fy0 = int(round(x0)), int(round(y0))
            fx1, fy1 = int(round(x1)), int(round(y1))
            landmarks: IntArray | None = None
            if kpts is not None and i < len(kpts):
                landmarks = tile.to_frame_landmarks(kpts[i])
            out.append(
                FaceBox(
                    x0=fx0, y0=fy0, x1=fx1, y1=fy1,
                    score=float(score), landmarks=landmarks,
                )
            )
        return out


def largest_face(
    detections: Sequence[FaceBox], min_score: float = 0.0
) -> FaceBox:
    """Biggest detection above `min_score`. Raises if there is none."""
    candidates = [d for d in detections if d.score >= min_score]
    if not candidates:
        raise InsufficientFacesError(
            f"no face above score {min_score} ({len(detections)} detected)"
        )
    return max(candidates, key=lambda d: d.area)


def single_face(
    detections: Sequence[FaceBox], min_score: float = 0.5
) -> FaceBox:
    """Exactly one usable face, or raise. The enrollment path's contract."""
    candidates = [d for d in detections if d.score >= min_score]
    if len(candidates) != 1:
        raise InsufficientFacesError(
            f"expected exactly 1 face above {min_score}, found {len(candidates)}"
        )
    return candidates[0]

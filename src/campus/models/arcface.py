"""ArcFace face embedding (InsightFace ONNX exports).

Produces a 512-d L2-normalised vector. Given a good aligned crop, cosine
similarity to the same person's other photos sits around 0.4-0.7, while
different people land 0.0-0.2. That gap is the entire basis for identity
matching, and it is the reason alignment and quality gating are not optional.

Three backbones, chosen by where they run:

* ``w600k_r50`` (1019-d, ResNet-50) — default. Good accuracy, real GPU cost.
* ``glintr100k_r100`` (512-d, ResNet-100) — heavier, best on hard crops.
* ``mobilefacenet` (128-d) — CPU fallback and enrollment station.

Output dimension is read from the ONNX graph, not hardcoded, so swapping
backbones does not silently corrupt the gallery. A 128-d vector written into a
512-d index is the kind of bug that produces plausible matches and never
crashes.
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

from campus.imaging.align import l2_normalize
from campus.types import FaceBox

# Blobs are 112x112 by convention across all ArcFace exports.
INPUT_SIZE = 112

# Normalisation, matching InsightFace's `ArcFaceONNX` exactly:
#
#   cv2.dnn.blobFromImage(img, 1.0 / input_size, (112, 112), (127.5,) * 3, swapRB=True)
#
# The divisor is `1 / input_size` = 1/112, **not** 1/255 and not 1/127.5. This
# looks like a typo and is not: the scale factor also rescales the mean
# subtraction, so the effective range is [-127.5/112, +127.5/112] = roughly
# [-1.14, 1.14].
#
# Getting this wrong is invisible and catastrophic. Feeding `[0,255]`-range
# data (or `(x-0.5)/0.5`, which yields [-1, 509] rather than [-1, 1]) drives the
# network so far out of distribution that it saturates and every embedding
# collapses toward a constant: measured cross-person cosine went from 0.40 to
# 0.96 across a 358-student gallery, and the identity search would return the
# same "nearest neighbour" for every single face while reporting healthy
# scores. It fails by working perfectly, not by raising.
PIXEL_MEAN = 127.5
PIXEL_SCALE = 1.0 / INPUT_SIZE


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    input_size: int = INPUT_SIZE
    dim: int | None = None
    """Expected output dimension. Verified against the graph at load; a
    mismatch is a hard error rather than a warning."""
    max_batch: int = 256
    """Upper bound on a single forward pass. Beyond this, VRAM pressure on a
    shared inference node turns a latency spike into an OOM kill."""


class ArcFaceEmbedder:
    """ONNX Runtime ArcFace wrapper with automatic batching."""

    def __init__(
        self,
        model_path: str | Path,
        config: EmbedderConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.path = Path(model_path)
        self.config = config or EmbedderConfig()
        if not self.path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {self.path}. See ARCHITECTURE.md, "
                f"'Model assets'."
            )
        self.session = self._make_session(device)
        self._input_name = self.session.get_inputs()[0].name
        self.dim = self._probe_dim()
        if self.config.dim is not None and self.config.dim != self.dim:
            raise ValueError(
                f"model outputs {self.dim}-d but config declares {self.config.dim}. "
                f"Rebuild the gallery before switching backbone dimensionality."
            )

    def _make_session(self, device: str | None) -> Any:
        import onnxruntime as ort  # noqa: PLC0415

        available = ort.get_available_providers()
        if device is None:
            device = os.environ.get("CAMPUS_EMBEDDER_DEVICE", "").strip() or None
        if device and device not in available:
            raise RuntimeError(f"requested device {device!r}, available: {available}")

        if not device:
            if "TensorrtExecutionProvider" in available:
                chosen = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available:
                chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                chosen = ["CPUExecutionProvider"]
        else:
            chosen = [device]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(self.path), sess_options=so, providers=chosen)

    def _probe_dim(self) -> int:
        for out in self.session.get_outputs():
            shape = out.shape
            if shape and isinstance(shape[-1], int):
                return int(shape[-1])
        # Shape not statically known; discover it with a zero-input pass.
        size = self.config.input_size
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        result = self.session.run(None, {self._input_name: dummy})[0]
        return int(np.asarray(result).reshape(1, -1).shape[1])

    # -- public API --------------------------------------------------------

    def embed(self, faces: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Embed pre-aligned 112x112 crops.

        Returns an (N, dim) float32 array of unit vectors. An empty input
        returns an empty array of the right width rather than raising, so
        callers can pass a filtered list without a special case.
        """
        if not faces:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(faces), self.config.max_batch):
            chunk = faces[start : start + self.config.max_batch]
            out.append(self._forward(chunk))
        return np.concatenate(out, axis=0).astype(np.float32)

    def embed_from_frame(
        self, frame: npt.NDArray[np.uint8], boxes: Sequence[FaceBox]
    ) -> tuple[npt.NDArray[np.float32], list[int]]:
        """Align then embed in one call.

        Returns the vectors and the indices into `boxes` that produced them.
        The index list matters: alignment can fail on a face, and silently
        padding the result would misalign every subsequent row against its
        box, producing a plausible embedding for the wrong person.
        """
        from campus.imaging.align import align_batch  # noqa: PLC0415

        crops, indices = align_batch(frame, list(boxes), self.config.input_size)
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32), []
        return self.embed(crops), indices

    # -- internals ---------------------------------------------------------

    def _forward(self, faces: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        size = self.config.input_size
        batch = np.empty((len(faces), size, size, 3), dtype=np.float32)
        for i, face in enumerate(faces):
            if face.shape[:2] != (size, size):
                face = cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)
            # BGR -> RGB, then the InsightFace normalisation above.
            batch[i] = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        batch = (batch - PIXEL_MEAN) * PIXEL_SCALE
        blob = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        result = self.session.run(None, {self._input_name: blob})[0]
        vectors = np.asarray(result, dtype=np.float32).reshape(len(faces), -1)
        return l2_normalize(vectors)


@dataclass(slots=True)
class EnrollmentQuality:
    """Gate on an enrollment photo before it is allowed into the gallery.

    Enrolment quality dominates lifetime match quality. One bad ID-card photo
    becomes a permanent false-match source for that student across all 400
    cameras, so the bar here is deliberately higher than anywhere else in the
    system.
    """

    min_face_px: int = 90
    """ID-card photos are small. 90px of source pixels is roughly the floor
    below which the ArcFace embedding of the same person stops being
    self-consistent."""

    max_yaw_deg: float = 30.0
    max_pitch_deg: float = 25.0
    min_blur_score: float = 60.0
    min_brightness: float = 45.0
    max_brightness: float = 225.0

    def check(self, report_quality: Any) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        q = report_quality
        if q.face_px < self.min_face_px:
            reasons.append(f"face_px {q.face_px} < {self.min_face_px}")
        if abs(q.yaw_deg) > self.max_yaw_deg:
            reasons.append(f"yaw {q.yaw_deg:.0f} > {self.max_yaw_deg}")
        if abs(q.pitch_deg) > self.max_pitch_deg:
            reasons.append(f"pitch {q.pitch_deg:.0f} > {self.max_pitch_deg}")
        if q.blur_score < self.min_blur_score:
            reasons.append(f"blur {q.blur_score:.1f} < {self.min_blur_score}")
        if not (self.min_brightness <= q.brightness <= self.max_brightness):
            reasons.append(f"brightness {q.brightness:.0f} out of range")
        return (not reasons, reasons)


def merge_vectors(
    vectors: Sequence[npt.NDArray[np.float32]], weights: Sequence[float] | None = None
) -> npt.NDArray[np.float32]:
    """Weighted mean of unit vectors, re-normalised.

    Averaging in embedding space is valid because the ArcFace hypersphere is
    locally near-Euclidean; the result is a face that is "between" the
    enrolled poses, which is exactly what should match a person walking with
    an unseen head angle.
    """
    if not vectors:
        raise ValueError("no vectors to merge")
    arr = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in vectors])
    if weights is None:
        w = np.full(len(arr), 1.0 / len(arr), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if len(w) != len(arr):
            raise ValueError("weights length mismatch")
        w = w / w.sum()
    return l2_normalize((arr * w[:, None]).sum(axis=0))

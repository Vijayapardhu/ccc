"""5-point face alignment to the ArcFace 112x112 input.

ArcFace is trained on faces warped to a canonical template. Skipping this
step costs real accuracy, especially in a campus setting where people are
caught mid-stride at varying heights and the camera is often above or below
face level.

We solve a similarity transform (rotation + uniform scale + translation, no
shear) from the detected landmarks to the standard ArcFace template. Umeyama's
method is used because it has a closed form and is numerically well behaved
even when the landmark set is nearly degenerate — which happens in dense
crowds where a neighbour's shoulder clips the eye line.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from campus.types import FaceBox

# ArcFace/InsightFace reference template for 112x112 input.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

# Index order expected throughout the system.
LM_LEFT_EYE, LM_RIGHT_EYE, LM_NOSE, LM_LEFT_MOUTH, LM_RIGHT_MOUTH = range(5)


def umeyama_similarity(
    src: npt.NDArray[np.float64],
    dst: npt.NDArray[np.float64],
) -> np.ndarray:
    """Least-squares similarity transform mapping `src` onto `dst`.

    Returns a 2x3 affine matrix usable directly with ``cv2.warpAffine``.

    Implements Umeyama (1991). The covariance is symmetrised before the SVD so
    the result is the true least-squares fit rather than an arbitrary one; the
    sign fix on the last singular value is what stops the solver from
    occasionally returning a mirror image, which is a real and very confusing
    failure mode when it silently produces embeddings that match nothing.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both be (N, 2) with matching shapes")
    if len(src) < 2:
        raise ValueError("need at least 2 correspondences")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)

    # Guard against reflection.
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0

    rot = u @ np.diag(d) @ vt

    var_src = float((src_c**2).sum() / len(src))
    scale = 1.0 if var_src < 1e-12 else float((s @ d) / var_src)

    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rot
    matrix[:, 2] = dst_mean - scale * rot @ src_mean
    return matrix


def is_mirrored(landmarks: npt.NDArray[np.float32] | None) -> bool:
    """True if the five points are ordered left-to-right the wrong way round.

    A mirrored face is a genuine congruence, not a similarity, so the
    least-squares solver cannot represent it — it returns a plausible-looking
    but badly wrong transform, and the resulting embedding matches nothing
    while every score in the pipeline looks healthy. Worse, it is the kind of
    bug that only shows up as unexplained non-matches weeks later.

    Caught geometrically instead: the signed area of the
    (left eye, nose, right eye) triangle is negative for a correctly ordered
    frontal face and flips sign under mirroring.
    """
    if landmarks is None or len(landmarks) < 3:
        return False
    pts = np.asarray(landmarks, dtype=np.float64)
    v1 = pts[2] - pts[0]  # left eye -> nose
    v2 = pts[1] - pts[0]  # left eye -> right eye
    return bool((v1[0] * v2[1] - v1[1] * v2[0]) > 0)


def align(
    frame: npt.NDArray[np.uint8],
    box: FaceBox,
    output_size: int = 112,
    template: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.uint8] | None:
    """Warp the face in `frame` to the ArcFace template.

    Returns ``None`` when landmarks are missing or the transform is degenerate.
    Callers must treat ``None`` as "skip this face" rather than falling back to
    an unaligned crop — an unaligned crop embeds into a region ArcFace never
    saw, and the resulting match scores look plausible while being wrong.
    """
    if box.landmarks is None or len(box.landmarks) < 5:
        return None

    # A landmark set with no spatial extent yields a finite-but-meaningless
    # transform that warps the whole frame into a flat crop. Rejecting it here
    # is better than embedding that crop, which produces a confident nonsense
    # match rather than an error.
    spread = float(np.max(np.ptp(box.landmarks.astype(np.float64), axis=0)))
    if spread < 1e-3:
        return None

    if is_mirrored(box.landmarks):
        return None

    ref = ARCFACE_TEMPLATE if template is None else template
    try:
        matrix = umeyama_similarity(
            box.landmarks.astype(np.float64),
            ref.astype(np.float64),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    if not np.isfinite(matrix).all():
        return None

    # An exactly mirrored landmark set has no valid similarity transform: the
    # least-squares fit collapses to scale ~0, which warps the whole frame into
    # a flat crop. A finite all-zero matrix would sail past the check above and
    # then embed a constant image, so the degeneracy has to be caught here.
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if scale < 1e-6:
        return None
    if np.linalg.det(matrix[:, :2]) <= 0:
        return None

    # Umeyama returns the forward (source -> destination) transform, which is
    # exactly what warpAffine wants WITHOUT WARP_INVERSE_MAP. That flag tells
    # OpenCV the matrix is the inverse, so passing a forward matrix with it set
    # produces a plausible-shaped crop of an entirely different part of the
    # image — and the embeddings from it look valid while matching nobody.
    return cv2.warpAffine(
        frame,
        matrix.astype(np.float32),
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_batch(
    frame: npt.NDArray[np.uint8],
    boxes: list[FaceBox],
    output_size: int = 112,
) -> tuple[list[npt.NDArray[np.uint8]], list[int]]:
    """Align many faces, returning the crops and the indices that succeeded.

    Splitting crops from indices lets the caller batch only the faces that
    actually warped, instead of padding a fixed-size batch with zeros — zeros
    embedded as if they were a face produce confident nonsense matches.
    """
    crops: list[npt.NDArray[np.uint8]] = []
    indices: list[int] = []
    for i, box in enumerate(boxes):
        warped = align(frame, box, output_size)
        if warped is not None:
            crops.append(warped)
            indices.append(i)
    return crops, indices


def estimate_arc(
    embedding_a: npt.NDArray[np.float32],
    embedding_b: npt.NDArray[np.float32],
) -> float:
    """Geodesic angle in degrees between two L2-normalised embeddings.

    `arccos` of a dot product is numerically unstable near the extremes, which
    is exactly where the interesting decisions sit. Clamping first avoids NaN
    and the resulting spurious rejections.
    """
    a = np.asarray(embedding_a, dtype=np.float64).ravel()
    b = np.asarray(embedding_b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def l2_normalize(vec: npt.NDArray[np.float32], axis: int = -1) -> npt.NDArray[np.float32]:
    """Normalise to unit length along `axis`, leaving zero vectors at zero.

    Zero-length rows are preserved rather than becoming NaN. An all-zero
    embedding reaching the gallery search should score 0 against everything,
    not poison a FAISS index with NaNs that silently corrupt neighbouring
    entries.
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return np.divide(arr, norm, out=np.zeros_like(arr), where=norm > 1e-12)

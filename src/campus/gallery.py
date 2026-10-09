"""Gallery persistence: a single-file format the identity plane and the
testing UI can both load.

Deliberately not Postgres-only. A developer validating thresholds needs to
build a gallery from a folder of photos and point a UI at it in one command,
with no database, no migrations and no Redis. Postgres remains the system of
record in production; this is the same data in a form you can hand to someone.

The format is a compressed ``.npz`` of one centroid vector per student plus a
JSON blob of provenance. Centroids rather than individual photos, matching what
the index actually searches over — a file that stored photos and centroids
separately would be one refactor away from disagreeing with itself.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from campus.types import StudentId

SCHEMA_VERSION = 1


@dataclass(slots=True)
class GalleryFile:
    student_ids: list[str]
    vectors: npt.NDArray[np.float32]
    dim: int
    photos_per_student: dict[str, int]
    built_at: float
    source: str = ""
    model: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "students": len(self.student_ids),
            "dim": self.dim,
            "built_at": self.built_at,
            "source": self.source,
            "model": self.model,
            "mean_photos": (
                sum(self.photos_per_student.values()) / len(self.student_ids)
                if self.student_ids
                else 0
            ),
        }


def save(path: str | Path, gallery: GalleryFile) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": SCHEMA_VERSION,
        "dim": gallery.dim,
        "built_at": gallery.built_at,
        "source": gallery.source,
        "model": gallery.model,
        "student_ids": gallery.student_ids,
        "photos_per_student": gallery.photos_per_student,
    }
    np.savez_compressed(
        p,
        vectors=gallery.vectors.astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    return p


def load(path: str | Path) -> GalleryFile:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gallery not found: {p}")
    with np.load(p, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        vectors = data["vectors"].astype(np.float32)

    version = meta.get("version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"gallery {p.name} is schema v{version}, this build writes "
            f"v{SCHEMA_VERSION}. Rebuild it."
        )

    return GalleryFile(
        student_ids=list(meta["student_ids"]),
        vectors=vectors,
        dim=int(meta["dim"]),
        photos_per_student=dict(meta.get("photos_per_student", {})),
        built_at=float(meta.get("built_at", 0.0)),
        source=meta.get("source", ""),
        model=meta.get("model", ""),
    )


def build_from_photos(
    photos_dir: str | Path,
    detector,
    embedder,
    *,
    quality,
    min_score: float = 0.6,
    on_progress=None,
) -> tuple[GalleryFile, dict[str, list[str]]]:
    """Enroll every ``<STUDENT>.jpg`` in a folder. Returns the file and rejects.

    A rejection is recorded, never silently dropped: a folder where 30 of 391
    students failed should be obvious, because those students will later be
    unmatchable and nobody will connect the two facts.
    """
    from campus.imaging.align import align  # noqa: PLC0415
    from campus.imaging.quality import assess  # noqa: PLC0415
    from campus.capture.source import frame_gray  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    from campus.index.gallery import CentroidIndexer  # noqa: PLC0414

    root = Path(photos_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    files = sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not files:
        raise FileNotFoundError(f"no images in {root}")

    grouped: dict[str, list[tuple[Path, npt.NDArray[np.float32]]]] = {}
    rejects: dict[str, list[str]] = {}
    dim = 0

    for i, path in enumerate(files):
        sid = path.stem
        img = cv2.imread(str(path))
        if img is None:
            rejects.setdefault(sid, []).append(f"{path.name}: unreadable")
            continue
        try:
            boxes = detector.detect(img)
        except Exception as exc:  # noqa: BLE001
            rejects.setdefault(sid, []).append(f"{path.name}: detect {type(exc).__name__}")
            continue

        if len(boxes) != 1:
            rejects.setdefault(sid, []).append(
                f"{path.name}: {len(boxes)} faces (need exactly 1)"
            )
            continue

        box = boxes[0]
        report = assess(frame_gray(img), box, quality)
        if not report.passed:
            rejects.setdefault(sid, []).append(f"{path.name}: {', '.join(report.reasons[:2])}")
            continue

        crop = align(img, box, embedder.config.input_size)
        if crop is None:
            rejects.setdefault(sid, []).append(f"{path.name}: alignment failed")
            continue

        vec = embedder.embed([crop])[0]
        dim = dim or int(vec.shape[0])
        grouped.setdefault(sid, []).append((path, vec))

        if on_progress is not None and (i + 1) % 25 == 0:
            on_progress(i + 1, len(files), len(grouped))

    if not grouped:
        raise ValueError(
            f"nothing enrolled from {root}: "
            + "; ".join(f"{k}: {v[0]}" for k, v in list(rejects.items())[:3])
        )

    indexer = CentroidIndexer(dim)
    centroids, ids = indexer.build(
        (sid, np.stack([v for _, v in vecs])) for sid, vecs in grouped.items()
    )
    counts = {sid: len(vecs) for sid, vecs in grouped.items()}

    return (
        GalleryFile(
            student_ids=ids,
            vectors=centroids.astype(np.float32),
            dim=dim,
            photos_per_student=counts,
            built_at=time.time(),
            source=str(root),
            model=str(getattr(embedder, "path", "")),
        ),
        rejects,
    )


def merge(
    base: GalleryFile,
    addition: GalleryFile,
) -> GalleryFile:
    """Fold `addition` into `base`, replacing students that appear in both.

    Re-enrollment is routine — a student re-uploads a bad photo, someone joins
    mid-semester — so it must not require rebuilding the gallery from 391
    files. The replacement is by `student_id`, so a student's entry can only
    ever come from one source, and the result is identical to a full rebuild.
    """
    if base.dim != addition.dim:
        raise ValueError(
            f"dimension mismatch: existing gallery is {base.dim}-d, new is "
            f"{addition.dim}-d. Mixing dimensionalities in one gallery makes "
            f"every cosine score meaningless."
        )

    by_id = {sid: i for i, sid in enumerate(base.student_ids)}
    ids = list(base.student_ids)
    vecs = [v for v in base.vectors]
    counts = dict(base.photos_per_student)
    replaced: list[str] = []

    for sid, vec in zip(addition.student_ids, addition.vectors, strict=True):
        if sid in by_id:
            vecs[by_id[sid]] = vec
            replaced.append(sid)
        else:
            by_id[sid] = len(ids)
            ids.append(sid)
            vecs.append(vec)
        counts[sid] = addition.photos_per_student.get(sid, 1)

    merged = GalleryFile(
        student_ids=ids,
        vectors=np.stack(vecs).astype(np.float32) if vecs else base.vectors,
        dim=base.dim,
        photos_per_student=counts,
        built_at=time.time(),
        source=addition.source or base.source,
        model=addition.model or base.model,
    )
    merged_replaced = replaced
    return merged, merged_replaced


def to_index(gallery: GalleryFile, backend: str = "auto", use_gpu: bool = True):
    """Build a searchable index from a gallery file."""
    from campus.index.gallery import build_index  # noqa: PLC0415

    index = build_index(gallery.dim, backend, use_gpu)
    if len(gallery.student_ids):
        index.add(gallery.vectors, gallery.student_ids)
    return index

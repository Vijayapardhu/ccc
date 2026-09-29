"""Gallery search over face embeddings.

Two backends behind one interface:

* **FAISS** — what production uses. `IndexFlatIP` on L2-normalised vectors is
  exact, and at 50k x 512 dimensions that is a ~2ms GPU search or ~15ms CPU
  search. Exactness matters more than raw speed here: the whole design rests on
  the runner-up score being *correct*, and an approximate index that returns a
  slightly wrong second candidate corrupts the margin that the temporal
  verifier depends on. Scale to millions before reaching for HNSW/IVF, and note
  that at that point the margin semantics need re-validating anyway.
* **NumPy** — a fallback so the control plane, the tests, and a laptop without
  faiss installed all work unchanged. Same semantics, just slower.

Embeddings are L2-normalised on write, so inner product *is* cosine similarity
and no per-query normalisation pass is needed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from campus.imaging.align import l2_normalize
from campus.types import GalleryCandidate, StudentId


class GalleryIndex(Protocol):
    def search(self, queries: npt.NDArray[np.float32], k: int) -> list[list[GalleryCandidate]]: ...

    def add(self, vectors: npt.NDArray[np.float32], student_ids: Sequence[str]) -> None: ...

    def remove_student(self, student_id: str) -> int: ...

    def __len__(self) -> int: ...


@dataclass(slots=True)
class _Record:
    student_id: str
    vector: npt.NDArray[np.float32]
    enrolled_photos: int = 1


@dataclass
class NumPyGalleryIndex:
    """Exact brute-force index. The reference implementation.

    Kept as the semantic definition of the interface: if FAISS and this
    disagree on the same data, FAISS is wrong.
    """

    dim: int
    _records: dict[int, _Record] = field(default_factory=dict)
    _next_slot: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> list[list[GalleryCandidate]]:
        with self._lock:
            if not self._records:
                return [[] for _ in range(len(queries))]
            slots = sorted(self._records)
            matrix = np.stack([self._records[s].vector for s in slots])
            ids = [self._records[s].student_id for s in slots]
            photos = [self._records[s].enrolled_photos for s in slots]

        q = l2_normalize(np.atleast_2d(np.asarray(queries, dtype=np.float32)))
        sims = q @ matrix.T  # (n_queries, n_gallery)

        out: list[list[GalleryCandidate]] = []
        for row in sims:
            k_eff = min(k, row.shape[0])
            if k_eff == 0:
                out.append([])
                continue
            top = np.argpartition(-row, k_eff - 1)[:k_eff]
            top = top[np.argsort(-row[top], kind="stable")]
            out.append(
                [
                    GalleryCandidate(
                        student_id=StudentId(ids[i]),
                        score=float(row[i]),
                        rank=rank,
                        enrolled_photos=photos[i],
                    )
                    for rank, i in enumerate(top, start=1)
                ]
            )
        return out

    def add(self, vectors: npt.NDArray[np.float32], student_ids: Sequence[str]) -> None:
        vecs = l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        if len(vecs) != len(student_ids):
            raise ValueError("vectors and student_ids length mismatch")
        with self._lock:
            for vec, sid in zip(vecs, student_ids, strict=True):
                self._records[self._next_slot] = _Record(
                    student_id=sid, vector=vec.astype(np.float32)
                )
                self._next_slot += 1

    def replace(self, student_id: str, vectors: npt.NDArray[np.float32]) -> None:
        """Swap a student's entry for a new vector set (re-enrollment)."""
        vecs = l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        with self._lock:
            self.remove_student(student_id)
            for vec in vecs:
                self._records[self._next_slot] = _Record(
                    student_id=student_id,
                    vector=vec.astype(np.float32),
                    enrolled_photos=len(vecs),
                )
                self._next_slot += 1

    def remove_student(self, student_id: str) -> int:
        with self._lock:
            doomed = [s for s, r in self._records.items() if r.student_id == student_id]
            for s in doomed:
                del self._records[s]
            return len(doomed)

    def student_ids(self) -> list[str]:
        with self._lock:
            return sorted({r.student_id for r in self._records.values()})

    def __len__(self) -> int:
        with self._lock:
            return len({r.student_id for r in self._records.values()})


@dataclass
class FaissGalleryIndex:
    """FAISS-backed index. Exact ``IndexFlatIP``, GPU when available.

    Rebuilds are done by writing to a fresh index and swapping the reference,
    so readers never observe a half-updated index. The alternative — mutating
    a live ``IndexFlat`` — cannot express deletion, and supporting deletion
    matters: DPDP erasure has to actually remove the vectors.
    """

    dim: int
    use_gpu: bool = True
    _index: Any = None
    _ids: list[str] = field(default_factory=list)
    _photos: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _fallback: NumPyGalleryIndex | None = None

    def __post_init__(self) -> None:
        try:
            import faiss  # noqa: PLC0415
        except ImportError:
            self._fallback = NumPyGalleryIndex(self.dim)
            return
        self._faiss = faiss
        self._index = self._build_index()
        self._ids = []
        self._photos = {}

    def _build_index(self) -> Any:
        faiss = self._faiss
        index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except (RuntimeError, AttributeError):
                # No GPU in this process. CPU is still correct, just slower.
                pass
        return index

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> list[list[GalleryCandidate]]:
        if self._fallback is not None:
            return self._fallback.search(queries, k)
        with self._lock:
            if not self._ids:
                return [[] for _ in range(len(queries))]
            q = l2_normalize(np.atleast_2d(np.asarray(queries, dtype=np.float32)))
            scores, indices = self._index.search(np.ascontiguousarray(q), min(k, len(self._ids)))
            ids = list(self._ids)
            photos = dict(self._photos)

        out: list[list[GalleryCandidate]] = []
        for srow, irow in zip(scores, indices, strict=True):
            row: list[GalleryCandidate] = []
            for rank, (score, idx) in enumerate(zip(srow, irow, strict=True), start=1):
                if idx < 0:
                    continue
                sid = ids[idx]
                row.append(
                    GalleryCandidate(
                        student_id=StudentId(sid),
                        score=float(score),
                        rank=rank,
                        enrolled_photos=photos.get(sid, 1),
                    )
                )
            out.append(row)
        return out

    def add(self, vectors: npt.NDArray[np.float32], student_ids: Sequence[str]) -> None:
        if self._fallback is not None:
            self._fallback.add(vectors, student_ids)
            return
        vecs = np.ascontiguousarray(
            l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        )
        with self._lock:
            self._index.add(vecs)
            self._ids.extend(student_ids)
            for sid in student_ids:
                self._photos[sid] = self._photos.get(sid, 0) + 1

    def replace(self, student_id: str, vectors: npt.NDArray[np.float32]) -> None:
        if self._fallback is not None:
            self._fallback.replace(student_id, vectors)
            return
        vecs = np.ascontiguousarray(
            l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
        )
        with self._lock:
            new_index = self._build_index()
            keep_vecs: list[npt.NDArray[np.float32]] = []
            keep_ids: list[str] = []
            for i, sid in enumerate(self._ids):
                if sid == student_id:
                    continue
                keep_ids.append(sid)
                keep_vecs.append(self._reconstruct(i))
            if keep_vecs:
                new_index.add(np.ascontiguousarray(np.stack(keep_vecs)))
            new_index.add(vecs)
            self._index = new_index
            self._ids = keep_ids + [student_id] * len(vecs)
            self._photos[student_id] = len(vecs)
            self._photos = {k: v for k, v in self._photos.items() if v > 0}

    def _reconstruct(self, idx: int) -> npt.NDArray[np.float32]:
        vec = np.zeros(self.dim, dtype=np.float32)
        try:
            self._index.reconstruct(idx, vec)
        except RuntimeError:
            # GPU flat indexes cannot reconstruct; fall back to a CPU mirror.
            if self._fallback is None:
                self._fallback = NumPyGalleryIndex(self.dim)
                self._fallback._ids = list(self._ids)
            return self._fallback._reconstruct(idx)
        return vec

    def remove_student(self, student_id: str) -> int:
        if self._fallback is not None:
            return self._fallback.remove_student(student_id)
        with self._lock:
            doomed = [i for i, sid in enumerate(self._ids) if sid == student_id]
            if not doomed:
                return 0
            keep = [(i, sid) for i, sid in enumerate(self._ids) if sid != student_id]
            if self._fallback is None:
                self._fallback = NumPyGalleryIndex(self.dim)
            keep_vecs = [self._reconstruct(i) for i, _ in keep]
            new_index = self._build_index()
            if keep_vecs:
                new_index.add(np.ascontiguousarray(np.stack(keep_vecs)))
            self._index = new_index
            self._ids = [sid for _, sid in keep]
            self._photos.pop(student_id, None)
            return len(doomed)

    def student_ids(self) -> list[str]:
        if self._fallback is not None:
            return self._fallback.student_ids()
        with self._lock:
            return sorted(set(self._ids))

    def __len__(self) -> int:
        if self._fallback is not None:
            return len(self._fallback)
        with self._lock:
            return len(set(self._ids))


@dataclass
class CentroidIndexer:
    """Collapses a student's multiple enrollment photos into one vector.

    A student with front / left / right photos should match on any of them. A
    single mean-centroid vector does that adequately, and costs one gallery
    slot instead of N — which at 50k students x 3 photos is a 3x saving on
    index memory and search time.

    The trade-off: a centroid averages away the variation that makes the index
    sensitive to angle. Worth revisiting if per-angle recall proves weak; the
    fix is to keep the variants and search all of them, at the storage cost.
    """

    dim: int
    normalize: bool = True

    def centroid(
        self, vectors: npt.NDArray[np.float32], weights: Sequence[float] | None = None
    ) -> npt.NDArray[np.float32]:
        vecs = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vecs.size == 0:
            raise ValueError("cannot build a centroid from zero vectors")
        if weights is None:
            w = np.full(len(vecs), 1.0 / len(vecs), dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32)
            w = w / w.sum()
        out = (vecs * w[:, None]).sum(axis=0).astype(np.float32)
        return l2_normalize(out) if self.normalize else out

    def build(
        self, records: Iterable[tuple[str, npt.NDArray[np.float32]]]
    ) -> tuple[npt.NDArray[np.float32], list[str]]:
        """Group vectors by student, emit one centroid and id per student."""
        grouped: dict[str, list[npt.NDArray[np.float32]]] = {}
        for sid, vec in records:
            grouped.setdefault(sid, []).append(np.asarray(vec, dtype=np.float32).ravel())
        if not grouped:
            return np.zeros((0, self.dim), dtype=np.float32), []
        ids = sorted(grouped)
        centroids = np.stack([self.centroid(grouped[sid]) for sid in ids])
        return centroids.astype(np.float32), ids


def build_index(
    dim: int = 512,
    backend: str = "auto",
    use_gpu: bool = True,
) -> GalleryIndex:
    """Factory. ``auto`` prefers FAISS and falls back to NumPy silently."""
    if backend == "numpy":
        return NumPyGalleryIndex(dim)
    if backend == "faiss":
        return FaissGalleryIndex(dim, use_gpu=use_gpu)
    try:
        import faiss  # noqa: F401, PLC0415
    except ImportError:
        return NumPyGalleryIndex(dim)
    return FaissGalleryIndex(dim, use_gpu=use_gpu)


@dataclass(slots=True)
class SearchStats:
    queries: int = 0
    total_ms: float = 0.0
    gallery_size: int = 0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.queries if self.queries else 0.0

    def record(self, elapsed_ms: float, gallery_size: int) -> None:
        self.queries += 1
        self.total_ms += elapsed_ms
        self.gallery_size = gallery_size


def timed_search(
    index: GalleryIndex, queries: npt.NDArray[np.float32], k: int = 5
) -> tuple[list[list[GalleryCandidate]], float]:
    start = time.perf_counter()
    results = index.search(queries, k)
    return results, (time.perf_counter() - start) * 1000.0

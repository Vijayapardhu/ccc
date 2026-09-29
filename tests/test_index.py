"""Gallery indexing, centroiding, and search semantics."""

from __future__ import annotations

import numpy as np
import pytest

from campus.index.gallery import (
    CentroidIndexer,
    NumPyGalleryIndex,
    build_index,
    timed_search,
)

DIM = 512


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def near(base: np.ndarray, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + noise * rng.normal(size=base.shape)
    return (v / np.linalg.norm(v)).astype(np.float32)


class TestCosineSemantics:
    def test_identical_vector_scores_one(self):
        idx = NumPyGalleryIndex(DIM)
        v = unit(1)
        idx.add(v[None, :], ["S1"])
        (hit,) = idx.search(v[None, :], 5)
        assert hit[0].student_id == "S1"
        assert hit[0].score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_scores_near_zero(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(unit(1)[None, :], ["S1"])
        (hit,) = idx.search(unit(2)[None, :], 5)
        assert abs(hit[0].score) < 0.2

    def test_vectors_are_normalised_on_write(self):
        """Unnormalised input must still yield a true cosine score, or the
        scores the temporal verifier thresholds on are meaningless."""
        idx = NumPyGalleryIndex(DIM)
        idx.add((unit(1) * 17.0)[None, :], ["S1"])
        (hit,) = idx.search(unit(1)[None, :], 1)
        assert hit[0].score == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_does_not_poison_index(self):
        """An all-zero embedding must score 0, not NaN. NaN silently corrupts
        neighbouring FAISS entries and the failure surfaces weeks later as
        impossible match scores."""
        idx = NumPyGalleryIndex(DIM)
        idx.add(unit(1)[None, :], ["S1"])
        (hit,) = idx.search(np.zeros((1, DIM), dtype=np.float32), 1)
        assert np.isfinite(hit[0].score)
        assert hit[0].score == pytest.approx(0.0, abs=1e-5)


class TestSearchResults:
    def test_ranking_is_correct(self):
        idx = NumPyGalleryIndex(DIM)
        base = unit(10)
        idx.add(
            np.stack([unit(20), base, near(base, 0.3, 21), unit(22)]),
            ["far", "self", "close", "other"],
        )
        (hits,) = idx.search(base[None, :], 4)
        assert hits[0].student_id == "self"
        assert hits[1].student_id == "close"
        assert [h.rank for h in hits] == [1, 2, 3, 4]
        assert hits[0].score >= hits[1].score >= hits[2].score

    def test_topk_is_respected(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(i) for i in range(50)]), [f"S{i}" for i in range(50)])
        (hits,) = idx.search(unit(1)[None, :], 5)
        assert len(hits) == 5

    def test_k_larger_than_gallery(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(1), unit(2)]), ["S1", "S2"])
        (hits,) = idx.search(unit(1)[None, :], 10)
        assert len(hits) == 2

    def test_empty_gallery_returns_empty(self):
        idx = NumPyGalleryIndex(DIM)
        assert idx.search(unit(1)[None, :], 5) == [[]]

    def test_batched_search(self):
        idx = NumPyGalleryIndex(DIM)
        a, b = unit(30), unit(31)
        idx.add(np.stack([a, b]), ["S1", "S2"])
        hits = idx.search(np.stack([a, b]), 2)
        assert hits[0][0].student_id == "S1"
        assert hits[1][0].student_id == "S2"

    def test_empty_query_batch(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(unit(1)[None, :], ["S1"])
        assert len(idx.search(np.zeros((0, DIM), dtype=np.float32), 5)) == 0


class TestSiblingDiscrimination:
    def test_family_members_separate(self):
        """The realistic hard case: siblings in the same gallery, seen at
        similar distances. The margin over the runner-up is what the temporal
        verifier consumes, so it must be a real number."""
        idx = NumPyGalleryIndex(DIM)
        sibling_a, sibling_b = unit(40), unit(41)
        idx.add(np.stack([sibling_a, sibling_b]), ["SIB_A", "SIB_B"])
        (hits,) = idx.search(near(sibling_a, 0.25, 42)[None, :], 2)
        margin = hits[0].score - hits[1].score
        assert hits[0].student_id == "SIB_A"
        assert margin > 0.0

    def test_stranger_scores_much_lower_than_the_student(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(50), unit(51)]), ["STUDENT", "STRANGER"])
        (hits,) = idx.search(unit(50)[None, :], 2)
        assert hits[0].score - hits[1].score > 0.2


class TestMutation:
    def test_replace_swaps_all_of_a_students_entries(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(1), unit(2)]), ["S1", "S1"])
        idx.replace("S1", np.stack([unit(9)]))
        (hits,) = idx.search(unit(9)[None, :], 1)
        assert hits[0].student_id == "S1"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_remove_student_actually_removes_vectors(self):
        """DPDP erasure has to delete the vectors, not just flag a row. An
        index that retains them after a 'removal' is a compliance failure."""
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(1), unit(2)]), ["S1", "S2"])
        assert idx.remove_student("S1") == 1
        assert "S1" not in idx.student_ids()
        (hits,) = idx.search(unit(1)[None, :], 2)
        assert all(h.student_id != "S1" for h in hits)

    def test_len_counts_students_not_vectors(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(1), unit(2), unit(3)]), ["S1", "S1", "S2"])
        assert len(idx) == 2

    def test_length_mismatch_rejected(self):
        idx = NumPyGalleryIndex(DIM)
        with pytest.raises(ValueError):
            idx.add(np.stack([unit(1)]), ["S1", "S2"])


class TestCentroid:
    def test_centroid_of_identical_vectors_is_that_vector(self):
        c = CentroidIndexer(DIM)
        v = unit(60)
        assert np.allclose(c.centroid(np.stack([v, v, v])), v, atol=1e-5)

    def test_centroid_is_unit_length(self):
        c = CentroidIndexer(DIM)
        out = c.centroid(np.stack([unit(61), unit(62), unit(63)]))
        assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-5)

    def test_weights_are_respected(self):
        c = CentroidIndexer(DIM)
        a, b = unit(64), unit(65)
        out = c.centroid(np.stack([a, b]), weights=[0.9, 0.1])
        assert float(np.dot(out, a)) > float(np.dot(out, b))

    def test_build_groups_by_student(self):
        """Multi-photo enrollment collapses to one gallery slot — a 3x saving
        on index size at 50k students."""
        c = CentroidIndexer(DIM)
        vecs, ids = c.build(
            [
                ("S1", unit(70)), ("S2", unit(71)),
                ("S1", unit(70)), ("S1", unit(72)),
            ]
        )
        assert ids == ["S1", "S2"]
        assert vecs.shape == (2, DIM)

    def test_build_with_no_records(self):
        vecs, ids = CentroidIndexer(DIM).build([])
        assert len(ids) == 0
        assert vecs.shape == (0, DIM)

    def test_zero_vectors_rejected(self):
        with pytest.raises(ValueError):
            CentroidIndexer(DIM).centroid(np.zeros((0, DIM), dtype=np.float32))


class TestBackendSelection:
    def test_numpy_backend_requested(self):
        assert isinstance(build_index(DIM, "numpy"), NumPyGalleryIndex)

    def test_auto_falls_back_without_faiss(self):
        """The control plane must run on a laptop with no faiss installed."""
        idx = build_index(DIM, "auto")
        idx.add(unit(80)[None, :], ["S1"])
        (hit,) = idx.search(unit(80)[None, :], 1)
        assert hit[0].student_id == "S1"

    def test_semantics_match_between_representations(self):
        """NumPyGalleryIndex is the semantic definition of the interface. If a
        faiss build disagrees, faiss is wrong."""
        a, b, c = unit(90), unit(91), unit(92)
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([a, b, c]), ["S1", "S2", "S3"])
        (hits,) = idx.search(b[None, :], 3)
        assert hits[0].student_id == "S2"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)
        assert hits[1].score > hits[2].score


class TestTiming:
    def test_timed_search_reports_latency(self):
        idx = NumPyGalleryIndex(DIM)
        idx.add(np.stack([unit(i) for i in range(1000)]), [f"S{i}" for i in range(1000)])
        hits, ms = timed_search(idx, unit(0)[None, :], 5)
        assert len(hits[0]) == 5
        assert ms > 0

    def test_gallery_of_50k_searches_within_frame_budget(self):
        """50k students is the design point. The search must fit inside a
        frame's budget or the temporal verifier's whole premise (several
        frames per track) stops being reachable."""
        idx = NumPyGalleryIndex(DIM)
        batch = np.stack([unit(i) for i in range(50_000)]).astype(np.float32)
        idx.add(batch, [f"S{i}" for i in range(50_000)])
        assert len(idx) == 50_000
        _, ms = timed_search(idx, unit(1)[None, :], 5)
        assert ms < 1000, f"50k gallery search took {ms:.0f}ms"

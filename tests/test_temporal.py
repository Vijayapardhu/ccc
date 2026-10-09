"""Temporal verification — the layer that decides what the system believes."""

from __future__ import annotations

import pytest

from campus.temporal.verifier import TemporalVerifier
from campus.types import CameraId, TrackId, ObservationOutcome

from .conftest import candidates, consistent_candidates


def feed(
    verifier: TemporalVerifier,
    track: str = "cam-1-0",
    n: int = 12,
    student: str = "S1001",
    score: float = 0.55,
    rival: str = "S9999",
    rival_score: float = 0.30,
    t0: float = 1000.0,
    step: float = 0.125,
) -> None:
    for i in range(n):
        verifier.observe(
            TrackId(track), CameraId("cam-1"),
            consistent_candidates(student, score, rival, rival_score),
            t0 + i * step,
        )


class TestCommitment:
    def test_consistent_evidence_commits(self):
        v = TemporalVerifier()
        feed(v)
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.student_id == "S1001"
        assert commit.evidence_frames == 12
        assert commit.margin > 0.06

    def test_single_frame_never_commits(self):
        """The whole design: one frame of a 30px face proves nothing."""
        v = TemporalVerifier()
        v.observe(TrackId("t"), CameraId("c"), consistent_candidates("S1001"), 100.0)
        assert v.commit(TrackId("t")) is None

    def test_too_few_frames_never_commits(self):
        v = TemporalVerifier()
        feed(v, n=4)
        assert v.commit(TrackId("cam-1-0")) is None

    def test_insufficient_agreement_never_commits(self):
        """A face whose top-1 changes every frame is not evidence of anything."""
        v = TemporalVerifier()
        for i in range(12):
            v.observe(
                TrackId("t"), CameraId("c"),
                candidates([(f"S{i:04d}", 0.55), (f"R{i:04d}", 0.30)]),
                1000.0 + i * 0.125,
            )
        assert v.commit(TrackId("t")) is None

    def test_margin_blocks_commit_when_rival_is_close(self):
        """The load-bearing test. A rival within the margin band means the
        evidence does not distinguish the two people, so no commitment."""
        v = TemporalVerifier()
        feed(v, score=0.55, rival_score=0.52)
        assert v.commit(TrackId("cam-1-0")) is None

    def test_low_absolute_score_still_commits_with_good_margin(self):
        """Median threshold sits below the sibling band on purpose: at campus
        face sizes the correct answer often scores lower than a naive
        single-frame threshold would demand, and the margin is what makes it
        trustworthy."""
        v = TemporalVerifier()
        feed(v, score=0.40, rival_score=0.20)
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.median_score == pytest.approx(0.40, abs=0.01)
        assert commit.margin > 0.15

    def test_score_below_floor_blocks_commit(self):
        v = TemporalVerifier()
        feed(v, score=0.20, rival_score=0.05)
        assert v.commit(TrackId("cam-1-0")) is None


class TestOneCommitPerTrack:
    def test_commit_is_emitted_exactly_once(self):
        v = TemporalVerifier()
        feed(v)
        assert v.commit(TrackId("cam-1-0")) is not None
        assert v.commit(TrackId("cam-1-0")) is None, "duplicate commit for one track"

    def test_finalize_drains_and_emits(self):
        v = TemporalVerifier()
        feed(v)
        commit = v.finalize(TrackId("cam-1-0"))
        assert commit is not None
        assert v.track_count == 0

    def test_finalize_discards_uncommitted_tracks(self):
        """A track that never earned an identity still has to be cleaned up,
        or the dict grows until the worker dies."""
        v = TemporalVerifier()
        v.observe(TrackId("t"), CameraId("c"), candidates([("S1", 0.5)]), 100.0)
        assert v.finalize(TrackId("t")) is None
        assert v.track_count == 0

    def test_tracks_are_independent(self):
        v = TemporalVerifier()
        feed(v, track="cam-1-0", student="S1001")
        feed(v, track="cam-1-1", student="S2002", t0=2000.0)
        assert v.commit(TrackId("cam-1-0")).student_id == "S1001"
        assert v.commit(TrackId("cam-1-1")).student_id == "S2002"


class TestStability:
    def test_commitment_is_sticky_through_bad_frames(self):
        """A person walking the length of a corridor passes through blur and
        occlusion. Without stickiness the commitment flickers and attendance
        records contradict each other."""
        v = TemporalVerifier()
        feed(v, n=8)
        # Eight bad frames: the face is garbage, top-1 is some stranger.
        for i in range(8):
            v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                candidates([("S7777", 0.20), ("S1001", 0.10)]),
                1000.0 + (8 + i) * 0.125,
            )
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.student_id == "S1001"

    def test_commitment_survives_bad_frames_in_the_record(self):
        """A person walking the length of a corridor passes through blur and
        occlusion. The *commit* must survive that, or attendance flickers."""
        v = TemporalVerifier()
        feed(v, n=8)
        for i in range(8):
            v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                candidates([("S7777", 0.20), ("S1001", 0.10)]),
                1000.0 + (8 + i) * 0.125,
            )
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.student_id == "S1001"

    def test_single_bad_frame_stays_sticky(self):
        """One bad frame is a bad frame, not a dispute.

        If it were treated as one, the flicker that stickiness exists to
        prevent would come straight back, just wearing a different label.
        """
        v = TemporalVerifier()
        feed(v, n=10)
        outcome, sid, _, _ = v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            candidates([("S7777", 0.20), ("S1001", 0.10)]), 1000.0 + 10 * 0.125,
        )
        assert outcome is ObservationOutcome.IDENTIFIED
        assert sid == "S1001"

    def test_sustained_disagreement_is_contested(self):
        """The observed failure this replaces.

        A track committed while the subject faced the camera; the evidence then
        favoured a different person frame after frame — usually because the
        subject turned away and now matches nobody. The challenger is far too
        weak to overturn, so the old behaviour re-asserted the standing
        identity as if it were current. Displaying a stale name as a confident
        identity is how a false attendance record gets attached to a real
        student, so the honest outcome is that the evidence disagrees.
        """
        v = TemporalVerifier()
        feed(v, n=10)
        for i in range(4):
            outcome, sid, score, _ = v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                candidates([("S7777", 0.23), ("S1001", 0.10)]),
                1000.0 + (10 + i) * 0.125,
            )
        assert outcome is ObservationOutcome.CONTESTED
        assert sid == "S1001", "the commitment is still what an event would carry"
        assert score == pytest.approx(0.23, abs=0.01)
        assert v.commit(TrackId("cam-1-0")) is not None

    def test_inconsistent_challenger_does_not_contest(self):
        """Noise, not disagreement. A challenger that cannot agree with itself
        across frames is a bad crop, and sticking is the right call."""
        v = TemporalVerifier()
        feed(v, n=10)
        for i in range(5):
            v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                candidates([(f"S77{i:02d}", 0.30), ("S1001", 0.10)]),
                1000.0 + (10 + i) * 0.125,
            )
        outcome, sid, _, _ = v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            consistent_candidates("S1001", 0.55, "S7777", 0.20), 1000.0 + 16 * 0.125,
        )
        assert outcome is ObservationOutcome.IDENTIFIED
        assert sid == "S1001"

    def test_agreeing_frames_are_not_contested(self):
        v = TemporalVerifier()
        feed(v, n=10)
        outcome, sid, _, _ = v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            consistent_candidates("S1001", 0.50, "S9999", 0.20), 1001.2,
        )
        assert outcome is ObservationOutcome.IDENTIFIED
        assert sid == "S1001"

    def test_stale_track_is_not_contested(self):
        """A track that has gone quiet is quiet, not disputed.

        The gap is deliberately between `min_gap_s` (1.0s) and `max_gap_s`
        (2.5s): long enough that the current evidence is not "live", short
        enough that the track has not been recycled onto a different person.
        Past `max_gap_s` the window resets instead, which is a separate and
        correct behaviour.
        """
        v = TemporalVerifier()
        feed(v, n=10)
        outcome, _, _, _ = v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            candidates([("S7777", 0.30), ("S1001", 0.10)]), 1000.0 + 1.5,
        )
        assert outcome is ObservationOutcome.IDENTIFIED

    def test_very_long_gap_resets_the_track_entirely(self):
        """Past `max_gap_s` the track id is treated as a different person."""
        v = TemporalVerifier()
        feed(v, n=10)
        outcome, sid, _, _ = v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            candidates([("S7777", 0.30), ("S1001", 0.10)]), 1000.0 + 600.0,
        )
        assert outcome is ObservationOutcome.UNKNOWN
        assert sid is None
        assert v.commit(TrackId("cam-1-0")) is None, "stale commitment survived"

    def test_sustained_strong_rival_overturns(self):
        """An identity switch is rare but real (a group splitting up). When it
        happens, the system must follow it rather than stay stuck."""
        v = TemporalVerifier()
        feed(v, n=10, score=0.55, rival_score=0.30)
        assert v.commit(TrackId("cam-1-0")) is not None
        outcomes = []
        for i in range(14):
            outcome, sid, _, _ = v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                consistent_candidates("S2002", 0.85, "S1001", 0.40),
                1001.5 + i * 0.125,
            )
            outcomes.append((outcome, sid))
        assert sid == "S2002", "a sustained strong rival did not take over the track"
        assert outcomes[-1][0] is ObservationOutcome.IDENTIFIED

    def test_identity_change_is_surfaced_exactly_once(self):
        """The switch is visible for alerting without duplicating the record:
        presence rows are keyed on track_id, so a second commit would be a
        duplicate rather than a correction."""
        v = TemporalVerifier()
        feed(v, n=10, score=0.55, rival_score=0.30)
        assert v.commit(TrackId("cam-1-0")) is not None
        outcomes = [
            v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                consistent_candidates("S2002", 0.85, "S1001", 0.40),
                1001.5 + i * 0.125,
            )[0]
            for i in range(14)
        ]
        changes = [o for o in outcomes if o is ObservationOutcome.IDENTITY_CHANGED]
        assert len(changes) == 1, f"identity change reported {len(changes)} times"
        assert v.commit(TrackId("cam-1-0")) is None, "overturn produced a second commit"

    def test_one_anomalous_frame_does_not_overturn(self):
        v = TemporalVerifier()
        feed(v, n=10)
        v.observe(
            TrackId("cam-1-0"), CameraId("c"),
            consistent_candidates("S2002", 0.90, "S1001", 0.10), 1001.4,
        )
        for i in range(9):
            v.observe(
                TrackId("cam-1-0"), CameraId("c"),
                consistent_candidates("S1001", 0.55, "S2002", 0.30),
                1001.4 + (i + 1) * 0.125,
            )
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.student_id == "S1001"


class TestTrackHygiene:
    def test_long_gap_resets_evidence(self):
        """A track id recycled by a re-association failure is a different
        person. Blending their evidence would produce a confident wrong answer."""
        v = TemporalVerifier()
        feed(v, n=8, student="S1001")
        feed(v, n=8, student="S2002", t0=2000.0)
        commit = v.commit(TrackId("cam-1-0"))
        assert commit is not None
        assert commit.student_id == "S2002", "stale evidence leaked into a new track"

    def test_window_is_bounded(self):
        v = TemporalVerifier()
        for i in range(200):
            v.observe(
                TrackId("t"), CameraId("c"),
                consistent_candidates("S1001"), 1000.0 + i * 0.125,
            )
        commit = v.commit(TrackId("t"))
        assert commit is not None
        assert commit.evidence_frames == 12, "window grew without bound"

    def test_finalize_all_clears_everything(self):
        v = TemporalVerifier()
        for k in range(5):
            feed(v, track=f"cam-1-{k}", t0=1000.0 + k * 100)
        assert len(v.finalize_all()) == 5
        assert v.track_count == 0

    def test_no_candidates_yields_unknown(self):
        v = TemporalVerifier()
        outcome, sid, score, _ = v.observe(TrackId("t"), CameraId("c"), [], 100.0)
        assert outcome is ObservationOutcome.UNKNOWN
        assert sid is None
        assert score == 0.0


class TestAmbiguity:
    def test_close_rivals_report_ambiguous(self):
        v = TemporalVerifier()
        for i in range(12):
            outcome, sid, _, _ = v.observe(
                TrackId("t"), CameraId("c"),
                candidates([("S1111", 0.50), ("S2222", 0.48)]),
                1000.0 + i * 0.125,
            )
        assert outcome is ObservationOutcome.AMBIGUOUS
        assert sid is None


class TestCommitContents:
    def test_commit_carries_full_evidence_chain(self):
        """An attendance record that cannot be explained is a record the
        university cannot defend. Every field here answers 'why did you say
        this person was here?'."""
        v = TemporalVerifier()
        feed(v, score=0.55, rival_score=0.30)
        c = v.commit(TrackId("cam-1-0"))
        assert c.runner_up_id == "S9999"
        assert c.runner_up_score == pytest.approx(0.30, abs=0.01)
        assert c.margin == pytest.approx(c.mean_score - c.runner_up_score, abs=0.01)
        assert c.window_frames == 12
        assert c.score_std >= 0.0
        assert c.duration_s > 0
        assert c.quality_frames == 12

    def test_commit_serialises(self):
        v = TemporalVerifier()
        feed(v)
        d = v.commit(TrackId("cam-1-0")).to_dict()
        for key in (
            "track_id", "camera_id", "student_id", "median_score", "margin",
            "evidence_frames", "duration_s", "runner_up_id",
        ):
            assert key in d

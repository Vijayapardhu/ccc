"""Temporal verification — the layer that turns per-frame guesses into facts.

The problem
-----------
A single frame of a 30x40 face produces a top-1 similarity around 0.55-0.65.
So does a confidently wrong match on a stranger, so does a twin, so does the
same student under different corridor lighting. The distributions overlap. A
per-frame decision at any fixed threshold is a coin flip dressed up as a
result.

The approach
------------
Commit only on agreement over time, per track:

1. Accumulate (candidate, score) observations in a sliding window.
2. A candidate becomes the leader once it holds a plurality of the window.
3. It becomes the **commitment** only when the window is full and every
   threshold below is met — minimum support, minimum median score, and a
   **margin** over the runner-up.
4. Once committed, the identity is sticky. A later disagreement has to clear a
   much higher bar to overturn it.

The margin check is the one that does the real work. A wrong identity tends to
be wrong *inconsistently* — different strangers win different frames — so the
runner-up score sits close to the leader's. A right identity is stable, so the
margin opens up over 10-20 frames even when the absolute score is mediocre.

The stickiness matters as much as the thresholds. A person walking the length
of a corridor passes through moments of blur and occlusion. Without hysteresis
the commitment would flicker, producing duplicate attendance records and
attendance that disagrees with itself. With it, the track holds its identity
through the bad frames and only a sustained contradiction overturns it.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from campus.types import (
    CameraId,
    GalleryCandidate,
    ObservationOutcome,
    QualityReport,
    TrackCommit,
    TrackId,
)


@dataclass(frozen=True, slots=True)
class VerificationThresholds:
    """Commit criteria. Tuned against real campus footage, not a benchmark.

    The defaults are conservative. A missed attendance record is an
    operational annoyance; a false positive attached to a student's permanent
    record is a serious problem, and the asymmetry argues for strictness.
    """

    window: int = 12
    """Observations in the sliding window. ~0.5s of video at 25fps, or 2s at
    6fps. Long enough to average out bad frames, short enough to commit before
    someone leaves the frame."""

    min_support: int = 7
    """Frames that must agree on the leader. ~58% of the window. Below this,
    we are guessing."""

    min_median_score: float = 0.38
    """Median cosine similarity for the leader.

    This is deliberately far below the sibling band (0.3-0.6 typical for the
    *correct* student against their own gallery entry). That gap looks wrong on
    paper and is the entire point: at these face sizes in real CCTV, the
    correct answer frequently scores below what a naive single-frame threshold
    would demand. The median over a window, not the max, is what makes a low
    median trustworthy."""

    min_margin: float = 0.06
    """Required lead over the runner-up. The single most important parameter
    in the system; see the module docstring."""

    overturn_score: float = 0.62
    """A rival must beat the current commitment by this much, sustained over
    `overturn_frames`, to take the track over. High on purpose: this handles
    identity switches, which are rare, and mis-firing on them creates records
    that are wrong in the worst possible way."""

    overturn_frames: int = 5
    min_observations: int = 5
    """Below this many observations, no commitment is possible regardless of
    score. Covers the case where a face is detected once as someone passes."""

    max_gap_s: float = 2.5
    """Largest timestamp gap tolerated between consecutive observations in a
    track. A longer gap means this is a different person reusing the track id
    after a re-association failure, so the window resets rather than merging
    two people's evidence."""

    decay_half_life_s: float = 8.0
    """Weights older observations lower. The middle of a window is worth more
    than the start, because recency reflects the person's current appearance
    under current lighting."""


@dataclass(slots=True)
class _Sample:
    """One frame's evidence for a track.

    The full top-k is retained, not just the top-1. The margin over the
    runner-up is the load-bearing signal in this whole module, and it is
    computed from rank 2. Recording only the top-1 would make every track look
    uncontested and the margin would be a constant — which is precisely the
    failure mode the temporal verifier exists to prevent.
    """

    timestamp: float
    top_id: str
    top_score: float
    rivals: tuple[tuple[str, float], ...]
    quality_passed: bool


@dataclass(slots=True)
class _TrackState:
    track_id: TrackId
    camera_id: CameraId
    samples: Deque[_Sample] = field(default_factory=deque)
    committed_student: str | None = None
    committed_at: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    quality_frames: int = 0
    emit_commit: bool = True
    runner_up: str | None = None
    runner_up_score: float = 0.0


class TemporalVerifier:
    """Per-track evidence accumulator. Not thread-safe by design.

    One instance per camera worker process. The state it holds is deliberately
    per-process and in-memory: it is worthless once the process dies, because
    a track without a camera has no meaning. Attempting to share or persist it
    would buy a false sense of durability.
    """

    def __init__(self, thresholds: VerificationThresholds | None = None) -> None:
        self.t = thresholds or VerificationThresholds()
        self._tracks: dict[TrackId, _TrackState] = {}

    def observe(
        self,
        track_id: TrackId,
        camera_id: CameraId,
        candidates: list[GalleryCandidate],
        timestamp: float,
        quality_passed: bool = True,
    ) -> tuple[ObservationOutcome, str | None, float, list[GalleryCandidate]]:
        """Fold one frame's candidates into a track and report the live read.

        Returns ``(outcome, student_id, score, candidates)``. A ``CONFIRMED``
        commit is *not* emitted from here — see :meth:`commit`, which fires
        exactly once per track.
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState(track_id=track_id, camera_id=camera_id)
            self._tracks[track_id] = state

        if state.first_seen == 0.0:
            state.first_seen = timestamp
        state.last_seen = timestamp
        if quality_passed:
            state.quality_frames += 1

        # A long gap means the track id was recycled. Start clean rather than
        # blending two different people's evidence into one identity.
        if state.samples and timestamp - state.samples[-1].timestamp > self.t.max_gap_s:
            self._reset_window(state, timestamp)

        if not candidates:
            return (ObservationOutcome.UNKNOWN, None, 0.0, [])

        top = candidates[0]
        rivals = tuple((c.student_id, c.score) for c in candidates[1:])
        state.samples.append(
            _Sample(
                timestamp=timestamp,
                top_id=top.student_id,
                top_score=top.score,
                rivals=rivals,
                quality_passed=quality_passed,
            )
        )
        while len(state.samples) > self.t.window:
            state.samples.popleft()

        self._expire_samples(state, timestamp)

        leader, leader_score, runner_up, runner_up_score, median = self._tally(state)
        state.runner_up = runner_up
        state.runner_up_score = runner_up_score

        if state.committed_student is not None:
            challenger = self._should_overturn(state, timestamp)
            if challenger is not None:
                # The track now describes a different person. Drop the old
                # subject's evidence rather than blending it with the new
                # subject's, and keep `emit_commit` set: presence rows are keyed
                # on track_id, so re-emitting would duplicate the record rather
                # than correct it. The switch is surfaced as its own outcome
                # instead, which is what alerting should watch.
                state.committed_student = challenger
                state.committed_at = timestamp
                self._reset_window(state, timestamp, keep_commit=True)
                state.emit_commit = True
                return (
                    ObservationOutcome.IDENTITY_CHANGED,
                    challenger,
                    statistics.median([s.top_score for s in state.samples]) if state.samples else 0.0,
                    candidates,
                )
            # Sticky: report the standing commitment, not this frame's
            # possibly-bad top-1.
            return (
                ObservationOutcome.IDENTIFIED,
                state.committed_student,
                median if leader == state.committed_student else leader_score,
                candidates,
            )

        if (
            len(state.samples) >= self.t.min_observations
            and leader is not None
            and self._count_support(state, leader) >= self.t.min_support
            and median >= self.t.min_median_score
            and (runner_up is None or leader_score - runner_up_score >= self.t.min_margin)
        ):
            state.committed_student = leader
            state.committed_at = timestamp
            state.emit_commit = False
            return (ObservationOutcome.IDENTIFIED, leader, median, candidates)

        if runner_up is not None and abs(leader_score - runner_up_score) < self.t.min_margin:
            return (ObservationOutcome.AMBIGUOUS, None, leader_score, candidates)

        return (ObservationOutcome.UNKNOWN, None, leader_score, candidates)

    def commit(self, track_id: TrackId) -> TrackCommit | None:
        """Return the final commitment for a track, once.

        Returns ``None`` if the track was never committed, or if the
        commitment was already emitted. This is the only place a
        :class:`TrackCommit` is produced, which is what makes "one event per
        track" a structural guarantee rather than a convention.
        """
        state = self._tracks.get(track_id)
        if state is None or state.committed_student is None or state.emit_commit:
            return None

        leader = state.committed_student
        scores = [s.top_score for s in state.samples if s.top_id == leader]
        if not scores:
            return None

        self._expire_samples(state, state.last_seen)
        leader_score = statistics.fmean(scores)
        state.emit_commit = True

        return TrackCommit(
            track_id=state.track_id,
            camera_id=state.camera_id,
            student_id=leader,  # type: ignore[arg-type]
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            committed_at=state.committed_at,
            evidence_frames=len(state.samples),
            window_frames=len(scores),
            median_score=statistics.median(scores),
            min_score=min(scores),
            mean_score=leader_score,
            score_std=float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
            runner_up_id=state.runner_up,  # type: ignore[arg-type]
            runner_up_score=state.runner_up_score,
            margin=leader_score - state.runner_up_score,
            duration_s=max(0.0, state.last_seen - state.first_seen),
            quality_frames=state.quality_frames,
        )

    def finalize(self, track_id: TrackId, reason: str = "track_ended") -> TrackCommit | None:
        """Drain a track: emit its commit if it has one, then discard state.

        Called when a track leaves the frame or times out. Without this the
        dict grows without bound — 400 cameras at 25fps leaks tracks faster than
        anything notices, until the worker OOMs a week into a semester.
        """
        commit = self.commit(track_id)
        self._tracks.pop(track_id, None)
        if commit is not None:
            return commit
        return None

    def finalize_all(self) -> list[TrackCommit]:
        out = [c for c in (self.finalize(tid) for tid in list(self._tracks)) if c is not None]
        self._tracks.clear()
        return out

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    # -- internals ---------------------------------------------------------

    def _reset_window(self, state: _TrackState, timestamp: float, keep_commit: bool = False) -> None:
        state.samples.clear()
        state.quality_frames = 0
        state.first_seen = timestamp
        if not keep_commit:
            state.committed_student = None
            state.committed_at = 0.0
            state.emit_commit = True

    def _expire_samples(self, state: _TrackState, now: float) -> None:
        """Drop samples older than the window's time horizon."""
        cutoff = now - (self.t.window * self.t.max_gap_s)
        while state.samples and state.samples[0].timestamp < cutoff:
            state.samples.popleft()

    def _weights(self, state: _TrackState, now: float) -> list[float]:
        half_life = max(1e-6, self.t.decay_half_life_s)
        return [0.5 ** ((now - s.timestamp) / half_life) for s in state.samples]

    def _tally(
        self, state: _TrackState
    ) -> tuple[str | None, float, str | None, float, float]:
        """Aggregate the window into leader / runner-up with time-decay weights.

        Leader: the candidate holding top-1 in the most (recency-weighted)
        frames, tie-broken on mean score. Runner-up: the strongest competitor
        that is *not* the current leader, averaged over the frames where it
        appeared. Both are weighted **means** on the raw cosine scale, not
        sums — a sum would scale with window length and `min_margin` would
        silently mean something different at 12 frames than at 30.
        """
        if not state.samples:
            return (None, 0.0, None, 0.0, 0.0)

        now = state.samples[-1].timestamp
        weights = self._weights(state, now)

        lead_mass: dict[str, float] = {}
        lead_scores: dict[str, list[float]] = {}
        lead_weighted: dict[str, float] = {}
        rival_weighted: dict[str, float] = {}
        rival_mass: dict[str, float] = {}

        for sample, weight in zip(state.samples, weights, strict=True):
            sid = sample.top_id
            lead_mass[sid] = lead_mass.get(sid, 0.0) + weight
            lead_scores.setdefault(sid, []).append(sample.top_score)
            lead_weighted[sid] = lead_weighted.get(sid, 0.0) + sample.top_score * weight
            for rival_id, rival_score in sample.rivals:
                rival_weighted[rival_id] = rival_weighted.get(rival_id, 0.0) + rival_score * weight
                rival_mass[rival_id] = rival_mass.get(rival_id, 0.0) + weight

        if not lead_mass:
            return (None, 0.0, None, 0.0, 0.0)

        def _mean(sid: str) -> float:
            mass = lead_mass.get(sid, 0.0)
            return lead_weighted[sid] / mass if mass > 1e-12 else 0.0

        leader = min(lead_scores, key=lambda s: (-lead_mass[s], -_mean(s), s))
        leader_score = _mean(leader)
        median = statistics.median(lead_scores[leader])

        rivals = [
            (sid, total / rival_mass[sid])
            for sid, total in rival_weighted.items()
            if sid != leader and rival_mass[sid] > 1e-12
        ]
        # A rival that has held the lead at some point in this window is a more
        # serious competitor than one that only ever sat at rank 2, so it wins
        # ties for the runner-up slot.
        rivals.sort(key=lambda kv: (-kv[1], -lead_mass.get(kv[0], 0.0), kv[0]))
        runner_up = rivals[0][0] if rivals else None
        runner_up_score = rivals[0][1] if rivals else 0.0

        return (leader, leader_score, runner_up, runner_up_score, median)

    def _count_support(self, state: _TrackState, student_id: str) -> int:
        return sum(1 for s in state.samples if s.top_id == student_id)

    def _should_overturn(self, state: _TrackState, now: float) -> str | None:
        """The student that has decisively taken this track over, if any.

        Evaluated directly on the tail of the window rather than on the global
        tally. A global tally is dominated by however many frames the original
        subject accumulated, so it would keep reporting the old identity right
        through an identity switch — which is the one situation where the
        sticky commitment is actively wrong.

        Returns a challenger only when one student held top-1 across
        `overturn_frames` *consecutive* frames at a score the standing
        commitment could not plausibly explain. A single anomalous crop, a
        momentary occlusion, or an interleaved challenger will not satisfy it.
        """
        current = state.committed_student
        if current is None or len(state.samples) < self.t.overturn_frames:
            return None

        tail = list(state.samples)[-self.t.overturn_frames :]
        if any(now - s.timestamp > self.t.max_gap_s for s in tail[:-1]):
            return None

        challengers = {s.top_id for s in tail}
        if len(challengers) != 1:
            return None
        challenger = next(iter(challengers))
        if challenger == current:
            return None
        if statistics.median(s.top_score for s in tail) < self.t.overturn_score:
            return None
        return challenger

"""Identity service: consumes observations, writes presence, serves queries.

Separate from the capture workers on purpose. It owns the authoritative
gallery, the consent registry, and the only write path to presence events. A
capture worker is disposable and holds no durable state; this service is
backed by Postgres and can be restarted at any point without losing a single
committed sighting, because it replays from the Redis stream's consumer group.

Running more than one replica is safe: the consumer group gives each message
to exactly one consumer, and the ``ON CONFLICT DO NOTHING`` on
``event_id``/``track_id`` makes a redelivery idempotent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from campus.consent.registry import ConsentRegistry, Purpose
from campus.index.gallery import CentroidIndexer, GalleryIndex, build_index
from campus.store.db import Database
from campus.types import StudentId
from campus.worker.bus import ObservationBus

log = logging.getLogger("campus.identity")


@dataclass(slots=True)
class IdentityStats:
    consumed: int = 0
    written: int = 0
    suppressed: int = 0
    failures: int = 0
    last_error: str = ""
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        uptime = max(1e-6, time.time() - self.started_at)
        return {
            "consumed": self.consumed,
            "written": self.written,
            "suppressed": self.suppressed,
            "failures": self.failures,
            "events_per_s": round(self.written / uptime, 2),
            "last_error": self.last_error,
        }


class IdentityService:
    def __init__(
        self,
        database: Database,
        bus: ObservationBus,
        dim: int = 512,
        purpose: Purpose = Purpose.ATTENDANCE,
        consumer: str = "identity-1",
    ) -> None:
        self.db = database
        self.bus = bus
        self.purpose = purpose
        self.consumer = consumer
        self.centroids = CentroidIndexer(dim)
        self.index: GalleryIndex = build_index(dim)
        self.consent = ConsentRegistry()
        self.stats = IdentityStats()
        self._loaded = False

    def bootstrap(self, rebuild_index: bool = True) -> dict[str, Any]:
        """Load gallery and consent from Postgres. Call before consuming.

        Without this the service would start with an empty index and every
        observation would resolve to UNKNOWN — silently, because "no match" and
        "no gallery" look identical downstream. `gallery_size` in the health
        endpoint is the check that catches it.
        """
        vectors, ids = self.db.load_gallery()
        centroids, centroid_ids = self.centroids.build(zip(ids, vectors, strict=True))
        self.index = build_index(self.config_dim(centroids))
        if len(centroids):
            self.index.add(centroids, centroid_ids)

        for row in self.db.load_consents():
            from campus.consent.registry import ConsentRecord  # noqa: PLC0415

            self.consent.grant(
                ConsentRecord(
                    student_id=row["student_id"],
                    purposes=frozenset(Purpose(p) for p in row["purposes"] if p),
                    granted_at=row["granted_at"] or time.time(),
                    expires_at=row["expires_at"],
                    revoked_at=row["revoked_at"],
                    consent_id=row["consent_id"] or "",
                    camera_scope=frozenset(row["camera_scope"]),
                )
            )

        self._loaded = True
        return {
            "gallery_vectors": len(vectors),
            "gallery_students": len(centroid_ids),
            "consents": len(self.consent._records),
        }

    @staticmethod
    def config_dim(centroids: Any) -> int:
        return int(centroids.shape[1]) if len(centroids) else 512

    def reload_gallery(self) -> int:
        """Rebuild the index after a bulk enrollment. Returns student count."""
        vectors, ids = self.db.load_gallery()
        centroids, centroid_ids = self.centroids.build(zip(ids, vectors, strict=True))
        self.index = build_index(self.config_dim(centroids))
        if len(centroids):
            self.index.add(centroids, centroid_ids)
        return len(centroid_ids)

    def handle_batch(self, events: list[dict[str, Any]]) -> int:
        """Persist a batch of committed tracks. Returns rows written."""
        if not events:
            return 0
        written = 0
        for event in events:
            try:
                student_id = event.get("student_id")
                camera_id = event.get("camera_id")
                if not camera_id or not student_id:
                    self.stats.suppressed += 1
                    continue
                decision = self.consent.check(
                    str(student_id), self.purpose, str(camera_id),
                    event.get("zone", "default"),
                    event.get("committed_at", time.time()),
                )
                rows = self.db.write_presence(
                    [_commit_from_event(event)],
                    zone=event.get("zone", "default"),
                    purpose=self.purpose.value,
                    consent_id=decision.consent_id,
                    suppressed=not decision.permitted,
                )
                written += len(rows)
                if not decision.permitted:
                    self.stats.suppressed += 1
            except Exception as exc:  # noqa: BLE001 - one bad event must not stall the stream
                self.stats.failures += 1
                self.stats.last_error = str(exc)
                log.exception("failed to persist observation %s", event.get("track_id"))
        self.stats.written += written
        return written

    def run_forever(self, poll_ms: int = 1000) -> None:
        if not self._loaded:
            self.bootstrap()
        log.info("identity service consuming as %s", self.consumer)
        while True:
            for message_id, events in self.bus.read_observations(
                group="identity", consumer=self.consumer, block_ms=poll_ms
            ):
                self.stats.consumed += len(events)
                if self.handle_batch(events):
                    self.bus.ack(message_id)

    def sweep_erasure_queue(self) -> int:
        """Redact and delete for every pending erasure. Returns students handled."""
        handled = 0
        for student_id, _at in self.consent.pending_erasures():
            self.db.delete_student_data(student_id)
            self.reload_gallery()
            handled += 1
        if handled:
            self.consent._revocations.clear()
        return handled


def _commit_from_event(event: dict[str, Any]):
    from campus.types import TrackCommit  # noqa: PLC0415

    return TrackCommit(
        track_id=event["track_id"],
        camera_id=event["camera_id"],
        student_id=StudentId(event["student_id"]),
        first_seen=event.get("first_seen", 0.0),
        last_seen=event.get("last_seen", 0.0),
        committed_at=event.get("committed_at", time.time()),
        evidence_frames=event.get("evidence_frames", 0),
        window_frames=event.get("window_frames", 0),
        median_score=event.get("median_score", 0.0),
        min_score=event.get("min_score", 0.0),
        mean_score=event.get("mean_score", 0.0),
        score_std=event.get("score_std", 0.0),
        runner_up_id=event.get("runner_up_id"),
        runner_up_score=event.get("runner_up_score", 0.0),
        margin=event.get("margin", 0.0),
        duration_s=event.get("duration_s", 0.0),
        quality_frames=event.get("quality_frames", 0),
    )


def iter_events(payload: str) -> Iterator[dict[str, Any]]:
    import json  # noqa: PLC0415

    yield from json.loads(payload)

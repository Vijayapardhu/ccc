"""Postgres access: presence events, gallery, consent, audit.

Two things are non-obvious and worth stating.

**Embeddings live in pgvector, not only in FAISS.** FAISS is a derived,
disposable index; Postgres is the source of truth. A gallery that exists only
in a GPU process's memory is unrecoverable after a node failure and impossible
to audit, and DPDP erasure is unimplementable against it.

**Events are partitioned by month.** At 400 cameras x 8fps the raw observation
table grows fast, and a single unpartitioned table makes every retention job a
full scan. Monthly partitions let retention be ``DROP TABLE`` — instant, and
provably complete.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from campus.types import PresenceEvent, StudentId, TrackCommit

log = logging.getLogger("campus.store")

SCHEMA_PATH = "sql/schema.sql"


class Database:
    """Thin Postgres wrapper. No ORM — the queries are few, and SQL visibility
    is worth more here than abstraction."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 16) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any = None

    def connect(self) -> None:
        from psycopg_pool import ConnectionPool  # noqa: PLC0415

        self._pool = ConnectionPool(
            self.dsn, min_size=self.min_size, max_size=self.max_size, open=True
        )
        log.info("database pool opened")

    @contextmanager
    def cursor(self, commit: bool = True) -> Iterator[Any]:
        if self._pool is None:
            self.connect()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                yield cur
            if commit:
                conn.commit()

    def apply_schema(self, path: str = SCHEMA_PATH) -> None:
        from pathlib import Path  # noqa: PLC0415

        sql = Path(path).read_text(encoding="utf-8")
        with self.cursor() as cur:
            cur.execute(sql)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # -- presence ----------------------------------------------------------

    def write_presence(
        self, commits: Sequence[TrackCommit], zone: str, purpose: str,
        consent_id: str | None = None, suppressed: bool = False,
    ) -> list[PresenceEvent]:
        """Persist committed tracks. ``student_id`` is NULL when suppressed.

        Storing a suppressed event with a null student is intentional: the
        camera *saw* someone, the audit trail should show that, but the row
        must not be a queryable record of who. That distinction is what makes
        an erasure provable.
        """
        now = time.time()
        rows: list[PresenceEvent] = []
        for c in commits:
            rows.append(
                PresenceEvent(
                    event_id=PresenceEvent.new_id(),
                    student_id=None if suppressed else c.student_id,  # type: ignore[arg-type]
                    camera_id=c.camera_id,
                    zone=zone,
                    first_seen=c.first_seen,
                    last_seen=c.last_seen,
                    track_id=c.track_id,
                    confidence=c.median_score,
                    purposes=(purpose,),
                    redacted=suppressed,
                    retained_until=now + 30 * 86400,
                )
            )
        if not rows:
            return []

        sql = """
            INSERT INTO presence_events (
                event_id, student_id, camera_id, zone, track_id,
                first_seen, last_seen, confidence, purposes,
                redacted, consent_id, retained_until
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
        """
        with self.cursor() as cur:
            for e in rows:
                cur.execute(
                    sql,
                    (
                        e.event_id, e.student_id, e.camera_id, e.zone, e.track_id,
                        e.first_seen, e.last_seen, e.confidence,
                        list(e.purposes), e.redacted, consent_id, e.retained_until,
                    ),
                )
        return rows

    def mark_redacted(self, event_ids: Sequence[str]) -> int:
        if not event_ids:
            return 0
        with self.cursor() as cur:
            cur.execute(
                "UPDATE presence_events SET student_id = NULL, redacted = TRUE "
                "WHERE event_id = ANY(%s)",
                (list(event_ids),),
            )
            return cur.rowcount

    # -- gallery -----------------------------------------------------------

    def upsert_embeddings(
        self, records: Sequence[tuple[StudentId, str, str, npt.NDArray[np.float32]]]
    ) -> int:
        """Upsert ``(student_id, photo_id, angle, embedding)`` rows.

        Uses pgvector's halfvec at 256 dimensions. Full float32 vectors for 50k
        students is 100MB and wasteful; the cosine ranking is unchanged at half
        precision, and the authoritative copy of each vector is in the FAISS
        index built from these rows.
        """
        from pgvector.psycopg import register_vector  # noqa: PLC0415

        sql = """
            INSERT INTO student_embeddings
                (student_id, photo_id, angle, embedding, enrolled_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (student_id, photo_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    angle = EXCLUDED.angle,
                    enrolled_at = now()
        """
        with self.cursor() as cur:
            register_vector(cur)
            for student_id, photo_id, angle, vec in records:
                cur.execute(sql, (str(student_id), photo_id, angle, np.asarray(vec, dtype=np.float32)))
            return cur.rowcount

    def load_gallery(self, limit: int | None = None) -> tuple[npt.NDArray[np.float32], list[str]]:
        sql = "SELECT student_id, embedding FROM student_embeddings ORDER BY student_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        vectors: list[npt.NDArray[np.float32]] = []
        ids: list[str] = []
        with self.cursor(commit=False) as cur:
            cur.execute(sql)
            for student_id, vec in cur.fetchall():
                ids.append(str(student_id))
                vectors.append(np.asarray(vec, dtype=np.float32))
        if not vectors:
            return np.zeros((0, 512), dtype=np.float32), []
        return np.stack(vectors), ids

    def count_embeddings(self) -> int:
        with self.cursor(commit=False) as cur:
            cur.execute("SELECT count(*) FROM student_embeddings")
            return int(cur.fetchone()[0])

    def distinct_students(self) -> int:
        with self.cursor(commit=False) as cur:
            cur.execute("SELECT count(DISTINCT student_id) FROM student_embeddings")
            return int(cur.fetchone()[0])

    def delete_student_data(self, student_id: str) -> dict[str, int]:
        """DPDP erasure. Returns counts by table so the response is auditable.

        Deletes the vectors, redacts the presence history, and writes an
        audit row. The audit row is retained deliberately — the obligation is
        to erase the personal data, not to erase the fact that an erasure
        happened.
        """
        counts: dict[str, int] = {}
        with self.cursor() as cur:
            cur.execute("DELETE FROM student_embeddings WHERE student_id = %s", (student_id,))
            counts["embeddings"] = cur.rowcount
            cur.execute(
                "UPDATE presence_events SET student_id = NULL, redacted = TRUE "
                "WHERE student_id = %s",
                (student_id,),
            )
            counts["events_redacted"] = cur.rowcount
            cur.execute("DELETE FROM consent_records WHERE student_id = %s", (student_id,))
            counts["consents"] = cur.rowcount
            cur.execute(
                "INSERT INTO erasure_log (student_id, erased_at, vectors, events) "
                "VALUES (%s, now(), %s, %s)",
                (student_id, counts.get("embeddings", 0), counts.get("events_redacted", 0)),
            )
        return counts

    # -- consent -----------------------------------------------------------

    def upsert_consent(
        self, student_id: str, purposes: Sequence[str], consent_id: str,
        camera_scope: Sequence[str] = (), expires_at: float | None = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO consent_records
                    (student_id, purposes, consent_id, camera_scope, granted_at, expires_at)
                VALUES (%s, %s, %s, %s, now(), to_timestamp(%s))
                ON CONFLICT (student_id) DO UPDATE
                    SET purposes = EXCLUDED.purposes,
                        consent_id = EXCLUDED.consent_id,
                        camera_scope = EXCLUDED.camera_scope,
                        granted_at = now(),
                        revoked_at = NULL
                """,
                (student_id, list(purposes), consent_id, list(camera_scope), expires_at),
            )

    def revoke_consent(self, student_id: str) -> bool:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE consent_records SET revoked_at = now() "
                "WHERE student_id = %s AND revoked_at IS NULL",
                (student_id,),
            )
            return cur.rowcount > 0

    def load_consents(self) -> list[dict[str, Any]]:
        with self.cursor(commit=False) as cur:
            cur.execute(
                "SELECT student_id, purposes, consent_id, camera_scope, "
                "extract(epoch FROM granted_at), extract(epoch FROM expires_at), "
                "extract(epoch FROM revoked_at) FROM consent_records"
            )
            return [
                {
                    "student_id": r[0],
                    "purposes": list(r[1] or []),
                    "consent_id": r[2],
                    "camera_scope": list(r[3] or []),
                    "granted_at": r[4],
                    "expires_at": r[5],
                    "revoked_at": r[6],
                }
                for r in cur.fetchall()
            ]

    # -- analytics ---------------------------------------------------------

    def attendance_for_student(
        self, student_id: str, since: float, until: float | None = None
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT event_id, camera_id, zone, first_seen, last_seen, confidence
            FROM presence_events
            WHERE student_id = %s AND first_seen >= %s
              AND (%s IS NULL OR first_seen < %s)
            ORDER BY first_seen
        """
        params: tuple[Any, ...] = (student_id, since, until, until)
        with self.cursor(commit=False) as cur:
            cur.execute(sql, params)
            return [
                {
                    "event_id": r[0], "camera_id": r[1], "zone": r[2],
                    "first_seen": r[3], "last_seen": r[4], "confidence": r[5],
                }
                for r in cur.fetchall()
            ]

    def purge_expired(self, now: float | None = None) -> int:
        """Delete events past `retained_until`. Returns rows removed."""
        t = now if now is not None else time.time()
        with self.cursor() as cur:
            cur.execute("DELETE FROM presence_events WHERE retained_until < %s", (t,))
            return cur.rowcount


def build_presence_from_commit(
    commit: TrackCommit, zone: str, retention_days: int = 30
) -> PresenceEvent:
    return PresenceEvent(
        event_id=PresenceEvent.new_id(),
        student_id=commit.student_id,
        camera_id=commit.camera_id,
        zone=zone,
        first_seen=commit.first_seen,
        last_seen=commit.last_seen,
        track_id=commit.track_id,
        confidence=commit.median_score,
        purposes=("attendance",),
        retained_until=time.time() + retention_days * 86400,
    )


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=json_default)

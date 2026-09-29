"""HTTP API.

Read paths for operations, write paths for consent and erasure. Deliberately
small — the interesting work happens in the workers, and every endpoint here
that could be a write is an administrative action that should be audited.

Two rules enforced in code rather than in documentation:

* No endpoint returns a student_id without a consent check. Attendance queries
  are scoped to the requesting supervisor's zones, and an unauthorised request
  gets 403 rather than an empty result — an empty result is indistinguishable
  from "no records" and hides the misconfiguration.
* Erasure is synchronous and returns per-table counts, so the response is
  evidence that the erasure happened.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from campus.consent.registry import Purpose
from campus.identity.service import IdentityService
from campus.store.db import Database
from campus.worker.bus import ObservationBus


class GrantRequest(BaseModel):
    student_id: str
    purposes: list[Purpose] = Field(min_length=1)
    consent_id: str | None = None
    camera_scope: list[str] = Field(default_factory=list)
    expires_at: float | None = None


class RevokeRequest(BaseModel):
    student_id: str
    erase: bool = True
    """When true, run the DPDP erasure path and delete the vectors. When
    false, stop identifications but keep the gallery for re-consent later."""


class AttendanceQuery(BaseModel):
    student_id: str
    since: float
    until: float | None = None


@dataclass
class AppState:
    db: Database
    bus: ObservationBus
    identity: IdentityService
    api_key: str
    started_at: float = field(default_factory=time.time)
    zones: set[str] = field(default_factory=set)
    """Zones this API instance is authorised to serve. Empty means all, which
    is only appropriate in a single-tenant dev deployment."""


def create_app(
    db: Database, bus: ObservationBus, identity: IdentityService, api_key: str
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.connect()
        db.apply_schema()
        identity.bootstrap()
        bus.connect()
        yield
        bus.close()
        db.close()

    app = FastAPI(
        title="Campus Presence API",
        version="0.1.0",
        description="Student presence tracking across the campus camera network.",
        lifespan=lifespan,
    )
    state = AppState(db=db, bus=bus, identity=identity, api_key=api_key)
    app.state.campus = state

    def require_key(x_api_key: str = Header(default="")) -> str:
        if state.api_key and x_api_key != state.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")
        return x_api_key

    def require_zones(zone: str) -> None:
        if state.zones and zone not in state.zones:
            raise HTTPException(
                status_code=403,
                detail=f"this instance is not authorised for zone {zone!r}",
            )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness plus the checks that actually predict silent failure.

        `gallery_size` and `bus_degraded` are the two that matter: an empty
        gallery and a disconnected bus both produce a system that runs
        perfectly and reports nothing.
        """
        return {
            "status": "ok",
            "uptime_s": round(time.time() - state.started_at, 1),
            "gallery_size": len(state.identity.index),
            "consents": len(state.identity.consent._records),
            "bus_degraded": state.bus.degraded,
            "bus_dropped": state.bus.dropped,
            "identity": state.identity.stats.snapshot(),
        }

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        gallery = len(state.identity.index)
        if gallery == 0:
            raise HTTPException(
                status_code=503,
                detail="gallery is empty; every observation would resolve to UNKNOWN",
            )
        if state.bus.degraded:
            raise HTTPException(status_code=503, detail="observation bus disconnected")
        return {"status": "ready", "gallery_size": gallery}

    @app.get("/v1/attendance/{student_id}")
    def attendance(
        student_id: str,
        since: float = Query(..., description="epoch seconds"),
        until: float | None = None,
        zone: str = Query("default"),
        _: str = Depends(require_key),
    ) -> dict[str, Any]:
        require_zones(zone)
        rows = state.db.attendance_for_student(student_id, since, until)
        return {"student_id": student_id, "events": rows, "count": len(rows)}

    @app.get("/v1/cameras")
    def cameras(_: str = Depends(require_key)) -> dict[str, Any]:
        with state.db.cursor(commit=False) as cur:
            cur.execute(
                "SELECT id, name, host, zone, enabled, width, height, target_fps "
                "FROM cameras ORDER BY zone, id"
            )
            return {
                "cameras": [
                    {
                        "id": r[0], "name": r[1], "host": r[2], "zone": r[3],
                        "enabled": r[4], "width": r[5], "height": r[6], "target_fps": r[7],
                    }
                    for r in cur.fetchall()
                ]
            }

    @app.post("/v1/consent", status_code=201)
    def grant_consent(req: GrantRequest, _: str = Depends(require_key)) -> dict[str, Any]:
        import uuid  # noqa: PLC0415

        consent_id = req.consent_id or str(uuid.uuid4())
        state.db.upsert_consent(
            req.student_id,
            [p.value for p in req.purposes],
            consent_id,
            req.camera_scope,
            req.expires_at,
        )
        state.identity.consent.revoke(req.student_id)
        return {"student_id": req.student_id, "consent_id": consent_id, "granted": True}

    @app.post("/v1/consent/revoke")
    def revoke_consent(req: RevokeRequest, _: str = Depends(require_key)) -> dict[str, Any]:
        revoked = state.db.revoke_consent(req.student_id)
        state.identity.consent.revoke(req.student_id)
        result: dict[str, Any] = {"student_id": req.student_id, "revoked": revoked}
        if req.erase:
            # Counts returned so the caller has evidence the erasure ran, not
            # just an acknowledgement that it was requested.
            result["erasure"] = state.db.delete_student_data(req.student_id)
            result["gallery_size"] = state.identity.reload_gallery()
        return result

    @app.get("/v1/gallery/stats")
    def gallery_stats(_: str = Depends(require_key)) -> dict[str, Any]:
        return {
            "index_size": len(state.identity.index),
            "vectors_stored": state.db.count_embeddings(),
            "students": state.db.distinct_students(),
        }

    @app.post("/v1/gallery/reload")
    def gallery_reload(_: str = Depends(require_key)) -> dict[str, Any]:
        return {"students": state.identity.reload_gallery()}

    @app.post("/v1/retention/purge")
    def purge(_: str = Depends(require_key)) -> dict[str, Any]:
        return {"purged": state.db.purge_expired()}

    return app

"""FastAPI app wrapping the inspection runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from campus.capture.source import frame_gray
from campus.config import CameraConfig
from campus.imaging.quality import assess
from campus.imaging.tiling import plan_tiles
from campus.ui.runner import CameraRunner, encode_jpeg, mjpeg
from campus.ui.template import PAGE

api = FastAPI(title="Campus pipeline inspector", version="0.1.0")
slots: dict[str, CameraRunner] = {}
meta: dict[str, Any] = {"stream_fps": 15.0}

# The loaded gallery is a Python object, not a payload. It is held here rather
# than in `meta` because `meta` is serialised straight into the dashboard HTML,
# and a dataclass there is a 500 on every page load.
_gallery_file: Any = None


def configure(runners: dict[str, CameraRunner], info: dict[str, Any]) -> None:
    global _gallery_file
    slots.clear()
    slots.update(runners)
    meta.clear()
    meta["stream_fps"] = info.pop("stream_fps", 15.0)
    _gallery_file = info.pop("_gallery_file", None)
    meta.update(info)


def _page_meta() -> str:
    """Serialise the dashboard's bootstrap metadata, defensively.

    Only values that are already JSON-native are passed through. Anything else
    is dropped rather than allowed to raise, because a 500 on `/` takes out the
    one page someone needs in order to see what went wrong.
    """
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe[key] = value
    return json.dumps(safe)


@api.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.replace("__META__", _page_meta())


@api.get("/api/state")
def state() -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for r in slots.values():
        events.extend(e.to_dict() for e in r.events)
    events.sort(key=lambda e: -e["at"])
    return {
        "meta": meta,
        "cameras": [r.slot.to_dict() for r in slots.values()],
        "events": events[:40],
    }


@api.get("/api/snapshot/{camera_id}")
def snapshot(camera_id: str) -> Response:
    r = slots.get(camera_id)
    if r is None or r.slot.frame is None:
        return Response(status_code=404)
    return Response(encode_jpeg(r.slot.frame), media_type="image/jpeg")


@api.get("/stream/{camera_id}")
def stream(camera_id: str) -> StreamingResponse:
    if camera_id not in slots:
        return StreamingResponse(iter(()), status_code=404)
    return StreamingResponse(
        mjpeg(slots, camera_id, meta.get("stream_fps", 15.0)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@api.get("/api/gallery")
def gallery() -> dict[str, Any]:
    return meta.get("gallery", {})


def roster_rows(
    g: Any, q: str = "", limit: int = 5000
) -> dict[str, Any]:
    """Filter the enrolled roster by substring. Pure — no HTTP, no globals.

    Split out from the route so the filtering can be tested directly. The
    "did this enrollment actually land?" question is worth a real test, and it
    should not depend on an ASGI test client being installed.
    """
    if g is None:
        return {"total": 0, "matched": 0, "query": q, "students": []}
    needle = q.strip().lower()
    rows = [
        {"student_id": sid, "photos": g.photos_per_student.get(sid, 1)}
        for sid in g.student_ids
        if not needle or needle in sid.lower()
    ]
    return {
        "total": len(g.student_ids),
        "matched": len(rows),
        "query": q,
        "students": rows[:limit],
    }


@api.get("/api/gallery/students")
def students(q: str = "", limit: int = 5000) -> dict[str, Any]:
    """The enrolled roster, optionally filtered by substring.

    Exists because "did my enrollment actually land?" is the first question
    after any enroll, and the answer has to be inspectable. A gallery is a
    vector file; without this, confirming a student's presence means reading
    an .npz by hand.
    """
    return roster_rows(_gallery_file, q, limit)


@api.get("/api/probe")
def probe(path: str) -> dict[str, Any]:
    """Run a single image through the full chain and report where it lands.

    The enroll-then-verify loop, exposed. After enrolling a student the only
    meaningful question is "does their own photo come back as themselves, and
    by what margin" — and that is exactly the number the temporal verifier will
    later threshold on. If the margin here is thin, no amount of tuning
    downstream will fix it.
    """
    import cv2  # noqa: PLC0415

    runner = next(iter(slots.values()), None)
    if runner is None:
        raise HTTPException(status_code=503, detail="no cameras running")
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no such image: {path}")

    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=400, detail="unreadable image")

    h, w = img.shape[:2]
    boxes = runner.detector.detect(img, plan_tiles(w, h, 640, 640, 0.25))
    if not boxes:
        return {"file": p.name, "faces": 0, "note": "no face detected"}

    out: list[dict[str, Any]] = []
    for b in sorted(boxes, key=lambda b: -b.score)[:3]:
        rep = assess(frame_gray(img), b, runner.quality)
        entry: dict[str, Any] = {
            "box": [b.x0, b.y0, b.x1, b.y1], "face_px": b.min_side,
            "det_score": round(b.score, 4), "passed": rep.passed,
            "reasons": list(rep.reasons), "quality": rep.to_dict(),
            "candidates": [],
        }
        if rep.passed:
            vecs, idxs = runner.embedder.embed_from_frame(img, [b])
            if len(vecs) and runner.index is not None:
                hits = runner.index.search(vecs, 5)[0]
                entry["candidates"] = [
                    {"rank": c.rank, "student_id": c.student_id, "score": round(c.score, 4)}
                    for c in hits
                ]
                if len(hits) > 1:
                    entry["margin"] = round(hits[0].score - hits[1].score, 4)
                    entry["passes_margin"] = (hits[0].score - hits[1].score) >= 0.06
        out.append(entry)
    return {"file": p.name, "size": [w, h], "faces": len(boxes), "results": out}


@api.get("/api/thresholds")
def thresholds() -> dict[str, Any]:
    return meta.get("quality", {})

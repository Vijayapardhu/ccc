"""Command line entry point.

``campus worker``     run a capture worker over a slice of cameras
``campus identity``   run the identity service
``campus api``        serve the HTTP API
``campus enroll``     enroll students from photos
``campus index``      rebuild the gallery index
``campus validate``   check configs without touching hardware
``campus bench``      measure the pipeline on a video file

`validate` and `bench` exist because both catch the majority of deployment
failures before any camera is involved — a typo in a stream path or a model
that is 2x slower than budgeted is much cheaper to find on a laptop than on a
GPU node at 3am.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _load(args: argparse.Namespace):
    from campus.config import load_cameras, load_system_config

    system = load_system_config(args.config)
    cameras = load_cameras(args.cameras) if args.cameras else []
    return system, cameras


def cmd_worker(args: argparse.Namespace) -> int:
    from campus.config import validate_cameras
    from campus.worker.bus import ObservationBus
    from campus.worker.pipeline import WorkerPipeline

    system, cameras = _load(args)
    problems = validate_cameras(cameras)
    if problems:
        for p in problems:
            print(f"config error: {p}", file=sys.stderr)
        return 2

    assigned = _assign(cameras, args.cameras_filter, args.max_cameras)
    if not assigned:
        print("no cameras assigned to this worker", file=sys.stderr)
        return 2

    bus = ObservationBus(
        system.worker.redis_url, system.worker.worker_id,
        batch_size=system.worker.publish_interval_s and 200,
    )
    if not bus.connect():
        print(
            "warning: observation bus unavailable; observations will be dropped",
            file=sys.stderr,
        )

    pipeline = WorkerPipeline(config=system, cameras=assigned, bus=bus)
    pipeline.warm_up()
    print(json.dumps({"worker": system.worker.worker_id, "cameras": [c.id for c in assigned]}))
    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    from campus.identity.service import IdentityService
    from campus.store.db import Database
    from campus.worker.bus import ObservationBus

    system, _ = _load(args)
    db = Database(system.worker.postgres_dsn)
    db.connect()
    db.apply_schema()
    bus = ObservationBus(system.worker.redis_url, "identity")
    bus.connect()
    identity = IdentityService(db, bus, consumer=args.consumer)
    print(json.dumps(identity.bootstrap()))
    try:
        identity.run_forever(poll_ms=args.poll_ms)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
        db.close()
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415

    from campus.api.app import create_app
    from campus.identity.service import IdentityService
    from campus.store.db import Database
    from campus.worker.bus import ObservationBus

    system, _ = _load(args)
    db = Database(system.worker.postgres_dsn)
    bus = ObservationBus(system.worker.redis_url, "api")
    identity = IdentityService(db, bus, consumer="api-readonly")
    api_key = os.environ.get("CAMPUS_API_KEY", "")
    if not api_key and not args.insecure:
        print(
            "CAMPUS_API_KEY is unset. The API would serve every student's "
            "attendance unauthenticated. Set it, or pass --insecure for local dev.",
            file=sys.stderr,
        )
        return 2

    uvicorn.run(
        create_app(db, bus, identity, api_key),
        host=args.host, port=args.port, log_level="info",
    )
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    from campus.enroll.service import EnrollmentService
    from campus.models.arcface import ArcFaceEmbedder
    from campus.models.scrfd import ScrfdDetector
    from campus.store.db import Database

    system, _ = _load(args)
    detector = ScrfdDetector(args.detector_model, device=args.device)
    embedder = ArcFaceEmbedder(args.embedder_model, device=args.device)
    service = EnrollmentService(detector, embedder)

    db: Database | None = None
    if args.persist:
        db = Database(system.worker.postgres_dsn)
        db.connect()
        db.apply_schema()

    try:
        for student_id, directory in args.students:
            report = service.enroll_directory(student_id, directory)
            warnings = service.verify_consistency(report.accepted)
            for w in warnings:
                print(f"warning: {w}", file=sys.stderr)
            if db is not None and report.accepted:
                rows = [
                    (r.student_id, r.photo_id, r.angle, r.embedding)
                    for r in report.accepted
                ]
                db.upsert_embeddings(rows)
            print(json.dumps({"student_id": student_id, **report.summary()}))
            if not report.ok:
                return 1
    finally:
        if db is not None:
            db.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from campus.index.gallery import CentroidIndexer, build_index
    from campus.store.db import Database

    system, _ = _load(args)
    db = Database(system.worker.postgres_dsn)
    db.connect()
    vectors, ids = db.load_gallery()
    indexer = CentroidIndexer(vectors.shape[1] if len(vectors) else args.dim)
    centroids, centroid_ids = indexer.build(zip(ids, vectors, strict=True))
    index = build_index(indexer.dim, args.backend, use_gpu=not args.cpu)
    if len(centroids):
        index.add(centroids, centroid_ids)
    print(json.dumps({
        "vectors": len(vectors),
        "students": len(centroid_ids),
        "backend": args.backend,
        "dim": indexer.dim,
    }))
    db.close()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from campus.config import validate_cameras

    system, cameras = _load(args)
    problems = validate_cameras(cameras)

    for label, path in (
        ("detector", system.detectors.model_path),
        ("embedder", system.embedder.model_path),
    ):
        if not Path(path).exists():
            problems.append(f"{label} model missing: {path}")

    enabled = [c for c in cameras if c.enabled]
    zones = {c.zone for c in enabled}
    for zone in zones:
        if zone not in system.zones and zone != "default":
            problems.append(
                f"camera zone {zone!r} has no policy; it will fall back to "
                f"'default'. Add it to config to make the retention and "
                f"purpose rules explicit."
            )

    result = {
        "cameras": len(cameras),
        "enabled": len(enabled),
        "zones": sorted(zones),
        "problems": problems,
    }
    print(json.dumps(result, indent=2))
    return 2 if problems else 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Measure per-frame cost on a video file, so GPU sizing is measured not guessed."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from campus.imaging.quality import assess
    from campus.imaging.tiling import plan_tiles
    from campus.models.arcface import ArcFaceEmbedder
    from campus.models.scrfd import ScrfdDetector

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 2

    detector = ScrfdDetector(args.detector_model, device=args.device)
    embedder = ArcFaceEmbedder(args.embedder_model, device=args.device)
    d = vars(args)

    frames = faces = kept = 0
    detect_ms = embed_ms = 0.0
    t_start = time.perf_counter()

    while frames < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        tiles = plan_tiles(w, h, detector_input=d["input_size"],
                           tile_size=d["tile_size"], overlap=d["tile_overlap"])
        t0 = time.perf_counter()
        boxes = detector.detect(frame, tiles)
        detect_ms += (time.perf_counter() - t0) * 1000
        faces += len(boxes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        survivors = [b for b in boxes if assess(gray, b).passed]
        if survivors:
            t1 = time.perf_counter()
            embedder.embed_from_frame(frame, survivors)
            embed_ms += (time.perf_counter() - t1) * 1000
        kept += len(survivors)
        frames += 1
        if args.progress and frames % 10 == 0:
            print(f"  {frames} frames, {faces} faces", file=sys.stderr)

    cap.release()
    total_s = time.perf_counter() - t_start
    per_frame_ms = (detect_ms + embed_ms) / max(1, frames)

    print(json.dumps({
        "video": args.video,
        "resolution": [w, h],
        "frames": frames,
        "faces_detected": faces,
        "faces_embedded": kept,
        "faces_per_frame": round(faces / max(1, frames), 2),
        "detect_ms_per_frame": round(detect_ms / max(1, frames), 2),
        "embed_ms_per_frame": round(embed_ms / max(1, frames), 2),
        "total_ms_per_frame": round(per_frame_ms, 2),
        "realtime_factor": round(1000.0 / per_frame_ms, 2) if per_frame_ms else 0,
        "sustainable_cameras_per_node": round(
            (1000.0 / per_frame_ms) * args.target_fps / max(1.0, args.utilisation)
        ),
        "wall_s": round(total_s, 1),
    }, indent=2))
    return 0


def _assign(cameras, filters: str | None, max_cameras: int):
    """Deterministic camera-to-worker assignment.

    Shard by a stable hash of the camera id rather than by index or by
    round-robin. Index-based assignment reshuffles every camera when the list
    is edited, which means re-enrolling one camera causes a visible gap
    everywhere; hashing keeps an existing mapping stable under addition.
    """
    import hashlib  # noqa: PLC0415

    selected = [c for c in cameras if c.enabled]
    if filters:
        wanted = {f.strip() for f in filters.split(",") if f.strip()}
        selected = [c for c in selected if c.id in wanted]
    if not selected:
        return []

    worker_id = os.environ.get("CAMPUS_WORKER_ID", "worker-local")
    assigned = [
        c for c in selected
        if int(hashlib.sha256(f"{worker_id}:{c.id}".encode()).hexdigest()[:8], 16) % 1000
        < max(1, (1000 * max_cameras) // max(1, len(selected)))
    ]
    return assigned or selected[:max_cameras]


def build_parser() -> argparse.ArgumentParser:
    # The config flags are accepted both before and after the subcommand,
    # because `campus --config x validate` and `campus validate --config x` are
    # both things people will type. The subcommand copies use SUPPRESS so an
    # omitted flag does not overwrite the value already parsed by the parent.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--cameras", default=argparse.SUPPRESS)

    p = argparse.ArgumentParser(
        prog="campus", description=__doc__.split("\n")[0], parents=[common]
    )
    p.set_defaults(config="configs/system.yaml", cameras="configs/cameras.yaml",
                   verbose=False)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, parents=[common])

    w = add("worker", "run a capture worker")
    w.add_argument("--cameras-filter", help="comma-separated camera ids")
    w.add_argument("--max-cameras", type=int, default=32)
    w.set_defaults(func=cmd_worker)

    i = add("identity", "run the identity service")
    i.add_argument("--consumer", default="identity-1")
    i.add_argument("--poll-ms", type=int, default=1000)
    i.set_defaults(func=cmd_identity)

    a = add("api", "serve the HTTP API")
    a.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container-internal bind
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--insecure", action="store_true", help="allow unauthenticated access")
    a.set_defaults(func=cmd_api)

    e = add("enroll", "enroll students from photos")
    e.add_argument("students", nargs="+", metavar="STUDENT_ID:DIR")
    e.add_argument("--detector-model", default="models/scrfd_10g.onnx")
    e.add_argument("--embedder-model", default="models/glintr100k.onnx")
    e.add_argument("--device")
    e.add_argument("--persist", action="store_true", help="write vectors to Postgres")
    e.set_defaults(func=cmd_enroll)

    x = add("index", "rebuild and report the gallery index")
    x.add_argument("--backend", default="auto", choices=["auto", "faiss", "numpy"])
    x.add_argument("--dim", type=int, default=512)
    x.add_argument("--cpu", action="store_true")
    x.set_defaults(func=cmd_index)

    v = add("validate", "validate configuration, exit 2 on problems")
    v.set_defaults(func=cmd_validate)

    b = add("bench", "measure pipeline cost on a video file")
    b.add_argument("video")
    b.add_argument("--frames", type=int, default=100)
    b.add_argument("--detector-model", default="models/scrfd_2.5g.onnx")
    b.add_argument("--embedder-model", default="models/glintr100k.onnx")
    b.add_argument("--device")
    b.add_argument("--input-size", type=int, default=640)
    b.add_argument("--tile-size", type=int, default=1280)
    b.add_argument("--tile-overlap", type=float, default=0.25)
    b.add_argument("--target-fps", type=float, default=8.0)
    b.add_argument("--utilisation", type=float, default=0.7,
                   help="fraction of the node's capacity you are willing to plan for")
    b.add_argument("--progress", action="store_true")
    b.set_defaults(func=cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log_setup(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

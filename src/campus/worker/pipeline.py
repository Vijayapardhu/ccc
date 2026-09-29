"""The capture worker: RTSP frames in, identity observations out.

Pipeline per frame
------------------
    frame
      -> tile plan (cached per resolution)
      -> SCRFD over all tiles, one batched forward pass
      -> NMS merge into frame coordinates
      -> quality gate (cheap, rejects most)
      -> ArcFace on survivors, batched
      -> ByteTrack, to attach the embedding to a person
      -> gallery search (top-k)
      -> temporal verification (per track)
      -> commit event when a track earns it

The ordering is load-bearing. Quality gating before embedding is a ~5x saving
on the expensive step, tracking before search means fewer searches than
detections, and verification before emitting means no per-frame identity
statement ever escapes this process.

One worker owns a fixed slice of cameras and holds no gallery state beyond a
local index mirror. Restarting a worker loses in-flight tracks — acceptable,
because a track is a few seconds of evidence — and never loses committed
events, which are already durable in Postgres.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from campus.capture.source import (
    CameraWorkerStats,
    DecodeBackend,
    Frame,
    FrameSource,
    StreamConfig,
    frame_gray,
)
from campus.config import CameraConfig, SystemConfig
from campus.consent.registry import ConsentRegistry, Purpose
from campus.imaging.quality import QualityThresholds, assess
from campus.imaging.tiling import plan_tiles
from campus.index.gallery import GalleryIndex, build_index
from campus.models.arcface import ArcFaceEmbedder
from campus.models.scrfd import ScrfdDetector
from campus.track.bytetrack import ByteTracker
from campus.types import (
    CameraId,
    FaceBox,
    ObservationOutcome,
    TrackCommit,
)
from campus.worker.bus import ObservationBus

log = logging.getLogger("campus.worker")


@dataclass(slots=True)
class WorkerPipeline:
    """One worker process: a fixed set of cameras, shared models, per-camera state.

    Models are loaded once and shared across cameras. SCRFD and ArcFace weights
    are ~300MB and ~170MB respectively; loading per camera would cost 32x the
    memory and 32x the start-up time for no benefit.
    """

    config: SystemConfig
    cameras: list[CameraConfig]
    purpose: Purpose = Purpose.ATTENDANCE
    bus: ObservationBus | None = None
    consent: ConsentRegistry = field(default_factory=ConsentRegistry)
    stats: CameraWorkerStats = field(default_factory=CameraWorkerStats)

    _detector: ScrfdDetector | None = None
    _embedder: ArcFaceEmbedder | None = None
    _index: GalleryIndex | None = None
    _trackers: dict[str, ByteTracker] = field(default_factory=dict)
    _verifiers: dict[str, Any] = field(default_factory=dict)
    _tile_cache: dict[tuple[int, int], list[Any]] = field(default_factory=dict)
    _thresholds: dict[str, QualityThresholds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stats.cameras = len(self.cameras)
        for cam in self.cameras:
            self._trackers[cam.id] = ByteTracker(CameraId(cam.id))
            from campus.temporal.verifier import TemporalVerifier  # noqa: PLC0415

            self._verifiers[cam.id] = TemporalVerifier(self.config.verification)
            self._thresholds[cam.id] = self.config.quality_for(cam)
        for zone in self.config.zones.values():
            self.consent.set_zone_policy(zone)

    # -- lazy model loading ------------------------------------------------

    @property
    def detector(self) -> ScrfdDetector:
        if self._detector is None:
            d = self.config.detectors
            self._detector = ScrfdDetector(d.model_path, device=d.device)
            log.info("detector loaded: %s", d.model_path)
        return self._detector

    @property
    def embedder(self) -> ArcFaceEmbedder:
        if self._embedder is None:
            e = self.config.embedder
            self._embedder = ArcFaceEmbedder(
                e.model_path, device=e.device, config=None
            )
            log.info("embedder loaded: %s (dim=%d)", e.model_path, self._embedder.dim)
        return self._embedder

    @property
    def index(self) -> GalleryIndex:
        if self._index is None:
            g = self.config.gallery
            self._index = build_index(self.config.embedder.dim, g.backend, g.use_gpu)
            log.info("gallery index ready: %d entries", len(self._index))
        return self._index

    def warm_up(self) -> None:
        """Load models and run one dummy inference.

        The first ONNX session initialises CUDA context and allocates
        workspace, which takes 2-5s. Doing it at deploy time rather than on
        the first real frame keeps a camera from being declared dead during
        start-up.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.detect(dummy)
        self.embedder.embed([np.zeros((112, 112, 3), dtype=np.uint8)])

    # -- tiles -------------------------------------------------------------

    def _tiles(self, cam: CameraConfig, width: int, height: int) -> list[Any]:
        key = (width, height)
        if key not in self._tile_cache:
            d = self.config.detectors
            self._tile_cache[key] = plan_tiles(
                width, height,
                detector_input=d.input_size,
                tile_size=d.tile_size,
                overlap=d.tile_overlap,
            )
        return self._tile_cache[key]

    # -- main loop ---------------------------------------------------------

    def process_frame(self, frame: Frame, cam: CameraConfig) -> list[TrackCommit]:
        """Run one frame end to end. Returns any commits produced this frame."""
        t0 = time.perf_counter()
        self.stats.frames_processed += 1

        tiles = self._tiles(cam, frame.width, frame.height)
        boxes = self.detector.detect(frame.image, tiles)
        boxes = [b for b in boxes if b.x1 > b.x0 and b.y1 > b.y0]
        self.stats.faces_detected += len(boxes)
        if not boxes:
            return []

        gray = frame_gray(frame.image)
        thresholds = self._thresholds[cam.id]
        kept: list[FaceBox] = []
        for box in boxes:
            report = assess(gray, box, thresholds)
            if report.passed:
                kept.append(box)
            else:
                self.stats.faces_rejected_quality += 1
        if not kept:
            return []

        vectors, kept_indices = self.embedder.embed_from_frame(frame.image, kept)
        if len(kept_indices) == 0:
            return []
        self.stats.faces_embedded += len(kept_indices)
        embedded_boxes = [kept[i] for i in kept_indices]

        tracker = self._trackers[cam.id]
        tracks = tracker.update(embedded_boxes, frame.seq, frame.timestamp)

        candidates = self.index.search(vectors, self.config.gallery.top_k)

        verifier = self._verifiers[cam.id]
        for track, cands in zip(tracks, candidates, strict=False):
            if not cands:
                continue
            verifier.observe(
                track.track_id, CameraId(cam.id), cands,
                frame.timestamp, quality_passed=True,
            )

        commits: list[TrackCommit] = []
        for expired in tracker.expired():
            commit = verifier.finalize(expired.track_id)
            if commit is not None:
                commits.append(commit)

        self.stats.inference_ms += (time.perf_counter() - t0) * 1000
        if commits:
            self.stats.identities_committed += len(commits)
            self._publish(cam, commits)
        return commits

    def _publish(self, cam: CameraConfig, commits: list[TrackCommit]) -> None:
        if self.bus is None:
            return
        for commit in commits:
            zone = cam.zone
            decision = self.consent.check(
                commit.student_id, self.purpose, cam.id, zone, commit.committed_at
            )
            event: dict[str, Any] = {
                **commit.to_dict(),
                "zone": zone,
                "worker_id": self.config.worker.worker_id,
                "purpose": self.purpose.value,
                "consent_id": decision.consent_id,
                "suppressed": not decision.permitted,
            }
            if not decision.permitted:
                event["reason"] = decision.reason
            self.bus.publish_observation(event)

    def run_forever(self) -> None:
        """Own every camera until interrupted.

        Cameras are advanced round-robin rather than each on its own thread.
        At 400 cameras, 400 OS threads each holding a decoder buffer and
        model input tensors is a memory and scheduler problem; a single thread
        iterating a frame-ready queue per camera is not. The trade is that one
        slow camera delays the others, which the per-camera health surface
        makes visible.
        """
        log.info("worker %s starting on %d cameras", self.config.worker.worker_id, len(self.cameras))
        sources = {cam.id: self._source(cam) for cam in self.cameras}
        for src in sources.values():
            src.start()

        try:
            self._pump(sources)
        finally:
            for src in sources.values():
                src.stop()
            log.info("worker %s stopped", self.config.worker.worker_id)

    def _source(self, cam: CameraConfig) -> FrameSource:
        cfg: StreamConfig = cam.to_stream_config()
        cfg.backend = cam.backend or self.config.worker.stream_backend
        return FrameSource(cfg)

    def _pump(self, sources: dict[str, FrameSource]) -> None:
        import threading

        ready: dict[str, Iterator[Frame]] = {}
        lock = threading.Lock()
        done: set[str] = set()

        def reader(cam_id: str, src: FrameSource) -> None:
            try:
                for frame in src.frames():
                    with lock:
                        ready[cam_id] = iter((frame,))
                    time.sleep(0)
            except Exception:  # noqa: BLE001 - a reader must never kill the worker
                log.exception("reader for %s died", cam_id)
            finally:
                with lock:
                    done.add(cam_id)

        threads = [
            threading.Thread(target=reader, args=(cid, src), daemon=True, name=f"rd-{cid}")
            for cid, src in sources.items()
        ]
        for t in threads:
            t.start()

        by_id = {c.id: c for c in self.cameras}
        try:
            while len(done) < len(sources):
                for cam_id, pending in list(ready.items()):
                    try:
                        frame = next(pending)
                    except StopIteration:
                        ready.pop(cam_id, None)
                        continue
                    except Exception:  # noqa: BLE001
                        log.exception("frame from %s failed", cam_id)
                        ready.pop(cam_id, None)
                        continue
                    try:
                        self.process_frame(frame, by_id[cam_id])
                    except Exception:  # noqa: BLE001
                        log.exception("pipeline error on %s frame %d", cam_id, frame.seq)
                time.sleep(0.005)
        finally:
            for t in threads:
                t.join(timeout=2.0)

    def health(self) -> list[dict[str, Any]]:
        out = []
        for cam in self.cameras:
            tracker = self._trackers[cam.id]
            verifier = self._verifiers[cam.id]
            out.append(
                {
                    "camera_id": cam.id,
                    "zone": cam.zone,
                    "active_tracks": len(tracker.active()),
                    "verifier_tracks": verifier.track_count,
                }
            )
        return out


def commits_to_json(commits: list[TrackCommit]) -> str:
    return json.dumps([c.to_dict() for c in commits])


def outcome_counts(decisions: list[tuple[ObservationOutcome, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome, _ in decisions:
        counts[outcome.value] = counts.get(outcome.value, 0) + 1
    return counts

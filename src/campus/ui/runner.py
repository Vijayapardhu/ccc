"""Live pipeline inspection UI.

Deliberately a *diagnostic* surface, not a demo. When the system is wrong the
question is always "which stage", and a UI that only shows a video feed makes
you re-derive that from logs. So every face carries its full evidence chain:
which stage dropped it, what the quality metrics were, what the gallery
returned, and what the temporal verifier currently believes.

    /                     the dashboard
    /api/state            everything, as JSON
    /stream/{camera}      MJPEG with overlays burned in
    /api/snapshot/{cam}   one annotated JPEG

Overlays are drawn server-side so the browser shows exactly what the pipeline
saw. A client-side overlay is a second implementation of the projection maths,
and a second implementation is a second set of bugs.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from campus.capture.source import (
    DecodeBackend,
    Frame,
    FrameSource,
    StreamConfig,
    StreamState,
    frame_gray,
)
from campus.config import CameraConfig
from campus.consent.registry import Purpose
from campus.gallery import GalleryFile, to_index
from campus.imaging.quality import assess
from campus.imaging.tiling import plan_tiles
from campus.index.gallery import CentroidIndexer
from campus.models.arcface import ArcFaceEmbedder
from campus.models.scrfd import ScrfdDetector
from campus.temporal.verifier import TemporalVerifier, VerificationThresholds
from campus.track.bytetrack import ByteTracker
from campus.types import CameraId, FaceBox, GalleryCandidate, TrackId

# Overlay palette. Green = identity resolved, amber = tracked but not yet
# committed, grey = detected but rejected by the quality gate. Red is reserved
# for a committed identity that the gallery disagrees with, because that is the
# one state that produces a wrong attendance record.
C_RESOLVED = (60, 220, 60)
C_TRACKED = (0, 190, 255)
C_REJECTED = (150, 150, 150)
C_WRONG = (60, 60, 235)
C_COMMIT = (255, 120, 0)
C_CONTESTED = (0, 215, 255)


@dataclass
class FaceState:
    """Everything known about one face in the most recent frame."""

    track_id: str = ""
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    det_score: float = 0.0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    face_px: int = 0
    blur: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    occlusion: float = 0.0
    outcome: str = "none"
    student_id: str | None = None
    score: float = 0.0
    margin: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    embedded: bool = False
    align_failed: bool = False
    searched_best_px: int = 0
    """Face size of the frame the search actually ran on, which is the best
    frame in the track rather than the current one. Worth showing: it is the
    difference between 'matched on a good view' and 'matched on whatever
    happened to be in front of the lens'."""

    contested_by: str | None = None
    """When contested, the student the fresh evidence is pointing at instead."""

    evidence: dict[str, Any] = field(default_factory=dict)
    report: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id, "box": list(self.box), "det_score": round(self.det_score, 3),
            "passed": self.passed, "reasons": self.reasons, "face_px": self.face_px,
            "blur": round(self.blur, 1), "yaw": round(self.yaw, 1), "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1), "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1), "occlusion": round(self.occlusion, 3),
            "outcome": self.outcome, "student_id": self.student_id,
            "score": round(self.score, 4), "margin": round(self.margin, 4),
            "candidates": self.candidates, "embedded": self.embedded,
            "align_failed": self.align_failed, "searched_best_px": self.searched_best_px,
            "contested_by": self.contested_by, "evidence": self.evidence,
        }


@dataclass
class CommitEvent:
    at: float
    camera_id: str
    student_id: str
    track_id: str
    median_score: float
    margin: float
    evidence_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at, "camera_id": self.camera_id, "student_id": self.student_id,
            "track_id": self.track_id, "median_score": round(self.median_score, 4),
            "margin": round(self.margin, 4), "evidence_frames": self.evidence_frames,
        }


@dataclass
class CameraSlot:
    """Latest annotated frame plus telemetry for one camera.

    Single-writer, single-reader, and the reader tolerates a torn frame. A
    dashboard that crashes because it read a half-written JPEG is worse than
    one that shows a slightly stale frame.
    """

    config: CameraConfig
    source: FrameSource
    frame: npt.NDArray[np.uint8] | None = None
    seq: int = 0
    updated_at: float = 0.0
    detect_ms: float = 0.0
    embed_ms: float = 0.0
    nms_ms: float = 0.0
    detected: int = 0
    kept: int = 0
    fps: float = 0.0
    analysis_fps: float = 0.0
    """Real pipeline throughput. Always <= `fps`, and the number to trust. The
    MJPEG panel can refresh faster than this; the content cannot."""
    skipped: int = 0
    """Frames the detect stride skipped. Non-zero means the stream is
    smoother than the evidence, which is fine for inspection and must never be
    mistaken for more analysis than actually happened."""
    tiles: int = 0
    faces: list[FaceState] = field(default_factory=list)
    error: str = ""
    _fps_count: int = 0
    _analysis_count: int = 0
    _fps_mark: float = 0.0
    _analysis_mark: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.config.id,
            "name": self.config.name,
            "zone": self.config.zone,
            "rotation": self.config.rotation,
            "resolution": [self.config.width, self.config.height],
            "stream_state": self.source.health.state.value,
            "stream_age_s": round(self.source.health.age_s, 2) if self.source.health.last_frame_at else None,
            "decoded_frames": self.source.health.frames_seen,
            "reconnects": self.source.health.reconnects,
            "seq": self.seq,
            "age_s": round(time.time() - self.updated_at, 2) if self.updated_at else None,
            "fps": round(self.fps, 2),
            "analysis_fps": round(self.analysis_fps, 2),
            "skipped_frames": self.skipped,
            "tiles": self.tiles,
            "detected": self.detected,
            "kept": self.kept,
            "detect_ms": round(self.detect_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "total_ms": round(self.detect_ms + self.embed_ms, 1),
            "stream_error": self.source.health.last_error[:200],
            "faces": [f.to_dict() for f in self.faces],
        }


class CameraRunner:
    """One thread per camera running the pipeline and publishing annotated frames."""

    def __init__(
        self,
        config: CameraConfig,
        detector: ScrfdDetector,
        embedder: ArcFaceEmbedder,
        index,
        verification: VerificationThresholds,
        quality,
        backend: DecodeBackend = DecodeBackend.SOFTWARE,
        detect_stride: int = 1,
        stream_fps: float = 15.0,
    ) -> None:
        self.config = config
        self.detector = detector
        self.embedder = embedder
        self.index = index
        self.quality = quality
        self.verification = verification
        self.detect_stride = max(1, int(detect_stride))
        self.stream_fps = max(1.0, float(stream_fps))
        self.tracker = ByteTracker(CameraId(config.id))
        self.verifier = TemporalVerifier(verification)
        self.slot = CameraSlot(
            config=config,
            source=FrameSource(
                StreamConfig(
                    camera_id=config.id,
                    url=config.url,
                    target_fps=config.target_fps,
                    width=config.width,
                    height=config.height,
                    backend=backend,
                    rotation=config.rotation,
                )
            ),
        )
        self.events: list[CommitEvent] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.slot.source.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ui-{self.config.id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.slot.source.stop()

    def _tiles(self, w: int, h: int) -> list[Any]:
        return plan_tiles(w, h, 640, 640, 0.25)

    def _loop(self) -> None:
        while not self._stop.is_set():
            for frame in self.slot.source.frames():
                if self._stop.is_set():
                    return
                try:
                    self._process(frame)
                except Exception as exc:  # noqa: BLE001 - the UI must survive anything
                    self.slot.error = f"{type(exc).__name__}: {exc}"
                    self.slot.source.health.last_error = self.slot.error

    def _process(self, frame: Frame) -> None:
        slot = self.slot

        # Detection is ~80% of the frame cost, so with a stride we skip it
        # entirely on intermediate frames and republish the previous annotated
        # image. That keeps the stream and the HTTP path moving at the full
        # frame rate while detection runs at 1/N of it.
        #
        # The skipped frames contribute NO new evidence: the verifier is not
        # advanced, no embedding is produced, and the UI reports the count.
        # Pretending otherwise would be the easy lie here.
        if self.detect_stride > 1 and (frame.seq % self.detect_stride) != 0:
            slot.skipped += 1
            slot.seq = frame.seq
            slot.updated_at = time.time()
            self._tick_fps(slot, analysed=False)
            return

        t_det0 = time.perf_counter()
        tiles = self._tiles(frame.width, frame.height)
        slot.tiles = len(tiles)
        boxes = self.detector.detect(frame.image, tiles)
        slot.detect_ms = (time.perf_counter() - t_det0) * 1000
        slot.detected = len(boxes)

        gray = frame_gray(frame.image)
        states: list[FaceState] = []
        for b in boxes:
            st = FaceState(
                box=(b.x0, b.y0, b.x1, b.y1), det_score=b.score, face_px=b.min_side
            )
            rep = assess(gray, b, self.quality)
            st.face_px, st.blur = rep.face_px, rep.blur_score
            st.yaw, st.pitch, st.roll = rep.yaw_deg, rep.pitch_deg, rep.roll_deg
            st.brightness, st.contrast, st.occlusion = rep.brightness, rep.contrast, rep.occlusion_ratio
            st.passed, st.reasons = rep.passed, list(rep.reasons)
            st.report = rep
            states.append(st)

        kept = [b for b, st in zip(boxes, states, strict=True) if st.passed]
        slot.kept = len(kept)

        if kept:
            t_emb0 = time.perf_counter()
            vecs, idxs = self.embedder.embed_from_frame(frame.image, kept)
            slot.embed_ms = (time.perf_counter() - t_emb0) * 1000
            tracks = self.tracker.update(kept, frame.seq, frame.timestamp)
            by_box = dict(zip(idxs, range(len(idxs)), strict=True))

            if len(vecs):
                for track, box_i in zip(tracks, by_box.values(), strict=False):
                    st = states[box_i]
                    st.embedded = True
                    st.track_id = track.track_id

                    # Feed this frame into the track's evidence buffer.
                    buffer = self.verifier.record_evidence(
                        track.track_id, CameraId(self.config.id), frame.timestamp,
                        vecs[box_i], st.report, st.det_score,
                    )
                    st.evidence = buffer.stats()

                    # Search on the BEST evidence in the track, not this frame.
                    # A person walking past is frontal for a second or two in
                    # the middle of the crossing; searching whatever arrived
                    # last throws that away and matches nobody.
                    best = buffer.best()
                    if best is None:
                        continue
                    if self.index is None:
                        continue
                    probe = best.embedding[None, :]
                    hits = self.index.search(probe, 5)[0]
                    if not hits:
                        st.outcome = "unknown"
                        continue
                    st.candidates = [
                        {"student_id": c.student_id, "score": round(c.score, 4), "rank": c.rank}
                        for c in hits
                    ]
                    st.score = hits[0].score
                    st.margin = hits[0].score - hits[1].score if len(hits) > 1 else hits[0].score
                    st.searched_best_px = best.face_px
                    outcome, sid, score, _ = self.verifier.observe(
                        track.track_id, CameraId(self.config.id), list(hits),
                        frame.timestamp,
                    )
                    st.outcome = outcome.value
                    st.student_id = sid
                    st.score = score
                    if outcome.value == "contested" and hits:
                        st.contested_by = hits[0].student_id
        for exp in self.tracker.expired():
            c = self.verifier.finalize(exp.track_id)
            if c is not None:
                self.events.insert(0, CommitEvent(
                    at=time.time(), camera_id=self.config.id, student_id=c.student_id,
                    track_id=c.track_id, median_score=c.median_score, margin=c.margin,
                    evidence_frames=c.evidence_frames,
                ))
                del self.events[50:]

        slot.faces = states
        slot.frame = self._annotate(frame.image, boxes, states)
        slot.seq = frame.seq
        slot.updated_at = time.time()
        slot.error = ""
        self._tick_fps(slot, analysed=True)

    def _tick_fps(self, slot: CameraSlot, *, analysed: bool) -> None:
        """Track display rate and true analysis rate separately."""
        slot._fps_count += 1
        if analysed:
            slot._analysis_count += 1
        if slot._fps_mark == 0.0:
            slot._fps_mark = slot.updated_at
            slot._analysis_mark = slot.updated_at
        elif slot.updated_at - slot._fps_mark >= 1.0:
            dt = slot.updated_at - slot._fps_mark
            slot.fps = slot._fps_count / dt
            slot.analysis_fps = slot._analysis_count / dt
            slot._fps_count = 0
            slot._analysis_count = 0
            slot._fps_mark = slot.updated_at

    def _annotate(self, img, boxes, states) -> npt.NDArray[np.uint8]:
        out = img.copy()
        for b, st in zip(boxes, states, strict=True):
            if not st.passed:
                color = C_REJECTED
            elif st.outcome == "contested":
                color = C_CONTESTED
            elif st.student_id and st.outcome in ("identified", "identity_changed"):
                color = C_RESOLVED if st.score >= 0.38 else C_WRONG
            else:
                color = C_TRACKED
            cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), color, 2)
            if b.landmarks is not None and st.passed:
                for lx, ly in b.landmarks:
                    cv2.circle(out, (int(lx), int(ly)), 2, (0, 0, 255), -1)
            if st.outcome == "contested" and st.student_id:
                label = f"CONTESTED {st.student_id}? vs {st.contested_by or '?'}"
            elif st.student_id:
                label = f"{b.min_side}px {st.student_id} {st.score:.2f}"
            elif st.passed:
                label = f"{b.min_side}px searching (best {st.searched_best_px}px)"
            else:
                label = f"{b.min_side}px REJECTED"
            cv2.putText(out, label, (b.x0, max(14, b.y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out


def encode_jpeg(img: npt.NDArray[np.uint8], quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def mjpeg(slots: dict[str, CameraRunner], camera_id: str, fps: float = 15.0) -> Iterator[bytes]:
    """MJPEG stream, re-sending the last annotated frame on a fixed cadence.

    The display rate and the analysis rate are deliberately decoupled. On CPU a
    960x1280 frame costs ~290ms to analyse (230ms of it inference, because this
    SCRFD export is fixed at batch=1 and runs 6 tiles as 6 forward passes), so
    new annotated content arrives at ~3.5fps no matter how fast the browser
    asks. Re-sending the held frame at a higher cadence makes the panel feel
    responsive without inventing data, and the header shows the real analysis
    fps next to the stream so the two are never confused.
    """
    runner = slots.get(camera_id)
    if runner is None:
        return
    interval = 1.0 / max(0.5, fps)
    last_sent: int | None = None
    while True:
        slot = runner.slot
        img, seq = slot.frame, slot.seq
        if img is not None:
            payload = encode_jpeg(img)
            if payload:
                last_sent = seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(payload)).encode()
                    + b"\r\n\r\n" + payload + b"\r\n"
                )
        time.sleep(interval)

"""Configuration loading and validation.

Configuration is per-worker and static for the process lifetime. Camera lists
are assigned to workers by the orchestrator, not by self-selection, because
self-selection produces a split-brain where two workers both believe they own
the busiest camera after a membership change.

The whole config is validated once at startup. A camera with an unreachable
host or an unparseable stream path should fail the deploy, not surface at 3am
as a silently dark camera in a dashboard.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from campus.capture.source import DecodeBackend, StreamConfig
from campus.consent.registry import Purpose, ZonePolicy
from campus.imaging.quality import QualityThresholds
from campus.temporal.verifier import VerificationThresholds


class DetectorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path = Path("models/scrfd_2.5g.onnx")
    device: str | None = None
    input_size: int = 640
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    tile_size: int = 640
    """Tile edge in source-frame pixels.

    Defaults to `input_size` so tiles are fed at 1:1 — a face is exactly as many
    pixels to the detector as it has in the frame. That is the whole point of
    tiling, and getting this wrong is the single most damaging config error:
    a 4K frame tiled at 1280 runs at 0.5x, so a 20px face reaches the detector
    as 10px and becomes undetectable.

    Rule of thumb: set tile_size == input_size for 1080p and below. For 4K,
    1280 (0.5x) is acceptable because faces are large in absolute pixels."""

    tile_overlap: float = 0.25
    """Raising this catches more seam-straddling faces at ~2.8x tile cost per
    0.25. Lower it to 0.125 first if a node is GPU-bound; the diagnostic is
    `tile_count > 1` in the observation telemetry.

    Note that overlap is proportionally more expensive on small frames: at
    1080p/640 a 0.25 overlap processes 1.58x the frame's pixels, and at 720p
    2.67x, because a 640px tile is a much larger fraction of the frame."""

    min_tile_size: int = 320
    """Tiles smaller than this are skipped. A face can only be detected if it
    is roughly a third of the tile, so sub-320px tiles are pure cost."""


class EmbedderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path = Path("models/glintr100k.onnx")
    device: str | None = None
    dim: int = 512
    max_batch: int = 256


class GallerySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["auto", "faiss", "numpy"] = "auto"
    use_gpu: bool = True
    top_k: int = 5
    """Candidates pulled per observation. 5 is enough for the margin check —
    a true match buried at rank 6 will not surface, but on a 30px CCTV face
    that is the right outcome rather than a guess."""

    rebuild_batch: int = 5000
    """Rows per bulk upsert. Large enough to amortise the index rebuild, small
    enough to stay inside the statement timeout."""


class WorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(default_factory=lambda: os.environ.get("CAMPUS_WORKER_ID", "worker-local"))
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://campus:campus@localhost:5432/campus"
    max_cameras: int = 32
    """Cameras per worker. Bounded by GPU memory, not CPU — see ARCHITECTURE.md
    for the sizing table."""

    publish_interval_s: float = 1.0
    """How often observations are flushed to Redis. Batching this matters at
    400 cameras: 3200 msgs/sec individually would spend more time in the Redis
    protocol than in inference."""

    observation_ttl_s: int = 300
    stream_backend: DecodeBackend = DecodeBackend.NVDEC
    jpeg_quality: int = 3
    on_error: Literal["drop_frame", "restart_camera", "fail_worker"] = "drop_frame"
    """Frame-level failures are dropped by default. Restarting the camera
    process for a single corrupt frame would be a self-inflicted outage across
    32 streams; the retry budget belongs in the reconnection backoff."""


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    host: str
    rtsp_path: str = "/Streaming/Channels/101"
    port: int = 554
    zone: str = "default"
    enabled: bool = True
    width: int = 1920
    height: int = 1080
    target_fps: float = 12.0
    """Analysis rate. Higher than 4K needs, because at 1080p the per-frame cost
    is low and the frames are the thing that substitutes for face size: the
    verifier needs 7 agreeing frames out of a 12-frame window, and a higher
    rate gives a person more frames while crossing the detection zone."""

    rotation: int = 0
    """Clockwise degrees, 0/90/180/270. Set this whenever the camera is mounted
    sideways or the image looks turned; a rotated face is detected and then
    rejected by the roll gate, which presents as a camera that sees nobody."""
    @field_validator("rotation")
    @classmethod
    def _valid_rotation(cls, v: int) -> int:
        if v % 90 != 0:
            raise ValueError(f"rotation must be a multiple of 90, got {v}")
        return v % 360
    backend: DecodeBackend | None = None
    username_env: str | None = None
    """Name of the env var holding the camera password. Credentials are never
    in config files — the config is committed and the cameras span departments
    that should not hold each other's passwords."""

    overrides: dict[str, Any] = Field(default_factory=dict)
    """Per-camera threshold overrides, e.g. ``{min_face_px: 40}`` for a
    corridor where people are further away. Merged over the global quality
    thresholds."""

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError("camera id must be non-empty and contain no spaces")
        return v

    @property
    def url(self) -> str:
        """RTSP URL, with userinfo only when a username is actually configured.

        The `@` must be omitted entirely when there is no username. Emitting
        `rtsp://@host:port/path` yields a URL with an empty userinfo section,
        which is malformed — some decoders reject it outright, and the rest
        handle it inconsistently. It surfaced as a camera that opened on one
        run and refused the next, which is the hardest kind of bug to chase.
        """
        user = os.environ.get(self.username_env, "") if self.username_env else ""
        auth = f"{user}@" if user else ""
        return f"rtsp://{auth}{self.host}:{self.port}{self.rtsp_path}"

    def to_stream_config(self, default_fps: float = 8.0) -> StreamConfig:
        return StreamConfig(
            camera_id=self.id,
            url=self.url,
            target_fps=self.target_fps or default_fps,
            width=self.width,
            height=self.height,
            backend=self.backend or DecodeBackend.NVDEC,
            jpeg_quality=3,
            rotation=self.rotation,
        )


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    gallery: GallerySettings = Field(default_factory=GallerySettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    verification: VerificationThresholds = Field(default_factory=VerificationThresholds)
    zones: dict[str, ZonePolicy] = Field(default_factory=dict)
    purposes: list[Purpose] = Field(default_factory=lambda: [Purpose.ATTENDANCE, Purpose.SAFETY])

    @model_validator(mode="after")
    def _zones_present(self) -> SystemConfig:
        if not self.zones:
            self.zones = {
                "default": ZonePolicy(
                    zone="default",
                    allowed_purposes=frozenset(self.purposes),
                    require_consent=True,
                    retention_days=30,
                )
            }
        return self

    def quality_for(self, camera: CameraConfig) -> QualityThresholds:
        """Global thresholds with per-camera overrides applied."""
        if not camera.overrides:
            return self.quality
        return QualityThresholds(
            **{
                **{f: getattr(self.quality, f) for f in self.quality.__slots__},
                **camera.overrides,
            }
        )


def load_system_config(path: str | Path) -> SystemConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SystemConfig.model_validate(raw)


def load_cameras(path: str | Path) -> list[CameraConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("cameras", raw if isinstance(raw, list) else [])
    return [CameraConfig.model_validate(c) for c in entries]


def validate_cameras(cameras: Sequence[CameraConfig]) -> list[str]:
    """Return a list of human-readable problems. Empty means valid.

    Duplicates are the failure that bites: two config entries with the same id
    produce two workers tracking the same camera, two sets of attendance
    records, and a bug report that reads as "the system double-counts people".
    """
    problems: list[str] = []
    seen: dict[str, int] = {}
    for i, cam in enumerate(cameras):
        if cam.id in seen:
            problems.append(
                f"duplicate camera id {cam.id!r} at entries {seen[cam.id]} and {i}"
            )
        else:
            seen[cam.id] = i
        if not cam.host.strip():
            problems.append(f"camera {cam.id!r}: empty host")
        if not cam.rtsp_path.startswith("/"):
            problems.append(f"camera {cam.id!r}: rtsp_path must start with '/'")
        if cam.target_fps <= 0 or cam.target_fps > 60:
            problems.append(f"camera {cam.id!r}: target_fps out of range (0, 60]")
    return problems

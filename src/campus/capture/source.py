"""RTSP ingest and frame sampling.

Decode is the hard part at 400 cameras. A 4K H.265 stream costs real CPU to
decode, and doing it 400 times at 25fps on general-purpose cores does not
fit on any machine we would be given. Hardware decode is not an optimisation
here, it is the difference between feasible and not.

Three decode paths, in order of preference:

* **NVDEC** via FFmpeg's ``h264_cuvid`` / ``hevc_cuvid`` — dedicated silicon,
  ~400 streams per GPU, leaves the CPU entirely alone.
* **VAAPI** for Intel iGPU deployments.
* **libavcodec software decode**, which is the fallback and the reason this
  module also has an fps cap.

`FrameSource` yields whole frames. `FrameSampler` wraps it to yield at a
capped rate, because face recognition does not benefit linearly from frame
rate: 6-8fps catches a person walking through a doorway with a dozen
good frames, and running 25fps burns 4x the GPU for zero accuracy gain.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
import numpy.typing as npt


class DecodeBackend(StrEnum):
    NVDEC = "nvdec"
    VAAPI = "vaapi"
    SOFTWARE = "software"


class StreamState(StrEnum):
    CONNECTING = "connecting"
    LIVE = "live"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StreamConfig:
    camera_id: str
    url: str
    target_fps: float = 8.0
    """Analysis rate. See module docstring for why this is well below the
    stream's native rate."""

    width: int = 3840
    height: int = 2160
    backend: DecodeBackend = DecodeBackend.NVDEC
    reconnect_backoff_s: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
    read_timeout_s: float = 15.0
    jpeg_quality: int = 3
    """FFmpeg ``-q:v`` scale, 2 (best) to 31 (worst). 3 is visually lossless
    and meaningfully cheaper to transfer across the farm."""

    drop_frames: bool = True
    """If the reader cannot keep up, skip frames rather than queue. A backlog
    of stale frames is worse than a gap: it inflates latency and every frame
    in it is stale evidence for a moving person."""

    rotation: int = 0
    """Clockwise rotation in degrees: 0, 90, 180 or 270.

    Required for any camera whose sensor is mounted sideways. Both bench
    cameras deliver portrait containers holding landscape content, so their
    faces arrive rotated ~90 degrees. SCRFD still finds them — it is trained to
    be rotation-tolerant — and then the quality gate correctly rejects every
    one of them on `roll`, which looks exactly like "the camera sees nobody".

    That is the worst possible failure mode: detection reports faces, nothing
    errors, and the answer is silently always wrong. So rotation is applied at
    ingest, before the detector, and the only correct value is whichever one
    the content actually needs."""


@dataclass(slots=True)
class Frame:
    image: npt.NDArray[np.uint8]
    camera_id: str
    seq: int
    timestamp: float
    width: int
    height: int


@dataclass(slots=True)
class StreamHealth:
    camera_id: str
    state: StreamState = StreamState.CONNECTING
    frames_seen: int = 0
    last_frame_at: float = 0.0
    last_error: str = ""
    reconnects: int = 0
    decode_fps: float = 0.0
    _last_measure: float = field(default=0.0)
    _frames_at_measure: int = 0

    @property
    def age_s(self) -> float:
        return time.time() - self.last_frame_at if self.last_frame_at else float("inf")

    def is_stale(self, threshold_s: float = 10.0) -> bool:
        """No frame in `threshold_s`.

        A camera that stops delivering is invisible in a dashboard that only
        shows frame counts, and a silently dead camera looks exactly like an
        empty corridor. Every health surface must call this.
        """
        return self.last_frame_at > 0 and self.age_s > threshold_s


class FrameSource:
    """One camera, yielding decoded frames at `target_fps`.

    Wraps either a hardware-decoded FFmpeg subprocess or OpenCV's VideoCapture,
    with transparent reconnection. Reconnection uses exponential backoff with
    a floor, because a 400-camera fleet where every camera retries every 5
    seconds after a switch reboot will re-trigger the same DDoS pattern that
    the XMEye units in this repo already rate-limit against.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.health = StreamHealth(camera_id=config.camera_id)
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._proc: subprocess.Popen[bytes] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self.health.state = StreamState.CONNECTING

    def stop(self) -> None:
        self._stop.set()
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self._proc = None

    # -- iteration ---------------------------------------------------------

    def frames(self) -> Iterator[Frame]:
        """Yield frames forever, reconnecting as needed.

        Never raises on a transient network fault; a camera going down must not
        take down the worker holding 30 other cameras. A camera that cannot be
        recovered exhausts its backoff and moves to FAILED, where it is visible
        to health checks instead of spinning forever.
        """
        backoff_idx = 0
        seq = 0

        while not self._stop.is_set():
            try:
                reader = self._open()
            except (OSError, RuntimeError) as exc:
                self.health.state = StreamState.FAILED
                self.health.last_error = str(exc)
                if backoff_idx >= len(self.config.reconnect_backoff_s):
                    return
                time.sleep(self.config.reconnect_backoff_s[backoff_idx])
                backoff_idx += 1
                continue

            backoff_idx = 0
            self.health.state = StreamState.LIVE
            last = time.monotonic()
            interval = 1.0 / max(0.1, self.config.target_fps)

            try:
                for image, ts in reader:
                    if self._stop.is_set():
                        return
                    now = time.monotonic()
                    if now - last < interval:
                        if self.config.drop_frames:
                            continue
                        time.sleep(interval - (now - last))
                    last = time.monotonic()

                    if image is None:
                        break

                    image = _apply_rotation(image, self.config.rotation)
                    seq += 1
                    h, w = image.shape[:2]
                    self.health.frames_seen += 1
                    self.health.last_frame_at = time.time()
                    self.health.state = StreamState.LIVE
                    self._measure(now)

                    yield Frame(
                        image=image, camera_id=self.config.camera_id, seq=seq,
                        timestamp=ts, width=w, height=h,
                    )
            except (OSError, RuntimeError, cv2.error) as exc:
                self.health.last_error = str(exc)
            finally:
                self._release()

            if self._stop.is_set():
                return
            self.health.state = StreamState.RECONNECTING
            self.health.reconnects += 1
            delay = self.config.reconnect_backoff_s[
                min(backoff_idx, len(self.config.reconnect_backoff_s) - 1)
            ]
            backoff_idx += 1
            time.sleep(delay)

    def _measure(self, now: float) -> None:
        if self.health._last_measure == 0.0:
            self.health._last_measure = now
            self.health._frames_at_measure = self.health.frames_seen
            return
        elapsed = now - self.health._last_measure
        if elapsed >= 5.0:
            delta = self.health.frames_seen - self.health._frames_at_measure
            self.health.decode_fps = delta / elapsed
            self.health._last_measure = now
            self.health._frames_at_measure = self.health.frames_seen
            self.health.state = (
                StreamState.LIVE
                if self.health.decode_fps >= self.config.target_fps * 0.5
                else StreamState.STALLED
            )

    def _open(self) -> Iterator[tuple[npt.NDArray[np.uint8] | None, float]]:
        if self.config.backend is DecodeBackend.SOFTWARE:
            return self._open_opencv()
        return self._open_ffmpeg()

    def _open_opencv(self) -> Iterator[tuple[npt.NDArray[np.uint8] | None, float]]:
        cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise OSError(f"cannot open {self.config.url}")
        self._cap = cap
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            yield frame, time.time()

    def _open_ffmpeg(self) -> Iterator[tuple[npt.NDArray[np.uint8] | None, float]]:
        """Hardware-decode via an FFmpeg subprocess piping raw BGR.

        Subprocess rather than in-process libav bindings: it keeps the codec
        and driver dependency isolated to one restartable unit, so a driver
        fault kills a child instead of the worker.
        """
        decoder = {
            DecodeBackend.NVDEC: {
                "h264": ["-hwaccel", "cuda", "-c:v", "h264_cuvid"],
                "h265": ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"],
            },
            DecodeBackend.VAAPI: {
                "h264": ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-c:v", "h264_vaapi"],
                "h265": ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-c:v", "hevc_vaapi"],
            },
        }.get(self.config.backend)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if decoder is not None:
            codec_probe = self._probe_codec()
            cmd += decoder.get(codec_probe, list(decoder.values())[0])
        cmd += [
            "-rtsp_transport", "tcp",
            "-i", self.config.url,
            "-an", "-sn", "-dn",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo", "-",
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=10**8, creationflags=_no_window(),
        )
        self._proc = proc

        frame_bytes = self.config.width * self.config.height * 3
        stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(proc,), daemon=True
        )
        stderr_thread.start()

        while not self._stop.is_set():
            buf = proc.stdout.read(frame_bytes) if proc.stdout else b""
            if not buf or len(buf) < frame_bytes:
                return
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                self.config.height, self.config.width, 3
            )
            yield frame, time.time()

    def _probe_codec(self) -> str:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
            self.config.url,
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15,
                creationflags=_no_window(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return "h265"
        return "h264" if "h264" in out.stdout else "h265"

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.decode("utf-8", "replace").strip()
            if text:
                self.health.last_error = text[:500]


def _apply_rotation(image: npt.NDArray[np.uint8], degrees: int) -> npt.NDArray[np.uint8]:
    """Rotate clockwise by 0/90/180/270, returning a contiguous array.

    `np.rot90` is counter-clockwise and returns a view with swapped strides,
    which OpenCV and ONNX both reject downstream. Both are fixed here rather
    than at each call site.
    """
    if degrees % 360 == 0:
        return image
    k = {90: 3, 180: 2, 270: 1}.get(degrees % 360)
    if k is None:
        raise ValueError(f"rotation must be 0, 90, 180 or 270, got {degrees}")
    return np.ascontiguousarray(np.rot90(image, k))


def _no_window() -> int:
    """Suppress the console window FFmpeg would flash on Windows."""
    import subprocess as sp  # noqa: PLC0415

    return sp.CREATE_NO_WINDOW if hasattr(sp, "CREATE_NO_WINDOW") else 0


@dataclass(slots=True)
class CameraWorkerStats:
    cameras: int = 0
    frames_processed: int = 0
    faces_detected: int = 0
    faces_embedded: int = 0
    faces_rejected_quality: int = 0
    identities_committed: int = 0
    inference_ms: float = 0.0

    def snapshot(self) -> dict[str, float | int]:
        n = max(1, self.frames_processed)
        return {
            "cameras": self.cameras,
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "faces_embedded": self.faces_embedded,
            "faces_rejected_quality": self.faces_rejected_quality,
            "identities_committed": self.identities_committed,
            "ms_per_frame": round(self.inference_ms / n, 2),
        }


def frame_gray(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Grayscale, tolerating grayscale or BGRA input."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

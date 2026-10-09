"""Runtime capability probing.

``campus validate`` answers "is my config shaped right". It does not answer
"will this machine actually run inference", which is the question that decides
whether a deploy works.

That gap is not hypothetical. Two failure modes hit during development, and
both are invisible to a config check:

1. **A GPU that is present but not enumerating.** A driver package can install
   cleanly — CUDA runtime, NVML, display container all healthy — while the
   adapter itself sits in ``CM_PROB_PHANTOM`` and never binds. Every symptom
   looks like a permissions problem, so it gets misdiagnosed as one.
2. **A CPU-only ONNX Runtime build.** ``onnxruntime`` reports its providers
   honestly, but nothing checks them, so a node silently runs 20x slower than
   planned and presents as "the pipeline is just slow".

Both are silent. Neither raises. So the check exists, and it is a *smoke test*
rather than an inspection: it builds a real session and runs a real forward
pass, because ORT's cuDNN/cuBLAS DLL loading fails at session-creation time,
long after ``get_available_providers()`` cheerfully reports CUDA.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

# Probing subprocesses must never hang a deploy check.
_PROBE_TIMEOUT_S = 30


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str
    driver_version: str = ""
    cuda_version: str = ""
    vram_mb: int = 0
    available: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "driver": self.driver_version,
            "cuda": self.cuda_version,
            "vram_mb": self.vram_mb,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(slots=True)
class RuntimeReport:
    gpus: list[GpuInfo] = field(default_factory=list)
    nvidia_smi: bool = False
    onnxruntime_version: str = ""
    onnxruntime_device: str = ""
    providers: list[str] = field(default_factory=list)
    cuda_session_ok: bool = False
    cuda_error: str = ""
    faiss_version: str = ""
    faiss_gpu: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def has_gpu_inference(self) -> bool:
        return self.cuda_session_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvidia_smi": self.nvidia_smi,
            "gpus": [g.to_dict() for g in self.gpus],
            "onnxruntime": {
                "version": self.onnxruntime_version,
                "device": self.onnxruntime_device,
                "providers": self.providers,
                "cuda_session_ok": self.cuda_session_ok,
                "cuda_error": self.cuda_error,
            },
            "faiss": {"version": self.faiss_version, "gpu": self.faiss_gpu},
            "has_gpu_inference": self.has_gpu_inference,
            "problems": self.problems,
        }


def query_gpus() -> tuple[bool, list[GpuInfo]]:
    """Enumerate GPUs via nvidia-smi. Returns ``(nvidia_smi_found, gpus)``.

    Reports *enumerated* GPUs only. A driver package that is installed but
    bound to nothing shows up here as zero GPUs, which is the signal that
    distinguishes a missing driver from a missing device.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, []

    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            creationflags=_no_window(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return True, [GpuInfo(0, "unknown", available=False, detail=str(exc))]

    if out.returncode != 0:
        detail = (out.stderr or out.stdout or "").strip()[:200]
        return True, [GpuInfo(0, "unknown", available=False, detail=detail)]

    gpus: list[GpuInfo] = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    driver_version=parts[2],
                    vram_mb=int(float(parts[3])),
                    available=True,
                )
            )
        except ValueError:
            continue

    try:
        cuda = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
            creationflags=_no_window(),
        )
        if cuda.returncode == 0 and cuda.stdout.strip():
            gpus[0].cuda_version = cuda.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return True, gpus


def probe_onnxruntime() -> RuntimeReport:
    """Inspect ONNX Runtime, then prove CUDA with a real forward pass."""
    report = RuntimeReport()
    report.nvidia_smi, report.gpus = query_gpus()

    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        report.problems.append("onnxruntime is not installed (pip install '.[inference]')")
        return report

    report.onnxruntime_version = ort.__version__
    report.providers = list(ort.get_available_providers())
    report.onnxruntime_device = ort.get_device()

    if "CUDAExecutionProvider" in report.providers:
        report.cuda_session_ok, report.cuda_error = _cuda_smoke_test(ort)
    else:
        report.cuda_error = (
            f"onnxruntime build has no CUDAExecutionProvider "
            f"(providers: {', '.join(report.providers)}). "
            f"Install onnxruntime-gpu for GPU inference."
        )

    if not report.gpus and report.nvidia_smi:
        report.problems.append(
            "nvidia-smi is present but reports no GPU. The driver package is "
            "installed but not bound to a device (check Device Manager for a "
            "phantom adapter, and GPU Power Saving / dGPU power state)."
        )
    if report.gpus and not report.cuda_session_ok:
        report.problems.append(
            f"GPU is present but ONNX Runtime cannot use it: {report.cuda_error}"
        )
    if not report.nvidia_smi and not report.gpus:
        report.problems.append(
            "no NVIDIA GPU detected. CPU inference is supported for development "
            "and enrollment, but is far too slow for the fleet."
        )
    return report


def _cuda_smoke_test(ort: Any) -> tuple[bool, str]:
    """Build a trivial CUDA session and run it.

    The provider being *listed* is not the same as it being *usable*. CUDA
    session creation is where a missing cuDNN or cuBLAS DLL surfaces, and that
    is the most common onnxruntime-gpu failure on Windows.
    """
    import numpy as np  # noqa: PLC0415

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(
            _identity_model_path(), sess_options=so, providers=["CUDAExecutionProvider"]
        )
        name = sess.get_inputs()[0].name
        sess.run(None, {name: np.zeros((1, 4), dtype=np.float32)})
    except Exception as exc:  # noqa: BLE001 - any failure means CUDA is unusable
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, ""


def _identity_model_path() -> str:
    """Path to a minimal ONNX graph, generating it once if absent.

    Avoids needing real model weights just to prove CUDA works, which is the
    whole point: the device check must be runnable before the models land.
    """
    from pathlib import Path  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    target = Path(sys.executable).parent / "campus_probe.onnx"
    if target.exists():
        return str(target)
    try:
        onnx = __import__("onnx")
        from onnx import helper, numpy_helper  # noqa: PLC0415

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node], "probe",
            [helper.make_tensor_value_info("x", 1, [1, 4])],
            [helper.make_tensor_value_info("y", 1, [1, 4])],
        )
        model = helper.make_model(graph)
        onnx.save(model, str(target))
    except Exception:  # noqa: BLE001 - fall back to whatever onnx ships
        return ""
    return str(target)


def probe_faiss() -> tuple[str, bool]:
    """Return ``(version, gpu_enabled)``. Empty version means not installed."""
    try:
        import faiss  # noqa: PLC0415
    except ImportError:
        return "", False
    version = getattr(faiss, "__version__", "unknown")
    return version, bool(getattr(faiss, "get_num_gpus", lambda: 0)())


def full_report() -> RuntimeReport:
    report = probe_onnxruntime()
    report.faiss_version, report.faiss_gpu = probe_faiss()
    if not report.faiss_version:
        report.problems.append(
            "faiss is not installed; gallery search will fall back to NumPy "
            "(correct but much slower at fleet scale). pip install '.[index]'"
        )
    return report


def format_report(report: RuntimeReport) -> str:
    d = report.to_dict()
    lines = [
        f"  nvidia-smi       : {'found' if d['nvidia_smi'] else 'NOT FOUND'}",
        f"  GPUs             : {len(d['gpus'])}",
    ]
    for g in d["gpus"]:
        lines.append(
            f"    [{g['index']}] {g['name']}  {g['vram_mb']}MB  driver {g['driver']}"
        )
    lines += [
        f"  onnxruntime      : {d['onnxruntime']['version']} ({d['onnxruntime']['device']})",
        f"  providers        : {', '.join(d['onnxruntime']['providers']) or 'none'}",
        f"  CUDA session     : {'OK' if d['onnxruntime']['cuda_session_ok'] else 'FAILED'}",
        f"  faiss            : {d['faiss']['version'] or 'not installed'}"
        f"{' (GPU)' if d['faiss']['gpu'] else ''}",
        f"  GPU inference    : {'yes' if d['has_gpu_inference'] else 'NO'}",
    ]
    return "\n".join(lines)


def _no_window() -> int:
    import subprocess as sp  # noqa: PLC0415

    return sp.CREATE_NO_WINDOW if hasattr(sp, "CREATE_NO_WINDOW") else 0


def dumps(report: RuntimeReport) -> str:
    return json.dumps(report.to_dict(), indent=2)

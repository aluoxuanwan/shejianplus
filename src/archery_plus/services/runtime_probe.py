from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeProbeResult:
    onnxruntime_ok: bool = False
    onnx_providers: list[str] = field(default_factory=list)
    opencv_ok: bool = False
    cuda_device_count: int = 0
    notes: list[str] = field(default_factory=list)


def probe_runtime() -> RuntimeProbeResult:
    result = RuntimeProbeResult()

    try:
        import onnxruntime as ort  # type: ignore

        result.onnxruntime_ok = True
        result.onnx_providers = ort.get_available_providers()
    except Exception as exc:  # pragma: no cover
        result.notes.append(f"ONNX Runtime 不可用: {exc}")

    try:
        import cv2  # type: ignore

        result.opencv_ok = True
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            result.cuda_device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        else:
            result.notes.append("OpenCV 未编译 CUDA 模块")
    except Exception as exc:  # pragma: no cover
        result.notes.append(f"OpenCV 不可用: {exc}")

    return result

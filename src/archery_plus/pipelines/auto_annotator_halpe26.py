from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


HALPE26_NAMES = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "Head",
    "Neck",
    "Hip",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
]


@dataclass
class AutoAnnotateResult:
    points: list[dict[str, float | str | int]]
    raw_count: int


class Halpe26AutoAnnotator:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._input_hw: tuple[int, int] | None = None

    def is_ready(self) -> bool:
        try:
            self._ensure_session()
            return True
        except Exception:
            return False

    def set_thresholds(self, confidence_threshold: float, nms_threshold: float) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)

    def predict_image(self, image_path: Path) -> AutoAnnotateResult:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image_bgr.shape[:2]
        input_tensor = self._preprocess(image_bgr)
        simcc_x, simcc_y = self._infer(input_tensor)
        points = self._decode_to_points(simcc_x, simcc_y, orig_w=orig_w, orig_h=orig_h)

        filtered = [p for p in points if float(p["score"]) >= self.confidence_threshold]
        filtered = self._nms_points(filtered, iou_threshold=self.nms_threshold)

        return AutoAnnotateResult(points=filtered, raw_count=len(points))

    def _ensure_session(self) -> None:
        if self._session is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Force CPU to avoid noisy CUDA provider warnings on machines without full cuDNN runtime.
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name

        shape = input_meta.shape
        if len(shape) != 4:
            raise RuntimeError(f"Unsupported model input shape: {shape}")

        in_h = int(shape[2]) if isinstance(shape[2], int) else 384
        in_w = int(shape[3]) if isinstance(shape[3], int) else 288
        self._input_hw = (in_h, in_w)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        self._ensure_session()
        assert self._input_hw is not None

        in_h, in_w = self._input_hw
        resized = cv2.resize(image_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image_rgb = (image_rgb - mean) / std

        chw = np.transpose(image_rgb, (2, 0, 1))[None, :, :, :]
        return chw.astype(np.float32)

    def _infer(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_session()
        assert self._session is not None
        assert self._input_name is not None

        outputs = self._session.run(None, {self._input_name: input_tensor})
        if len(outputs) < 2:
            raise RuntimeError("Unexpected model outputs, need simcc_x and simcc_y")
        simcc_x = np.asarray(outputs[0], dtype=np.float32)
        simcc_y = np.asarray(outputs[1], dtype=np.float32)
        return simcc_x, simcc_y

    def _decode_to_points(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        *,
        orig_w: int,
        orig_h: int,
    ) -> list[dict[str, float | str | int]]:
        self._ensure_session()
        assert self._input_hw is not None

        if simcc_x.ndim != 3 or simcc_y.ndim != 3:
            raise RuntimeError(f"Unsupported output dims: x={simcc_x.shape}, y={simcc_y.shape}")

        x_dist = simcc_x[0]
        y_dist = simcc_y[0]
        kp_count = min(x_dist.shape[0], y_dist.shape[0])

        in_h, in_w = self._input_hw
        split_x = x_dist.shape[1] / max(in_w, 1)
        split_y = y_dist.shape[1] / max(in_h, 1)
        split_x = split_x if split_x > 0 else 2.0
        split_y = split_y if split_y > 0 else 2.0

        points: list[dict[str, float | str | int]] = []
        for idx in range(kp_count):
            x_logits = x_dist[idx]
            y_logits = y_dist[idx]

            x_idx = int(np.argmax(x_logits))
            y_idx = int(np.argmax(y_logits))

            x_conf = self._sigmoid(float(np.max(x_logits)))
            y_conf = self._sigmoid(float(np.max(y_logits)))
            score = float(np.sqrt(max(x_conf, 0.0) * max(y_conf, 0.0)))

            x_in = x_idx / split_x
            y_in = y_idx / split_y
            x = x_in * orig_w / max(in_w, 1)
            y = y_in * orig_h / max(in_h, 1)

            name = HALPE26_NAMES[idx] if idx < len(HALPE26_NAMES) else f"H{idx:02d}"
            points.append(
                {
                    "name": name,
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "score": round(score, 4),
                    "v": 2,
                }
            )

        return points

    def _sigmoid(self, x: float) -> float:
        x = np.clip(x, -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def _nms_points(
        self,
        points: list[dict[str, float | str | int]],
        iou_threshold: float,
    ) -> list[dict[str, float | str | int]]:
        # This model is single-person top-down in current MVP.
        # Keep a lightweight point-level suppression so the nms control has effect.
        if not points:
            return []

        threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        if threshold >= 0.999:
            return points

        radius = max(2.0, (1.0 - threshold) * 20.0)
        radius2 = radius * radius

        kept: list[dict[str, float | str | int]] = []
        for point in sorted(points, key=lambda p: float(p.get("score", 0.0)), reverse=True):
            px = float(point["x"])
            py = float(point["y"])
            duplicate = False
            for kept_point in kept:
                dx = px - float(kept_point["x"])
                dy = py - float(kept_point["y"])
                if (dx * dx + dy * dy) <= radius2:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(point)

        return kept

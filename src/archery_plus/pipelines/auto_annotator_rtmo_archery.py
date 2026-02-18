from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from archery_plus.core.keypoint_schema import DEFAULT_ARCHERY_KEYPOINT_NAMES

ARCHERY_KEYPOINTS = list(DEFAULT_ARCHERY_KEYPOINT_NAMES)


@dataclass
class ArcheryAutoAnnotateResult:
    points: list[dict[str, float | str | int]]
    raw_detection_count: int
    kept_detection_count: int


class RtmoArcheryAutoAnnotator:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._input_hw: tuple[int, int] | None = None

    def is_ready(self) -> bool:
        try:
            self._ensure_session()
            return True
        except Exception:
            return False

    def set_thresholds(self, confidence_threshold: float, iou_threshold: float) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

    def predict_image(self, image_path: Path) -> ArcheryAutoAnnotateResult:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image_bgr.shape[:2]
        input_tensor, ratio, pad_x, pad_y = self._preprocess(image_bgr)
        dets, keypoints = self._infer(input_tensor)

        boxes, kps = self._normalize_outputs(dets, keypoints)
        raw_count = int(boxes.shape[0])
        if raw_count == 0:
            return ArcheryAutoAnnotateResult(points=[], raw_detection_count=0, kept_detection_count=0)

        conf_scores = self._sigmoid_array(boxes[:, 4])
        conf_mask = conf_scores >= self.confidence_threshold
        boxes = boxes[conf_mask]
        kps = kps[conf_mask]
        conf_scores = conf_scores[conf_mask]
        if boxes.shape[0] == 0:
            return ArcheryAutoAnnotateResult(points=[], raw_detection_count=raw_count, kept_detection_count=0)

        keep = self._nms_indices(boxes, iou_threshold=self.iou_threshold)
        if keep.size == 0:
            return ArcheryAutoAnnotateResult(points=[], raw_detection_count=raw_count, kept_detection_count=0)

        best_idx = keep[np.argmax(conf_scores[keep])]
        selected = kps[best_idx]

        points: list[dict[str, float | str | int]] = []
        for idx in range(min(selected.shape[0], len(ARCHERY_KEYPOINTS))):
            x = (float(selected[idx, 0]) - pad_x) / ratio
            y = (float(selected[idx, 1]) - pad_y) / ratio
            x = min(max(x, 0.0), float(orig_w - 1))
            y = min(max(y, 0.0), float(orig_h - 1))
            kp_score = float(selected[idx, 2])
            points.append(
                {
                    "name": ARCHERY_KEYPOINTS[idx],
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "score": round(kp_score, 4),
                    "v": 2,
                }
            )

        return ArcheryAutoAnnotateResult(
            points=points,
            raw_detection_count=raw_count,
            kept_detection_count=int(keep.size),
        )

    def _ensure_session(self) -> None:
        if self._session is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name

        shape = input_meta.shape
        if len(shape) != 4:
            raise RuntimeError(f"Unsupported model input shape: {shape}")

        in_h = int(shape[2]) if isinstance(shape[2], int) else 640
        in_w = int(shape[3]) if isinstance(shape[3], int) else 640
        self._input_hw = (in_h, in_w)

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        self._ensure_session()
        assert self._input_hw is not None

        in_h, in_w = self._input_hw
        letterboxed, ratio, pad_x, pad_y = self._letterbox(image_bgr, in_w, in_h)
        chw = np.transpose(letterboxed.astype(np.float32), (2, 0, 1))[None, :, :, :]
        return chw.astype(np.float32), ratio, pad_x, pad_y

    def _letterbox(
        self,
        image: np.ndarray,
        target_w: int,
        target_h: int,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, float, float]:
        src_h, src_w = image.shape[:2]
        ratio = min(target_w / max(src_w, 1), target_h / max(src_h, 1))

        new_w = int(round(src_w * ratio))
        new_h = int(round(src_h * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        pad_x = float((target_w - new_w) // 2)
        pad_y = float((target_h - new_h) // 2)

        x0 = int(pad_x)
        y0 = int(pad_y)
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas, float(ratio), pad_x, pad_y

    def _infer(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_session()
        assert self._session is not None
        assert self._input_name is not None

        outputs = self._session.run(None, {self._input_name: input_tensor})
        if len(outputs) < 2:
            raise RuntimeError("Unexpected RTMO outputs, need dets and keypoints")

        dets = np.asarray(outputs[0], dtype=np.float32)
        keypoints = np.asarray(outputs[1], dtype=np.float32)
        return dets, keypoints

    def _normalize_outputs(self, dets: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if dets.ndim == 3:
            dets = dets[0]
        if keypoints.ndim == 4:
            keypoints = keypoints[0]

        if dets.ndim != 2 or dets.shape[1] < 5:
            raise RuntimeError(f"Unexpected det shape: {dets.shape}")
        if keypoints.ndim != 3 or keypoints.shape[2] < 3:
            raise RuntimeError(f"Unexpected keypoints shape: {keypoints.shape}")

        if keypoints.shape[0] != dets.shape[0]:
            count = min(dets.shape[0], keypoints.shape[0])
            dets = dets[:count]
            keypoints = keypoints[:count]

        return dets, keypoints

    def _nms_indices(self, boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)

        threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = self._sigmoid_array(boxes[:, 4])

        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.asarray(keep, dtype=np.int64)

    def _sigmoid_array(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x.astype(np.float64), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

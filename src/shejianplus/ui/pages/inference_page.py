from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shejianplus.config import BASE_DIR, MODEL_FILES
from shejianplus.core.ema_filter import ButterworthBidirectionalPointFilter, OneEuroPointFilter
from shejianplus.core.inference_session_cache import set_inference_session_snapshot
from shejianplus.core.inference_session_exporter import export_inference_session_timeseries
from shejianplus.core.keypoint_schema import DEFAULT_ARCHERY_KEYPOINT_NAMES, DEFAULT_HUMAN_KEYPOINT_NAMES
from shejianplus.core.scale_converter import BowScaleConverter
from shejianplus.pipelines.auto_annotator_halpe26 import Halpe26AutoAnnotator
from shejianplus.pipelines.auto_annotator_rtmo_archery import RtmoArcheryAutoAnnotator
from shejianplus.ui.widgets.inference_analysis_dialog import AnalysisDialog
from shejianplus.ui.widgets.realtime_overlay_widget import RealtimeOverlayWidget

KEYPOINT_NAMES = list(DEFAULT_ARCHERY_KEYPOINT_NAMES)
HUMAN_KEYPOINT_NAMES = list(DEFAULT_HUMAN_KEYPOINT_NAMES)
FILTER_MODE_ONE_EURO = "one_euro"
FILTER_MODE_BUTTER = "butterworth"
TARGET_ARCHERY = "archery"
TARGET_HUMAN = "human"
TARGET_BOTH = "both"



class InferenceWorker(QObject):
    frameReady = Signal(object)
    logMessage = Signal(str)
    fatalError = Signal(str)
    finished = Signal()

    def __init__(
        self,
        video_path: Path,
        model_paths: dict[str, Path],
        target_mode: str,
        confidence: float,
        iou: float,
        smooth_alpha: float,
        bow_length_cm: float,
        filter_mode: str = FILTER_MODE_ONE_EURO,
        one_euro_min_cutoff: float = 2.8,
        one_euro_beta: float = 4.2,
        one_euro_d_cutoff: float = 1.0,
        butter_cutoff_hz: float = 3.0,
        butter_window_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self._video_path = Path(video_path)
        self._target_mode = str(target_mode)
        self._keypoint_names = self._resolve_keypoint_names(self._target_mode)
        self._model_paths = {str(k): Path(v) for k, v in model_paths.items()}
        self._annotators = self._build_annotators(self._model_paths, self._target_mode)
        self._one_euro_filters: dict[str, OneEuroPointFilter] = {}
        self._butter_filters: dict[str, ButterworthBidirectionalPointFilter | None] = {}
        for tgt in self._enabled_targets(self._target_mode):
            oe = OneEuroPointFilter(alpha=smooth_alpha)
            oe.set_one_euro_params(
                min_cutoff=one_euro_min_cutoff,
                beta=one_euro_beta,
                d_cutoff=one_euro_d_cutoff,
            )
            self._one_euro_filters[tgt] = oe
            self._butter_filters[tgt] = None
        self._filter_mode = FILTER_MODE_ONE_EURO
        self._init_butterworth(butter_cutoff_hz, sample_hz=30.0, window_seconds=butter_window_seconds)
        self._set_filter_mode(filter_mode)
        self._scale = (
            BowScaleConverter(bow_length_cm=bow_length_cm, scale_conf_threshold=confidence)
            if self._target_mode in (TARGET_ARCHERY, TARGET_BOTH)
            else None
        )
        self._scale_raw = (
            BowScaleConverter(bow_length_cm=bow_length_cm, scale_conf_threshold=confidence)
            if self._target_mode in (TARGET_ARCHERY, TARGET_BOTH)
            else None
        )

        self._lock = threading.Lock()
        self._running = True
        self._paused = False
        self._params = {
            "confidence": float(confidence),
            "iou": float(iou),
            "target_mode": str(target_mode),
            "smooth_alpha": float(smooth_alpha),
            "bow_length_cm": float(bow_length_cm),
            "filter_mode": str(filter_mode),
            "one_euro_min_cutoff": float(one_euro_min_cutoff),
            "one_euro_beta": float(one_euro_beta),
            "one_euro_d_cutoff": float(one_euro_d_cutoff),
            "butter_cutoff_hz": float(butter_cutoff_hz),
            "butter_window_seconds": float(butter_window_seconds),
        }
        self._last_drop_log_ts = 0.0

    def _resolve_keypoint_names(self, target_mode: str) -> list[str]:
        mode = str(target_mode)
        if mode == TARGET_HUMAN:
            return list(HUMAN_KEYPOINT_NAMES)
        if mode == TARGET_BOTH:
            return list(KEYPOINT_NAMES) + list(HUMAN_KEYPOINT_NAMES)
        return list(KEYPOINT_NAMES)

    def _enabled_targets(self, target_mode: str | None = None) -> list[str]:
        mode = str(target_mode or self._target_mode)
        if mode == TARGET_BOTH:
            return [TARGET_ARCHERY, TARGET_HUMAN]
        if mode == TARGET_HUMAN:
            return [TARGET_HUMAN]
        return [TARGET_ARCHERY]

    def _build_annotators(self, model_paths: dict[str, Path], target_mode: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tgt in self._enabled_targets(target_mode):
            model_path = model_paths.get(tgt)
            if model_path is None:
                raise RuntimeError(f"缺少推理模型路径: {tgt}")
            if tgt == TARGET_HUMAN:
                out[tgt] = Halpe26AutoAnnotator(model_path=model_path)
            else:
                out[tgt] = RtmoArcheryAutoAnnotator(model_path=model_path)
        return out

    def _init_butterworth(self, cutoff_hz: float, sample_hz: float, window_seconds: float) -> None:
        for tgt in self._enabled_targets():
            try:
                butter_filter = self._butter_filters.get(tgt)
                if butter_filter is None:
                    butter_filter = ButterworthBidirectionalPointFilter(
                        cutoff_hz=float(cutoff_hz),
                        sample_hz=float(sample_hz),
                        window_seconds=float(window_seconds),
                    )
                    self._butter_filters[tgt] = butter_filter
                else:
                    butter_filter.set_cutoff_hz(float(cutoff_hz))
                    butter_filter.set_sample_hz(float(sample_hz))
                    butter_filter.set_window_seconds(float(window_seconds))
            except Exception as exc:
                self._butter_filters[tgt] = None
                self.logMessage.emit(f"[WARN] 巴特沃斯滤波不可用({tgt})，已回退 One Euro: {exc}")

    def _set_filter_mode(self, mode: str) -> None:
        mode_text = str(mode).strip().lower()
        if mode_text == FILTER_MODE_BUTTER and any(self._butter_filters.get(t) is not None for t in self._enabled_targets()):
            self._filter_mode = FILTER_MODE_BUTTER
        else:
            self._filter_mode = FILTER_MODE_ONE_EURO

    def _reset_filters(self) -> None:
        for oe in self._one_euro_filters.values():
            oe.reset()
        for bf in self._butter_filters.values():
            if bf is not None:
                bf.reset()
        if self._scale is not None:
            self._scale.reset()
        if self._scale_raw is not None:
            self._scale_raw.reset()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def update_params(
        self,
        confidence: float,
        iou: float,
        smooth_alpha: float,
        bow_length_cm: float,
        filter_mode: str,
        one_euro_min_cutoff: float,
        one_euro_beta: float,
        one_euro_d_cutoff: float,
        butter_cutoff_hz: float,
        butter_window_seconds: float,
    ) -> None:
        with self._lock:
            self._params["confidence"] = float(confidence)
            self._params["iou"] = float(iou)
            # Target/model is fixed during one worker run.
            self._params["smooth_alpha"] = float(smooth_alpha)
            self._params["bow_length_cm"] = float(bow_length_cm)
            self._params["filter_mode"] = str(filter_mode)
            self._params["one_euro_min_cutoff"] = float(one_euro_min_cutoff)
            self._params["one_euro_beta"] = float(one_euro_beta)
            self._params["one_euro_d_cutoff"] = float(one_euro_d_cutoff)
            self._params["butter_cutoff_hz"] = float(butter_cutoff_hz)
            self._params["butter_window_seconds"] = float(butter_window_seconds)

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = cv2.VideoCapture(str(self._video_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self._video_path}")

            for tgt, annotator in self._annotators.items():
                if not annotator.is_ready():
                    model_desc = "RTMPose(HALPE26)" if tgt == TARGET_HUMAN else "RTMO(弓箭)"
                    raise RuntimeError(f"{model_desc} ONNX 模型不可用，请检查模型文件。")

            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps <= 1e-6:
                src_fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._init_butterworth(
                self._params["butter_cutoff_hz"],
                sample_hz=src_fps,
                window_seconds=self._params["butter_window_seconds"],
            )
            self._reset_filters()

            self.logMessage.emit(f"[INFO] 开始推理: {self._video_path}")
            self.logMessage.emit(f"[INFO] 视频FPS={src_fps:.2f}, 总帧数={total_frames}")

            while True:
                with self._lock:
                    running = self._running
                    paused = self._paused
                    p = dict(self._params)

                if not running:
                    break

                if paused:
                    time.sleep(0.03)
                    continue

                t0 = time.perf_counter()
                ret, frame_bgr = cap.read()
                if not ret:
                    self.logMessage.emit("[INFO] 视频播放结束。")
                    break

                for oe in self._one_euro_filters.values():
                    oe.set_alpha(p["smooth_alpha"])
                    oe.set_one_euro_params(
                        min_cutoff=p["one_euro_min_cutoff"],
                        beta=p["one_euro_beta"],
                        d_cutoff=p["one_euro_d_cutoff"],
                    )
                self._init_butterworth(
                    p["butter_cutoff_hz"],
                    sample_hz=src_fps,
                    window_seconds=p["butter_window_seconds"],
                )
                self._set_filter_mode(str(p["filter_mode"]))
                if self._scale is not None:
                    self._scale.set_bow_length_cm(p["bow_length_cm"])
                    self._scale.set_threshold(p["confidence"])
                if self._scale_raw is not None:
                    self._scale_raw.set_bow_length_cm(p["bow_length_cm"])
                    self._scale_raw.set_threshold(p["confidence"])

                all_points: list[dict[str, Any]] = []
                all_points_raw: list[dict[str, Any]] = []
                archery_filtered_points: list[dict[str, Any]] = []
                archery_raw_points: list[dict[str, Any]] = []
                for tgt in self._enabled_targets():
                    annotator = self._annotators[tgt]
                    annotator.set_thresholds(p["confidence"], p["iou"])
                    result = annotator.predict_frame(frame_bgr)
                    if tgt == TARGET_ARCHERY:
                        archery_raw_points = [dict(x) for x in result.points if isinstance(x, dict)]
                    filtered = self._filter_points_for_target(tgt, result.points)
                    if tgt == TARGET_ARCHERY:
                        archery_filtered_points = [dict(x) for x in filtered]
                    all_points_raw.extend(self._decorate_points_for_display(tgt, result.points))
                    all_points.extend(self._decorate_points_for_display(tgt, filtered))

                cm_per_px_raw = self._scale_raw.update(archery_raw_points) if self._scale_raw is not None else None
                cm_per_px = self._scale.update(archery_filtered_points) if self._scale is not None else None
                raw_rows = self._build_rows(all_points_raw, cm_per_px_raw)
                rows = self._build_rows(all_points, cm_per_px)

                dt = max(time.perf_counter() - t0, 1e-6)
                infer_fps = 1.0 / dt

                dropped = 0
                frame_interval = 1.0 / max(src_fps, 1e-6)
                remain = frame_interval - dt
                if remain > 0:
                    time.sleep(remain)
                else:
                    lag = -remain
                    drop_n = min(8, int(lag / frame_interval))
                    for _ in range(drop_n):
                        if not cap.grab():
                            break
                        dropped += 1

                    now = time.time()
                    if dropped > 0 and (now - self._last_drop_log_ts) >= 1.0:
                        self.logMessage.emit(f"[WARN] 推理速度低于视频FPS，已丢帧: {dropped}")
                        self._last_drop_log_ts = now

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

                payload = {
                    "frame": frame_bgr,
                    "points": all_points,
                    "raw_rows": raw_rows,
                    "rows": rows,
                    "cm_per_px": cm_per_px,
                    "fps": infer_fps,
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "dropped": dropped,
                    "filter_mode": self._filter_mode,
                    "src_fps": src_fps,
                    "target_mode": self._target_mode,
                    "keypoint_names": list(self._keypoint_names),
                }
                self.frameReady.emit(payload)

        except Exception as exc:
            self.fatalError.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            self.finished.emit()

    def _build_rows(self, points: list[dict[str, Any]], cm_per_px: float | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, name in enumerate(self._keypoint_names):
            point = self._pick_point(points, idx, name)
            if point is None:
                rows.append(
                    {
                        "name": name,
                        "x_px": None,
                        "y_px": None,
                        "x_cm": None,
                        "y_cm": None,
                        "score": None,
                    }
                )
                continue

            x_px = float(point.get("x", 0.0))
            y_px = float(point.get("y", 0.0))
            score = float(point.get("score", 0.0))

            if cm_per_px is None:
                x_cm = None
                y_cm = None
            else:
                x_cm = round(x_px * cm_per_px, 2)
                y_cm = round(y_px * cm_per_px, 2)

            rows.append(
                {
                    "name": name,
                    "x_px": round(x_px, 2),
                    "y_px": round(y_px, 2),
                    "x_cm": x_cm,
                    "y_cm": y_cm,
                    "score": round(score, 4),
                }
            )
        return rows

    def _pick_point(self, points: list[dict[str, Any]], key_id: int, key_name: str) -> dict[str, Any] | None:
        for p in points:
            if not isinstance(p, dict):
                continue
            try:
                if int(p.get("id", -1)) == key_id:
                    return p
            except Exception:
                continue

        key_upper = key_name.strip().upper()
        for p in points:
            if not isinstance(p, dict):
                continue
            if str(p.get("name", "")).strip().upper() == key_upper:
                return p
        return None

    def _filter_points_for_target(self, target: str, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._filter_mode == FILTER_MODE_BUTTER:
            bf = self._butter_filters.get(target)
            if bf is not None:
                return bf.update(points)
        oe = self._one_euro_filters.get(target)
        if oe is None:
            return [dict(p) for p in points if isinstance(p, dict)]
        return oe.update(points)

    def _decorate_points_for_display(self, target: str, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        id_offset = 1000 if (self._target_mode == TARGET_BOTH and target == TARGET_HUMAN) else 0
        for p in points:
            if not isinstance(p, dict):
                continue
            one = dict(p)
            one["_target"] = str(target)
            try:
                one["id"] = int(one.get("id", -1)) + id_offset
            except Exception:
                pass
            out.append(one)
        return out


class InferencePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: InferenceWorker | None = None
        self._paused = False
        self._analysis_dialog: AnalysisDialog | None = None
        self._analysis_history: deque[dict[str, Any]] = deque(maxlen=600)
        self._active_keypoint_names: list[str] = list(KEYPOINT_NAMES)
        self._table_rows_meta: list[dict[str, Any]] = []
        self._session_raw_rows: list[dict[str, Any]] = []
        self._session_filtered_rows: list[dict[str, Any]] = []
        self._session_meta: dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        desc = QLabel(
            "应用模块 v1.0：视频实时推理（RTMO）+ One Euro/巴特沃斯滤波 + 弓长单位换算（px->cm）。"
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        top_box = QGroupBox("推理参数与控制")
        top_form = QFormLayout(top_box)

        video_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件")
        video_row.addWidget(self.video_path_edit, stretch=1)

        self.choose_video_btn = QPushButton("选择视频")
        self.choose_video_btn.clicked.connect(self._choose_video)
        video_row.addWidget(self.choose_video_btn)

        params_row = QHBoxLayout()

        self.bow_length_spin = QDoubleSpinBox()
        self.bow_length_spin.setRange(10.0, 500.0)
        self.bow_length_spin.setDecimals(2)
        self.bow_length_spin.setValue(177.8)
        self.bow_length_spin.setSuffix(" cm")
        self.bow_length_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("登记弓长"))
        params_row.addWidget(self.bow_length_spin)

        self.target_combo = QComboBox()
        self.target_combo.addItem("弓箭关键点（RTMO）", TARGET_ARCHERY)
        self.target_combo.addItem("人体关键点（RTMPose-HALPE26）", TARGET_HUMAN)
        self.target_combo.addItem("双模型（弓箭+人体）", TARGET_BOTH)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        params_row.addWidget(QLabel("推理目标"))
        params_row.addWidget(self.target_combo)

        self.filter_mode_combo = QComboBox()
        self.filter_mode_combo.addItem("One Euro（一欧元滤波）", FILTER_MODE_ONE_EURO)
        self.filter_mode_combo.addItem("双向四阶巴特沃斯", FILTER_MODE_BUTTER)
        self.filter_mode_combo.currentIndexChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("滤波方式"))
        params_row.addWidget(self.filter_mode_combo)

        self.ema_alpha_spin = QDoubleSpinBox()
        self.ema_alpha_spin.setRange(0.0, 1.0)
        self.ema_alpha_spin.setSingleStep(0.05)
        self.ema_alpha_spin.setValue(0.70)
        self.ema_alpha_spin.setToolTip("平滑强度语义：越大越跟手、越小越平滑。")
        self.ema_alpha_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("平滑"))
        params_row.addWidget(self.ema_alpha_spin)

        self.one_euro_min_cutoff_spin = QDoubleSpinBox()
        self.one_euro_min_cutoff_spin.setRange(0.01, 20.0)
        self.one_euro_min_cutoff_spin.setDecimals(2)
        self.one_euro_min_cutoff_spin.setSingleStep(0.1)
        self.one_euro_min_cutoff_spin.setValue(2.8)
        self.one_euro_min_cutoff_spin.setToolTip("One Euro 最小截止频率，越小越平滑。")
        self.one_euro_min_cutoff_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("min_cutoff"))
        params_row.addWidget(self.one_euro_min_cutoff_spin)

        self.one_euro_beta_spin = QDoubleSpinBox()
        self.one_euro_beta_spin.setRange(0.0, 30.0)
        self.one_euro_beta_spin.setDecimals(2)
        self.one_euro_beta_spin.setSingleStep(0.2)
        self.one_euro_beta_spin.setValue(4.2)
        self.one_euro_beta_spin.setToolTip("One Euro 速度响应系数，越大动态越跟手。")
        self.one_euro_beta_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("beta"))
        params_row.addWidget(self.one_euro_beta_spin)

        self.one_euro_d_cutoff_spin = QDoubleSpinBox()
        self.one_euro_d_cutoff_spin.setRange(0.01, 20.0)
        self.one_euro_d_cutoff_spin.setDecimals(2)
        self.one_euro_d_cutoff_spin.setSingleStep(0.1)
        self.one_euro_d_cutoff_spin.setValue(1.0)
        self.one_euro_d_cutoff_spin.setToolTip("One Euro 导数低通截止频率。")
        self.one_euro_d_cutoff_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("d_cutoff"))
        params_row.addWidget(self.one_euro_d_cutoff_spin)

        self.butter_cutoff_hz_spin = QDoubleSpinBox()
        self.butter_cutoff_hz_spin.setRange(0.05, 20.0)
        self.butter_cutoff_hz_spin.setDecimals(2)
        self.butter_cutoff_hz_spin.setSingleStep(0.1)
        self.butter_cutoff_hz_spin.setValue(3.0)
        self.butter_cutoff_hz_spin.setSuffix(" Hz")
        self.butter_cutoff_hz_spin.setToolTip("双向四阶巴特沃斯低通截止频率。")
        self.butter_cutoff_hz_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("巴特沃斯Hz"))
        params_row.addWidget(self.butter_cutoff_hz_spin)

        self.butter_window_seconds_spin = QDoubleSpinBox()
        self.butter_window_seconds_spin.setRange(0.3, 10.0)
        self.butter_window_seconds_spin.setDecimals(2)
        self.butter_window_seconds_spin.setSingleStep(0.1)
        self.butter_window_seconds_spin.setValue(2.0)
        self.butter_window_seconds_spin.setSuffix(" s")
        self.butter_window_seconds_spin.setToolTip("双向巴特沃斯滚动窗口长度（秒），越大越稳但边界效应更明显。")
        self.butter_window_seconds_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("巴特沃斯窗长"))
        params_row.addWidget(self.butter_window_seconds_spin)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        self.conf_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("置信度"))
        params_row.addWidget(self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.50)
        self.iou_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("IoU"))
        params_row.addWidget(self.iou_spin)

        params_row.addStretch(1)

        control_row = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        self.analysis_btn = QPushButton("参数")
        self.export_session_btn = QPushButton("导出推理CSV")

        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        self.analysis_btn.clicked.connect(self._open_analysis_dialog)
        self.export_session_btn.clicked.connect(self._export_inference_session_csv)

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.export_session_btn.setEnabled(False)

        control_row.addWidget(self.start_btn)
        control_row.addWidget(self.pause_btn)
        control_row.addWidget(self.stop_btn)
        control_row.addWidget(self.analysis_btn)
        control_row.addWidget(self.export_session_btn)

        self.status_label = QLabel("状态: 未开始")
        control_row.addWidget(self.status_label)
        self.live_fps_label = QLabel("推理FPS: -")
        self.video_fps_label = QLabel("视频FPS: -")
        control_row.addWidget(self.live_fps_label)
        control_row.addWidget(self.video_fps_label)
        control_row.addStretch(1)

        top_form.addRow("视频文件", video_row)
        top_form.addRow("参数", params_row)
        top_form.addRow("控制", control_row)
        root.addWidget(top_box)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.overlay = RealtimeOverlayWidget()
        left_layout.addWidget(self.overlay, stretch=1)
        bottom_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["关键点", "x_px", "y_px", "x_cm", "y_cm", "score"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._rebuild_table(self._active_keypoint_names)
        right_layout.addWidget(self.table, stretch=1)

        self.monitor = QTextEdit()
        self.monitor.setReadOnly(True)
        self.monitor.setPlaceholderText("推理日志")
        self.monitor.setMinimumHeight(120)
        right_layout.addWidget(self.monitor)

        bottom_splitter.addWidget(right_panel)
        bottom_splitter.setStretchFactor(0, 7)
        bottom_splitter.setStretchFactor(1, 3)
        bottom_splitter.setSizes([1120, 480])

        root.addWidget(bottom_splitter, stretch=1)
        self._refresh_filter_param_enable_state()
        self._refresh_target_ui_state()

    def _choose_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            str(BASE_DIR),
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv)",
        )
        if selected:
            self.video_path_edit.setText(selected)

    def _start(self) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "提示", "推理任务已在运行。")
            return

        video_text = self.video_path_edit.text().strip()
        if not video_text:
            QMessageBox.warning(self, "提示", "请先选择视频文件。")
            return

        video_path = Path(video_text)
        if not video_path.exists():
            QMessageBox.warning(self, "提示", f"视频文件不存在: {video_path}")
            return

        target_mode = self._current_target_mode()
        model_paths = self._target_model_paths(target_mode)
        missing = [f"{k}:{v}" for k, v in model_paths.items() if not Path(v).exists()]
        if missing:
            QMessageBox.warning(self, "提示", "以下模型文件不存在:\n" + "\n".join(missing))
            return

        self.monitor.clear()
        self.overlay.clear()
        self._analysis_history.clear()
        self._session_raw_rows.clear()
        self._session_filtered_rows.clear()
        self._active_keypoint_names = self._target_keypoint_names(target_mode)
        self._rebuild_table(self._active_keypoint_names)
        for k, v in model_paths.items():
            self._append_log(f"[INFO] 模型路径({k}): {v}")
        self._session_meta = {
            "video_path": str(video_path),
            "target_mode": target_mode,
            "model_paths": {k: str(v) for k, v in model_paths.items()},
            "confidence": float(self.conf_spin.value()),
            "iou": float(self.iou_spin.value()),
            "filter_mode": self._current_filter_mode(),
            "one_euro_min_cutoff": float(self.one_euro_min_cutoff_spin.value()),
            "one_euro_beta": float(self.one_euro_beta_spin.value()),
            "one_euro_d_cutoff": float(self.one_euro_d_cutoff_spin.value()),
            "butter_cutoff_hz": float(self.butter_cutoff_hz_spin.value()),
            "butter_window_seconds": float(self.butter_window_seconds_spin.value()),
            "bow_length_cm": float(self.bow_length_spin.value()),
        }
        set_inference_session_snapshot(raw_rows=[], filtered_rows=[], meta=dict(self._session_meta))

        worker = InferenceWorker(
            video_path=video_path,
            model_paths=model_paths,
            target_mode=target_mode,
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            smooth_alpha=float(self.ema_alpha_spin.value()),
            bow_length_cm=float(self.bow_length_spin.value()),
            filter_mode=self._current_filter_mode(),
            one_euro_min_cutoff=float(self.one_euro_min_cutoff_spin.value()),
            one_euro_beta=float(self.one_euro_beta_spin.value()),
            one_euro_d_cutoff=float(self.one_euro_d_cutoff_spin.value()),
            butter_cutoff_hz=float(self.butter_cutoff_hz_spin.value()),
            butter_window_seconds=float(self.butter_window_seconds_spin.value()),
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.frameReady.connect(self._on_frame_ready)
        worker.logMessage.connect(self._append_log)
        worker.fatalError.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._worker = worker
        self._thread = thread
        self._paused = False

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("暂停")
        self.status_label.setText("状态: 运行中")
        self.live_fps_label.setText("推理FPS: -")
        self.video_fps_label.setText("视频FPS: -")
        self.export_session_btn.setEnabled(False)

        thread.start()

    def _toggle_pause(self) -> None:
        if self._worker is None:
            return

        if self._paused:
            self._worker.resume()
            self._paused = False
            self.pause_btn.setText("暂停")
            self.status_label.setText("状态: 运行中")
            self._append_log("[INFO] 已恢复。")
        else:
            self._worker.pause()
            self._paused = True
            self.pause_btn.setText("继续")
            self.status_label.setText("状态: 暂停")
            self._append_log("[INFO] 已暂停。")

    def _stop(self) -> None:
        if self._worker is None:
            return
        self.status_label.setText("状态: 正在停止")
        self._worker.stop()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _current_target_mode(self) -> str:
        data = self.target_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return TARGET_ARCHERY

    def _target_model_paths(self, target_mode: str) -> dict[str, Path]:
        mode = str(target_mode)
        if mode == TARGET_HUMAN:
            return {TARGET_HUMAN: BASE_DIR / MODEL_FILES["human_halpe26_onnx"]}
        if mode == TARGET_BOTH:
            return {
                TARGET_ARCHERY: BASE_DIR / MODEL_FILES["archery_keypoints_onnx"],
                TARGET_HUMAN: BASE_DIR / MODEL_FILES["human_halpe26_onnx"],
            }
        return {TARGET_ARCHERY: BASE_DIR / MODEL_FILES["archery_keypoints_onnx"]}

    def _target_keypoint_names(self, target_mode: str) -> list[str]:
        mode = str(target_mode)
        if mode == TARGET_HUMAN:
            return list(HUMAN_KEYPOINT_NAMES)
        if mode == TARGET_BOTH:
            return list(KEYPOINT_NAMES) + list(HUMAN_KEYPOINT_NAMES)
        return list(KEYPOINT_NAMES)

    def _refresh_target_ui_state(self) -> None:
        is_archery = self._current_target_mode() in (TARGET_ARCHERY, TARGET_BOTH)
        self.bow_length_spin.setEnabled(is_archery)
        self.bow_length_spin.setToolTip("" if is_archery else "人体关键点模式下不进行弓长像素-厘米换算。")

    def _on_target_changed(self) -> None:
        if self._worker is not None:
            self._append_log("[INFO] 推理运行中，推理目标切换将在下次开始时生效。")
        self._refresh_target_ui_state()
        self._active_keypoint_names = self._target_keypoint_names(self._current_target_mode())
        self._rebuild_table(self._active_keypoint_names)
        if self._analysis_dialog is not None:
            self._analysis_dialog.close()

    def _rebuild_table(self, keypoint_names: list[str]) -> None:
        names = [str(x) for x in keypoint_names]
        mode = self._current_target_mode()
        rows_meta: list[dict[str, Any]] = []
        if mode == TARGET_BOTH:
            rows_meta.append({"kind": "header", "title": "弓箭关键点"})
            for name in KEYPOINT_NAMES:
                if name in names:
                    rows_meta.append({"kind": "kp", "name": str(name)})
            rows_meta.append({"kind": "header", "title": "人体关键点"})
            for name in HUMAN_KEYPOINT_NAMES:
                if name in names:
                    rows_meta.append({"kind": "kp", "name": str(name)})
        else:
            rows_meta = [{"kind": "kp", "name": str(name)} for name in names]

        self._table_rows_meta = rows_meta
        self.table.setRowCount(len(rows_meta))
        for r, meta in enumerate(rows_meta):
            item0 = self.table.item(r, 0)
            if item0 is None:
                item0 = QTableWidgetItem()
                self.table.setItem(r, 0, item0)
            if meta.get("kind") == "header":
                item0.setText(f"—— {meta.get('title', '')} ——")
                item0.setBackground(QColor(236, 240, 245))
            else:
                item0.setText(str(meta.get("name", "")))
                item0.setBackground(QColor(255, 255, 255))
            for c in range(1, 6):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)
                item.setText("" if meta.get("kind") == "header" else "-")
                item.setBackground(QColor(236, 240, 245) if meta.get("kind") == "header" else QColor(255, 255, 255))

    def _current_filter_mode(self) -> str:
        data = self.filter_mode_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return FILTER_MODE_ONE_EURO

    def _refresh_filter_param_enable_state(self) -> None:
        is_one_euro = self._current_filter_mode() == FILTER_MODE_ONE_EURO
        self.one_euro_min_cutoff_spin.setEnabled(is_one_euro)
        self.one_euro_beta_spin.setEnabled(is_one_euro)
        self.one_euro_d_cutoff_spin.setEnabled(is_one_euro)
        self.butter_cutoff_hz_spin.setEnabled(not is_one_euro)
        self.butter_window_seconds_spin.setEnabled(not is_one_euro)

    def _on_runtime_params_changed(self) -> None:
        self._refresh_filter_param_enable_state()
        self._refresh_target_ui_state()
        if self._worker is None:
            return
        self._worker.update_params(
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            smooth_alpha=float(self.ema_alpha_spin.value()),
            bow_length_cm=float(self.bow_length_spin.value()),
            filter_mode=self._current_filter_mode(),
            one_euro_min_cutoff=float(self.one_euro_min_cutoff_spin.value()),
            one_euro_beta=float(self.one_euro_beta_spin.value()),
            one_euro_d_cutoff=float(self.one_euro_d_cutoff_spin.value()),
            butter_cutoff_hz=float(self.butter_cutoff_hz_spin.value()),
            butter_window_seconds=float(self.butter_window_seconds_spin.value()),
        )

    def _open_analysis_dialog(self) -> None:
        if self._analysis_dialog is None or self._analysis_dialog.keypoint_names() != self._active_keypoint_names:
            if self._analysis_dialog is not None:
                self._analysis_dialog.close()
            self._analysis_dialog = AnalysisDialog(self._active_keypoint_names, self)
            self._analysis_dialog.finished.connect(self._on_analysis_dialog_closed)
        self._analysis_dialog.set_history(list(self._analysis_history))
        self._analysis_dialog.show()
        self._analysis_dialog.raise_()
        self._analysis_dialog.activateWindow()

    def _on_analysis_dialog_closed(self, *_args: Any) -> None:
        self._analysis_dialog = None

    def _export_inference_session_csv(self) -> None:
        if not self._session_raw_rows and not self._session_filtered_rows:
            QMessageBox.information(self, "提示", "当前没有可导出的推理时序数据。请先完成一次推理。")
            return

        default_root = BASE_DIR / "output" / "inference_sessions"
        default_root.mkdir(parents=True, exist_ok=True)
        selected = QFileDialog.getExistingDirectory(self, "选择推理时序导出目录", str(default_root))
        if not selected:
            return

        try:
            result = export_inference_session_timeseries(
                export_root=Path(selected),
                session_meta=dict(self._session_meta),
                raw_rows=list(self._session_raw_rows),
                filtered_rows=list(self._session_filtered_rows),
            )
        except Exception as exc:
            self._append_log(f"[ERROR] 推理时序导出失败: {exc}")
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        self._append_log(
            "[INFO] 推理时序导出完成: "
            f"{result.output_dir} | raw_rows={result.raw_rows}, filtered_rows={result.filtered_rows}"
        )
        self._append_log(f"[FILE] {result.raw_csv}")
        self._append_log(f"[FILE] {result.filtered_csv}")
        self._append_log(f"[FILE] {result.meta_json}")
        QMessageBox.information(self, "导出完成", f"推理时序导出成功：\n{result.output_dir}")

    def _push_analysis_snapshot(
        self,
        *,
        rows: list[dict[str, Any]],
        frame_idx: int,
        cm_per_px: float | None,
        src_fps: float | None,
    ) -> None:
        row_map = {}
        for one in rows:
            if not isinstance(one, dict):
                continue
            key = str(one.get("name", "")).strip().upper()
            if key:
                row_map[key] = dict(one)
        self._analysis_history.append(
            {
                "ts": time.perf_counter(),
                "frame_idx": int(frame_idx),
                "cm_per_px": None if cm_per_px is None else float(cm_per_px),
                "src_fps": None if src_fps is None else float(src_fps),
                "row_map": row_map,
            }
        )
        if self._analysis_dialog is not None:
            self._analysis_dialog.set_history(list(self._analysis_history))

    def _append_session_timeseries_rows(
        self,
        *,
        raw_rows: list[dict[str, Any]],
        filtered_rows: list[dict[str, Any]],
        frame_idx: int,
        src_fps: float,
        wall_time_s: float,
    ) -> None:
        video_time_s = (float(frame_idx) / float(src_fps)) if src_fps > 1e-6 else None
        self._session_raw_rows.extend(
            self._flatten_timeseries_rows(
                rows=raw_rows,
                frame_idx=frame_idx,
                src_fps=src_fps,
                wall_time_s=wall_time_s,
                video_time_s=video_time_s,
                display_filter="raw",
            )
        )
        self._session_filtered_rows.extend(
            self._flatten_timeseries_rows(
                rows=filtered_rows,
                frame_idx=frame_idx,
                src_fps=src_fps,
                wall_time_s=wall_time_s,
                video_time_s=video_time_s,
                display_filter="filtered",
            )
        )

    def _flatten_timeseries_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        frame_idx: int,
        src_fps: float,
        wall_time_s: float,
        video_time_s: float | None,
        display_filter: str,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            target = self._infer_target_from_name(name)
            out.append(
                {
                    "frame_idx": int(frame_idx),
                    "video_time_s": None if video_time_s is None else round(float(video_time_s), 6),
                    "wall_time_s": round(float(wall_time_s), 6),
                    "target": target,
                    "keypoint_name": name,
                    "x_px": row.get("x_px"),
                    "y_px": row.get("y_px"),
                    "x_cm": row.get("x_cm"),
                    "y_cm": row.get("y_cm"),
                    "score": row.get("score"),
                    "src_fps": round(float(src_fps), 6) if src_fps > 0 else None,
                    "display_filter": display_filter,
                    "row_kind": "keypoint",
                }
            )
        return out

    def _infer_target_from_name(self, keypoint_name: str) -> str:
        name = str(keypoint_name).strip()
        if name in KEYPOINT_NAMES:
            return TARGET_ARCHERY
        if name in HUMAN_KEYPOINT_NAMES:
            return TARGET_HUMAN
        return "unknown"

    def _on_frame_ready(self, payload: dict[str, Any]) -> None:
        wall_now = time.perf_counter()
        frame = payload.get("frame")
        points = payload.get("points", [])
        raw_rows = payload.get("raw_rows", [])
        rows = payload.get("rows", [])
        cm_per_px = payload.get("cm_per_px")
        infer_fps = float(payload.get("fps", 0.0))
        frame_idx = int(payload.get("frame_idx", 0))
        total = int(payload.get("total_frames", 0))
        dropped = int(payload.get("dropped", 0))
        filter_mode = str(payload.get("filter_mode", self._current_filter_mode()))
        src_fps = float(payload.get("src_fps", 0.0) or 0.0)
        target_mode = str(payload.get("target_mode", self._current_target_mode()))
        payload_kps = payload.get("keypoint_names")
        if isinstance(payload_kps, list):
            names = [str(x) for x in payload_kps]
            if names and names != self._active_keypoint_names:
                self._active_keypoint_names = names
                self._rebuild_table(self._active_keypoint_names)

        if frame is not None:
            self.overlay.set_frame(frame, points)

        scale_text = "-" if cm_per_px is None else f"{float(cm_per_px):.4f}cm"
        filter_name = "One Euro" if filter_mode == FILTER_MODE_ONE_EURO else "Butterworth"
        if target_mode == TARGET_HUMAN:
            target_name = "人体"
        elif target_mode == TARGET_BOTH:
            target_name = "双模型"
        else:
            target_name = "弓箭"
        status = (
            f"目标:{target_name} | FPS:{infer_fps:.1f} | 1px={scale_text} | 滤波:{filter_name} | "
            f"进度:{frame_idx}/{max(total, 0)} | 丢帧:{dropped}"
        )
        self.overlay.set_overlay_status(status)

        self.status_label.setText(f"状态: 运行中 | {status}")
        self.live_fps_label.setText(f"推理FPS: {infer_fps:.1f}")
        self.video_fps_label.setText(f"视频FPS: {src_fps:.2f}" if src_fps > 0 else "视频FPS: -")
        self._update_table(rows)
        self._push_analysis_snapshot(rows=rows, frame_idx=frame_idx, cm_per_px=cm_per_px, src_fps=src_fps)
        self._append_session_timeseries_rows(
            raw_rows=raw_rows if isinstance(raw_rows, list) else [],
            filtered_rows=rows if isinstance(rows, list) else [],
            frame_idx=frame_idx,
            src_fps=src_fps,
            wall_time_s=wall_now,
        )

    def _update_table(self, rows: list[dict[str, Any]]) -> None:
        row_map = {str(one.get("name", "")).strip().upper(): one for one in rows if isinstance(one, dict)}

        for r, meta in enumerate(self._table_rows_meta):
            if meta.get("kind") == "header":
                for c in range(1, 6):
                    item = self.table.item(r, c)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(r, c, item)
                    item.setText("")
                continue
            name = str(meta.get("name", ""))
            one = row_map.get(name.strip().upper())
            if one is None:
                vals = ["-", "-", "-", "-", "-"]
            else:
                vals = [
                    self._fmt_num(one.get("x_px"), 2),
                    self._fmt_num(one.get("y_px"), 2),
                    self._fmt_num(one.get("x_cm"), 2),
                    self._fmt_num(one.get("y_cm"), 2),
                    self._fmt_num(one.get("score"), 4),
                ]

            for c in range(1, 6):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)
                item.setText(vals[c - 1])

    def _fmt_num(self, value: Any, digits: int) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def _on_worker_error(self, message: str) -> None:
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "推理失败", message)

    def _on_worker_finished(self) -> None:
        self._append_log("[INFO] 推理任务结束。")
        self.status_label.setText("状态: 已结束")
        self.live_fps_label.setText("推理FPS: -")
        self.video_fps_label.setText("视频FPS: -")
        self.export_session_btn.setEnabled(bool(self._session_raw_rows or self._session_filtered_rows))
        if self._session_raw_rows or self._session_filtered_rows:
            set_inference_session_snapshot(
                raw_rows=list(self._session_raw_rows),
                filtered_rows=list(self._session_filtered_rows),
                meta=dict(self._session_meta),
            )
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("暂停")

        self._worker = None
        self._thread = None
        self._paused = False

    def _append_log(self, text: str) -> None:
        self.monitor.append(text)


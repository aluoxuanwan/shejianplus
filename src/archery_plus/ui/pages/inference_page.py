from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
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

from archery_plus.config import BASE_DIR, MODEL_FILES
from archery_plus.core.ema_filter import EmaPointFilter
from archery_plus.core.keypoint_schema import DEFAULT_ARCHERY_KEYPOINT_NAMES
from archery_plus.core.scale_converter import BowScaleConverter
from archery_plus.pipelines.auto_annotator_rtmo_archery import RtmoArcheryAutoAnnotator
from archery_plus.ui.widgets.realtime_overlay_widget import RealtimeOverlayWidget

KEYPOINT_NAMES = list(DEFAULT_ARCHERY_KEYPOINT_NAMES)


class InferenceWorker(QObject):
    frameReady = Signal(object)
    logMessage = Signal(str)
    fatalError = Signal(str)
    finished = Signal()

    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        confidence: float,
        iou: float,
        ema_alpha: float,
        bow_length_cm: float,
    ) -> None:
        super().__init__()
        self._video_path = Path(video_path)
        self._annotator = RtmoArcheryAutoAnnotator(model_path=model_path)
        self._ema = EmaPointFilter(alpha=ema_alpha)
        self._scale = BowScaleConverter(bow_length_cm=bow_length_cm, scale_conf_threshold=confidence)

        self._lock = threading.Lock()
        self._running = True
        self._paused = False
        self._params = {
            "confidence": float(confidence),
            "iou": float(iou),
            "ema_alpha": float(ema_alpha),
            "bow_length_cm": float(bow_length_cm),
        }
        self._last_drop_log_ts = 0.0

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def update_params(self, confidence: float, iou: float, ema_alpha: float, bow_length_cm: float) -> None:
        with self._lock:
            self._params["confidence"] = float(confidence)
            self._params["iou"] = float(iou)
            self._params["ema_alpha"] = float(ema_alpha)
            self._params["bow_length_cm"] = float(bow_length_cm)

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = cv2.VideoCapture(str(self._video_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self._video_path}")

            if not self._annotator.is_ready():
                raise RuntimeError("RTMO ONNX 模型不可用，请检查模型文件。")

            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps <= 1e-6:
                src_fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

                self._annotator.set_thresholds(p["confidence"], p["iou"])
                self._ema.set_alpha(p["ema_alpha"])
                self._scale.set_bow_length_cm(p["bow_length_cm"])
                self._scale.set_threshold(p["confidence"])

                result = self._annotator.predict_frame(frame_bgr)
                filtered_points = self._ema.update(result.points)
                cm_per_px = self._scale.update(filtered_points)
                rows = self._build_rows(filtered_points, cm_per_px)

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
                    "points": filtered_points,
                    "rows": rows,
                    "cm_per_px": cm_per_px,
                    "fps": infer_fps,
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "dropped": dropped,
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
        for idx, name in enumerate(KEYPOINT_NAMES):
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


class InferencePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: InferenceWorker | None = None
        self._paused = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        desc = QLabel("应用模块 v1.0：视频实时推理（RTMO）+ EMA平滑 + 弓长单位换算（px->cm）。")
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

        self.ema_alpha_spin = QDoubleSpinBox()
        self.ema_alpha_spin.setRange(0.0, 1.0)
        self.ema_alpha_spin.setSingleStep(0.05)
        self.ema_alpha_spin.setValue(0.70)
        self.ema_alpha_spin.valueChanged.connect(self._on_runtime_params_changed)
        params_row.addWidget(QLabel("EMA"))
        params_row.addWidget(self.ema_alpha_spin)

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

        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        control_row.addWidget(self.start_btn)
        control_row.addWidget(self.pause_btn)
        control_row.addWidget(self.stop_btn)

        self.status_label = QLabel("状态: 未开始")
        control_row.addWidget(self.status_label)
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

        self.table = QTableWidget(len(KEYPOINT_NAMES), 6)
        self.table.setHorizontalHeaderLabels(["关键点", "x_px", "y_px", "x_cm", "y_cm", "score"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for r, name in enumerate(KEYPOINT_NAMES):
            self.table.setItem(r, 0, QTableWidgetItem(name))
            for c in range(1, 6):
                self.table.setItem(r, c, QTableWidgetItem("-"))
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

        model_path = BASE_DIR / MODEL_FILES["archery_keypoints_onnx"]
        if not model_path.exists():
            QMessageBox.warning(self, "提示", f"RTMO ONNX 不存在: {model_path}")
            return

        self.monitor.clear()
        self.overlay.clear()
        self._append_log(f"[INFO] 模型路径: {model_path}")

        worker = InferenceWorker(
            video_path=video_path,
            model_path=model_path,
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            ema_alpha=float(self.ema_alpha_spin.value()),
            bow_length_cm=float(self.bow_length_spin.value()),
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

    def _on_runtime_params_changed(self) -> None:
        if self._worker is None:
            return
        self._worker.update_params(
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            ema_alpha=float(self.ema_alpha_spin.value()),
            bow_length_cm=float(self.bow_length_spin.value()),
        )

    def _on_frame_ready(self, payload: dict[str, Any]) -> None:
        frame = payload.get("frame")
        points = payload.get("points", [])
        rows = payload.get("rows", [])
        cm_per_px = payload.get("cm_per_px")
        infer_fps = float(payload.get("fps", 0.0))
        frame_idx = int(payload.get("frame_idx", 0))
        total = int(payload.get("total_frames", 0))
        dropped = int(payload.get("dropped", 0))

        if frame is not None:
            self.overlay.set_frame(frame, points)

        scale_text = "-" if cm_per_px is None else f"{float(cm_per_px):.4f}cm"
        status = f"FPS:{infer_fps:.1f} | 1px={scale_text} | 进度:{frame_idx}/{max(total, 0)} | 丢帧:{dropped}"
        self.overlay.set_overlay_status(status)

        self.status_label.setText(f"状态: 运行中 | {status}")
        self._update_table(rows)

    def _update_table(self, rows: list[dict[str, Any]]) -> None:
        row_map = {str(one.get("name", "")).strip().upper(): one for one in rows if isinstance(one, dict)}

        for r, name in enumerate(KEYPOINT_NAMES):
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
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("暂停")

        self._worker = None
        self._thread = None
        self._paused = False

    def _append_log(self, text: str) -> None:
        self.monitor.append(text)


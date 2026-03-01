from __future__ import annotations

import math
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SPEED_TIMEBASE_VIDEO = "video_fps"
SPEED_TIMEBASE_REALTIME = "realtime"


def _pt_from_row(row: dict[str, Any] | None, *, use_cm: bool = False) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    x_key = "x_cm" if use_cm else "x_px"
    y_key = "y_cm" if use_cm else "y_px"
    try:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            return None
        return float(x), float(y)
    except Exception:
        return None


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    x1, y1 = v1
    x2, y2 = v2
    n1 = (x1 * x1 + y1 * y1) ** 0.5
    n2 = (x2 * x2 + y2 * y2) ** 0.5
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None
    cosine = (x1 * x2 + y1 * y2) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


class AnalysisDialog(QDialog):
    def __init__(self, keypoint_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("参数")
        self.resize(760, 640)
        self._history: list[dict[str, Any]] = []
        self._keypoint_names = [str(x) for x in keypoint_names]

        root = QVBoxLayout(self)
        tips = QLabel(
            "分析功能：瞬时速度(cm/s)、关键点轨迹、两点与上下左右夹角、三点夹角、四点夹角。"
            "速度默认按视频FPS与帧号计算，更适合动作分析。"
        )
        tips.setWordWrap(True)
        root.addWidget(tips)

        form = QFormLayout()

        self.speed_kp_combo = QComboBox()
        self.speed_timebase_combo = QComboBox()
        self.speed_timebase_combo.addItem("视频时间（FPS/帧号）", SPEED_TIMEBASE_VIDEO)
        self.speed_timebase_combo.addItem("实时时间（系统时钟）", SPEED_TIMEBASE_REALTIME)
        self.speed_window_spin = QSpinBox()
        self.speed_window_spin.setRange(1, 30)
        self.speed_window_spin.setSingleStep(2)
        self.speed_window_spin.setValue(3)
        self.speed_label = QLabel("-")
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("关键点"))
        speed_row.addWidget(self.speed_kp_combo)
        speed_row.addSpacing(8)
        speed_row.addWidget(QLabel("时间基准"))
        speed_row.addWidget(self.speed_timebase_combo)
        speed_row.addSpacing(8)
        speed_row.addWidget(QLabel("窗口帧数"))
        speed_row.addWidget(self.speed_window_spin)
        speed_row.addSpacing(8)
        speed_row.addWidget(QLabel("瞬时速度"))
        speed_row.addWidget(self.speed_label)
        speed_row.addStretch(1)
        speed_wrap = QWidget()
        speed_wrap.setLayout(speed_row)
        form.addRow("速度", speed_wrap)

        self.traj_kp_combo = QComboBox()
        self.traj_len_spin = QSpinBox()
        self.traj_len_spin.setRange(5, 300)
        self.traj_len_spin.setValue(30)
        traj_row = QHBoxLayout()
        traj_row.addWidget(QLabel("关键点"))
        traj_row.addWidget(self.traj_kp_combo)
        traj_row.addSpacing(8)
        traj_row.addWidget(QLabel("轨迹回看帧数"))
        traj_row.addWidget(self.traj_len_spin)
        traj_row.addStretch(1)
        traj_wrap = QWidget()
        traj_wrap.setLayout(traj_row)
        form.addRow("轨迹", traj_wrap)

        self.a2_p1_combo = QComboBox()
        self.a2_p2_combo = QComboBox()
        self.a2_axis_combo = QComboBox()
        self.a2_axis_combo.addItems(["上", "下", "左", "右"])
        self.a2_result = QLabel("-")
        a2_row = QHBoxLayout()
        a2_row.addWidget(self.a2_p1_combo)
        a2_row.addWidget(QLabel("→"))
        a2_row.addWidget(self.a2_p2_combo)
        a2_row.addWidget(QLabel("相对"))
        a2_row.addWidget(self.a2_axis_combo)
        a2_row.addWidget(QLabel("夹角"))
        a2_row.addWidget(self.a2_result)
        a2_row.addStretch(1)
        a2_wrap = QWidget()
        a2_wrap.setLayout(a2_row)
        form.addRow("两点角度", a2_wrap)

        self.a3_p1_combo = QComboBox()
        self.a3_p2_combo = QComboBox()
        self.a3_p3_combo = QComboBox()
        self.a3_result = QLabel("-")
        a3_row = QHBoxLayout()
        a3_row.addWidget(self.a3_p1_combo)
        a3_row.addWidget(QLabel("-"))
        a3_row.addWidget(self.a3_p2_combo)
        a3_row.addWidget(QLabel("-"))
        a3_row.addWidget(self.a3_p3_combo)
        a3_row.addWidget(QLabel("夹角"))
        a3_row.addWidget(self.a3_result)
        a3_row.addStretch(1)
        a3_wrap = QWidget()
        a3_wrap.setLayout(a3_row)
        form.addRow("三点角度", a3_wrap)

        self.a4_p1_combo = QComboBox()
        self.a4_p2_combo = QComboBox()
        self.a4_p3_combo = QComboBox()
        self.a4_p4_combo = QComboBox()
        self.a4_result = QLabel("-")
        a4_row = QHBoxLayout()
        a4_row.addWidget(self.a4_p1_combo)
        a4_row.addWidget(QLabel("→"))
        a4_row.addWidget(self.a4_p2_combo)
        a4_row.addWidget(QLabel("与"))
        a4_row.addWidget(self.a4_p3_combo)
        a4_row.addWidget(QLabel("→"))
        a4_row.addWidget(self.a4_p4_combo)
        a4_row.addWidget(QLabel("夹角"))
        a4_row.addWidget(self.a4_result)
        a4_row.addStretch(1)
        a4_wrap = QWidget()
        a4_wrap.setLayout(a4_row)
        form.addRow("四点角度", a4_wrap)

        form_host = QWidget()
        form_host.setLayout(form)
        root.addWidget(form_host)

        self.traj_view = QTextEdit()
        self.traj_view.setReadOnly(True)
        self.traj_view.setPlaceholderText("请先开始推理，轨迹和计算结果会实时刷新。")
        root.addWidget(self.traj_view, stretch=1)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.close_btn = QPushButton("关闭")
        self.refresh_btn.clicked.connect(self.refresh)
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self._populate_keypoint_combos()

        for combo in (
            self.speed_kp_combo,
            self.speed_timebase_combo,
            self.traj_kp_combo,
            self.a2_p1_combo,
            self.a2_p2_combo,
            self.a2_axis_combo,
            self.a3_p1_combo,
            self.a3_p2_combo,
            self.a3_p3_combo,
            self.a4_p1_combo,
            self.a4_p2_combo,
            self.a4_p3_combo,
            self.a4_p4_combo,
        ):
            combo.currentIndexChanged.connect(self.refresh)
        self.speed_window_spin.valueChanged.connect(self.refresh)
        self.traj_len_spin.valueChanged.connect(self.refresh)

    def keypoint_names(self) -> list[str]:
        return list(self._keypoint_names)

    def _populate_keypoint_combos(self) -> None:
        combos = [
            self.speed_kp_combo,
            self.traj_kp_combo,
            self.a2_p1_combo,
            self.a2_p2_combo,
            self.a3_p1_combo,
            self.a3_p2_combo,
            self.a3_p3_combo,
            self.a4_p1_combo,
            self.a4_p2_combo,
            self.a4_p3_combo,
            self.a4_p4_combo,
        ]
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._keypoint_names)
            combo.blockSignals(False)

    def set_history(self, history: list[dict[str, Any]]) -> None:
        self._history = history
        self.refresh()

    def refresh(self) -> None:
        history = [h for h in self._history if isinstance(h, dict)]
        if not history:
            self.speed_label.setText("-")
            self.a2_result.setText("-")
            self.a3_result.setText("-")
            self.a4_result.setText("-")
            self.traj_view.setPlainText("暂无数据。")
            return

        latest = history[-1]
        row_map = latest.get("row_map", {})
        if not isinstance(row_map, dict):
            row_map = {}

        self._refresh_speed(history)
        self._refresh_angles(row_map)
        self._refresh_trajectory(history)

    def _refresh_speed(self, history: list[dict[str, Any]]) -> None:
        name = self.speed_kp_combo.currentText().strip().upper()
        use_video_time = self.speed_timebase_combo.currentData() == SPEED_TIMEBASE_VIDEO
        window_frames = max(1, int(self.speed_window_spin.value()))
        samples: list[dict[str, Any]] = []
        for frame in reversed(history):
            row_map = frame.get("row_map", {})
            if not isinstance(row_map, dict):
                continue
            row = row_map.get(name)
            pt_cm = _pt_from_row(row, use_cm=True)
            pt_px = _pt_from_row(row, use_cm=False)
            try:
                ts = float(frame.get("ts", 0.0))
            except Exception:
                ts = 0.0
            try:
                frame_idx = int(frame.get("frame_idx", -1))
            except Exception:
                frame_idx = -1
            try:
                src_fps = float(frame.get("src_fps", 0.0))
            except Exception:
                src_fps = 0.0
            if pt_cm is None and pt_px is None:
                continue
            samples.append(
                {
                    "ts": ts,
                    "pt_cm": pt_cm,
                    "pt_px": pt_px,
                    "frame_idx": frame_idx,
                    "src_fps": src_fps,
                }
            )
            if len(samples) >= (window_frames + 1):
                break

        if len(samples) < 2:
            self.speed_label.setText("-")
            return

        latest = samples[0]
        prev = samples[min(window_frames, len(samples) - 1)]
        use_cm = latest.get("pt_cm") is not None and prev.get("pt_cm") is not None
        p2 = latest["pt_cm"] if use_cm else latest["pt_px"]
        p1 = prev["pt_cm"] if use_cm else prev["pt_px"]
        if p2 is None or p1 is None:
            self.speed_label.setText("-")
            return

        if use_video_time:
            fps = float(latest.get("src_fps", 0.0) or prev.get("src_fps", 0.0) or 0.0)
            fi2 = int(latest.get("frame_idx", -1))
            fi1 = int(prev.get("frame_idx", -1))
            if fps <= 1e-6 or fi2 < 0 or fi1 < 0:
                self.speed_label.setText("-")
                return
            dt = abs(fi2 - fi1) / fps
        else:
            t2 = float(latest.get("ts", 0.0))
            t1 = float(prev.get("ts", 0.0))
            dt = t2 - t1

        if dt <= 1e-6:
            self.speed_label.setText("-")
            return
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        speed = ((dx * dx + dy * dy) ** 0.5) / dt
        unit = "cm/s" if use_cm else "px/s"
        self.speed_label.setText(f"{speed:.2f} {unit}")

    def _refresh_angles(self, row_map: dict[str, Any]) -> None:
        axis_map = {"上": (0.0, -1.0), "下": (0.0, 1.0), "左": (-1.0, 0.0), "右": (1.0, 0.0)}

        p1 = _pt_from_row(row_map.get(self.a2_p1_combo.currentText().strip().upper()))
        p2 = _pt_from_row(row_map.get(self.a2_p2_combo.currentText().strip().upper()))
        if p1 is None or p2 is None:
            self.a2_result.setText("-")
        else:
            v = (p2[0] - p1[0], p2[1] - p1[1])
            ang = _angle_between(v, axis_map.get(self.a2_axis_combo.currentText(), (0.0, -1.0)))
            self.a2_result.setText("-" if ang is None else f"{ang:.2f}°")

        a = _pt_from_row(row_map.get(self.a3_p1_combo.currentText().strip().upper()))
        b = _pt_from_row(row_map.get(self.a3_p2_combo.currentText().strip().upper()))
        c = _pt_from_row(row_map.get(self.a3_p3_combo.currentText().strip().upper()))
        if a is None or b is None or c is None:
            self.a3_result.setText("-")
        else:
            ang = _angle_between((a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1]))
            self.a3_result.setText("-" if ang is None else f"{ang:.2f}°")

        p41 = _pt_from_row(row_map.get(self.a4_p1_combo.currentText().strip().upper()))
        p42 = _pt_from_row(row_map.get(self.a4_p2_combo.currentText().strip().upper()))
        p43 = _pt_from_row(row_map.get(self.a4_p3_combo.currentText().strip().upper()))
        p44 = _pt_from_row(row_map.get(self.a4_p4_combo.currentText().strip().upper()))
        if p41 is None or p42 is None or p43 is None or p44 is None:
            self.a4_result.setText("-")
        else:
            ang = _angle_between((p42[0] - p41[0], p42[1] - p41[1]), (p44[0] - p43[0], p44[1] - p43[1]))
            self.a4_result.setText("-" if ang is None else f"{ang:.2f}°")

    def _refresh_trajectory(self, history: list[dict[str, Any]]) -> None:
        name = self.traj_kp_combo.currentText().strip().upper()
        limit = int(self.traj_len_spin.value())
        lines = [f"关键点轨迹：{name}（最近{limit}帧）"]
        shown = 0
        for frame in reversed(history):
            row_map = frame.get("row_map", {})
            if not isinstance(row_map, dict):
                continue
            row = row_map.get(name)
            if not isinstance(row, dict):
                continue
            if row.get("x_px") is None or row.get("y_px") is None:
                continue
            frame_idx = frame.get("frame_idx", "-")
            lines.append(
                f"帧{frame_idx}: "
                f"px=({self._fmt(row.get('x_px'))},{self._fmt(row.get('y_px'))}) "
                f"cm=({self._fmt(row.get('x_cm'))},{self._fmt(row.get('y_cm'))}) "
                f"score={self._fmt(row.get('score'), 4)}"
            )
            shown += 1
            if shown >= limit:
                break
        if shown == 0:
            lines.append("暂无有效轨迹点。")
        self.traj_view.setPlainText("\n".join(lines))

    def _fmt(self, value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

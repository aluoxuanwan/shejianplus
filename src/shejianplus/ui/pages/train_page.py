
from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)

from shejianplus.config import DEFAULT_PROJECT_DIR
from shejianplus.core.keypoint_schema import (
    TARGET_ARCHERY,
    TARGET_HUMAN,
    ensure_project_keypoint_sets,
    get_active_keypoint_set,
    import_keypoint_set_for_target,
)
from shejianplus.core.mmpose_training import MMPoseDatasetBuildResult, build_mmpose_coco_dataset
from shejianplus.core.training_config import (
    GeneratedTrainingConfig,
    INIT_MODE_OPTIONS,
    OPTIMIZER_OPTIONS,
    REQUIRED_MMPOSE_VERSION,
    SCHEDULER_OPTIONS,
    TRAIN_TARGET_OPTIONS,
    TRAIN_TASK_OPTIONS,
    TrainingPreset,
    default_training_preset,
    generate_training_config,
    is_required_mmpose_version,
    list_model_names,
    load_training_preset,
    resolve_model_assets,
    save_training_preset,
)


class LineChartWidget(QWidget):
    def __init__(self, title: str, color: QColor) -> None:
        super().__init__()
        self._title = title
        self._color = color
        self._points: list[tuple[float, float]] = []
        self.setMinimumHeight(220)

    def clear_data(self) -> None:
        self._points.clear()
        self.update()

    def add_point(self, x: float, y: float) -> None:
        self._points.append((float(x), float(y)))
        if len(self._points) > 2000:
            self._points = self._points[-2000:]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        p.fillRect(rect, QColor(18, 22, 28))

        margin_l = 46
        margin_r = 14
        margin_t = 26
        margin_b = 28
        plot = rect.adjusted(margin_l, margin_t, -margin_r, -margin_b)

        p.setPen(QColor(210, 215, 223))
        p.drawText(10, 18, self._title)

        p.setPen(QPen(QColor(72, 78, 90), 1.0))
        p.drawRect(plot)

        if len(self._points) < 2:
            p.setPen(QColor(135, 142, 152))
            p.drawText(plot, Qt.AlignCenter, "等待训练日志...")
            return

        xs = [pt[0] for pt in self._points]
        ys = [pt[1] for pt in self._points]

        x_min = min(xs)
        x_max = max(xs)
        if x_max <= x_min:
            x_max = x_min + 1.0

        y_min = min(ys)
        y_max = max(ys)
        if y_max <= y_min:
            y_max = y_min + 1.0

        y_pad = (y_max - y_min) * 0.1
        y_min -= y_pad
        y_max += y_pad

        p.setPen(QColor(160, 166, 176))
        p.drawText(plot.left() - 38, plot.top() + 6, f"{y_max:.4f}")
        p.drawText(plot.left() - 38, plot.bottom() + 4, f"{y_min:.4f}")

        line_pen = QPen(self._color, 1.8)
        p.setPen(line_pen)

        prev = None
        for x, y in self._points:
            px = plot.left() + (x - x_min) * plot.width() / (x_max - x_min)
            py = plot.bottom() - (y - y_min) * plot.height() / (y_max - y_min)
            if prev is not None:
                p.drawLine(prev[0], prev[1], px, py)
            prev = (px, py)

        p.setPen(QColor(220, 225, 232))
        p.drawText(plot.right() - 140, plot.top() + 16, f"latest={ys[-1]:.4f}")


class TrainPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_dir = Path(DEFAULT_PROJECT_DIR)
        self._process: QProcess | None = None
        self._dataset_result: MMPoseDatasetBuildResult | None = None
        self._generated_config: GeneratedTrainingConfig | None = None

        self._last_target_key = TARGET_HUMAN
        self._category_name_by_target: dict[str, str] = {TARGET_HUMAN: "", TARGET_ARCHERY: ""}
        self._metric_step = 0
        self._latest_metrics: dict[str, float | None] = {
            "loss": None,
            "mAP": None,
            "precision": None,
            "recall": None,
        }
        self._best_metrics: dict[str, float | None] = {
            "loss": None,
            "mAP": None,
            "precision": None,
            "recall": None,
        }
        self._latest_epoch: int | None = None
        self._latest_iter: tuple[int, int] | None = None
        self._latest_eta: str = "-"
        self._shown_low_pagefile_hint = False
        self._shown_encoding_hint = False

        self._build_ui()

        preset = default_training_preset(self._project_dir)
        self._apply_preset_to_ui(preset)
        self._on_init_mode_changed(self.init_mode_combo.currentText())
        self._apply_model_input_size_policy()
        self._refresh_keypoint_set_info(show_errors=False, set_category_from_set=True)

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        info = QLabel(
            f"训练模块 v0.3：项目目录固定为 {self._project_dir}，左侧配置+日志，右侧显示 loss/precision 曲线与 mAP/recall 实时状态。"
        )
        info.setWordWrap(True)
        left_layout.addWidget(info)

        config_box = QGroupBox("训练配置")
        form = QFormLayout(config_box)

        fixed_project = QLabel(str(self._project_dir))
        fixed_project.setTextInteractionFlags(Qt.TextSelectableByMouse)

        task_row = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(TRAIN_TASK_OPTIONS)
        task_row.addWidget(self.task_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list_model_names())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        task_row.addWidget(self.model_combo)

        self.target_combo = QComboBox()
        self.target_combo.addItems(TRAIN_TARGET_OPTIONS)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        task_row.addWidget(self.target_combo)

        init_row = QHBoxLayout()
        self.init_mode_combo = QComboBox()
        self.init_mode_combo.addItems(INIT_MODE_OPTIONS)
        self.init_mode_combo.currentTextChanged.connect(self._on_init_mode_changed)
        init_row.addWidget(self.init_mode_combo)

        self.pretrained_weight_edit = QLineEdit()
        self.pretrained_weight_edit.setPlaceholderText("预训练权重路径（.pth/.pt）")
        init_row.addWidget(self.pretrained_weight_edit, stretch=1)

        self.choose_weight_btn = QPushButton("权重")
        self.choose_weight_btn.setFixedWidth(80)
        self.choose_weight_btn.clicked.connect(self._select_weight_file)
        init_row.addWidget(self.choose_weight_btn)

        split_row = QHBoxLayout()
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setPrefix("train=")
        split_row.addWidget(self.train_ratio_spin)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 1.0)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setValue(0.1)
        self.val_ratio_spin.setPrefix("val=")
        split_row.addWidget(self.val_ratio_spin)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.1)
        self.test_ratio_spin.setPrefix("test=")
        split_row.addWidget(self.test_ratio_spin)

        self.build_dataset_btn = QPushButton("生成MMPose数据")
        self.build_dataset_btn.clicked.connect(self._build_dataset_clicked)
        split_row.addWidget(self.build_dataset_btn)

        dataset_row = QHBoxLayout()
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setPlaceholderText("数据集输出目录")
        dataset_row.addWidget(self.dataset_dir_edit, stretch=1)

        choose_dataset_btn = QPushButton("目录")
        choose_dataset_btn.setFixedWidth(80)
        choose_dataset_btn.clicked.connect(self._select_dataset_dir)
        dataset_row.addWidget(choose_dataset_btn)

        keypoint_set_row = QHBoxLayout()
        self.keypoint_set_label = QLabel("Keypoint Set: -")
        keypoint_set_row.addWidget(self.keypoint_set_label, stretch=1)

        self.import_set_btn = QPushButton("导入关键点集合")
        self.import_set_btn.clicked.connect(self._import_keypoint_set_for_current_target)
        keypoint_set_row.addWidget(self.import_set_btn)

        category_row = QHBoxLayout()
        self.category_edit = QLineEdit()
        self.category_edit.setPlaceholderText("类别名称（默认=集合名）")
        category_row.addWidget(self.category_edit)

        hp_row = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setPrefix("epoch=")
        hp_row.addWidget(self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.batch_spin.setPrefix("batch=")
        hp_row.addWidget(self.batch_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(4)
        self.workers_spin.setPrefix("workers=")
        hp_row.addWidget(self.workers_spin)

        self.gpus_spin = QSpinBox()
        self.gpus_spin.setRange(0, 8)
        self.gpus_spin.setValue(1)
        self.gpus_spin.setPrefix("gpus=")
        hp_row.addWidget(self.gpus_spin)

        self.input_w_spin = QSpinBox()
        self.input_w_spin.setRange(64, 2048)
        self.input_w_spin.setSingleStep(32)
        self.input_w_spin.setValue(640)
        self.input_w_spin.setPrefix("w=")
        self.input_w_spin.valueChanged.connect(self._on_input_size_changed)
        hp_row.addWidget(self.input_w_spin)

        self.input_h_spin = QSpinBox()
        self.input_h_spin.setRange(64, 2048)
        self.input_h_spin.setSingleStep(32)
        self.input_h_spin.setValue(640)
        self.input_h_spin.setPrefix("h=")
        self.input_h_spin.valueChanged.connect(self._on_input_size_changed)
        hp_row.addWidget(self.input_h_spin)

        self.input_size_hint_label = QLabel("input_size=640x640")
        hp_row.addWidget(self.input_size_hint_label)


        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setPrefix("lr=")
        hp_row.addWidget(self.lr_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(OPTIMIZER_OPTIONS)
        hp_row.addWidget(self.optimizer_combo)

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(SCHEDULER_OPTIONS)
        hp_row.addWidget(self.scheduler_combo)

        self.train_params_help_label = QLabel("ⓘ")
        self.train_params_help_label.setStyleSheet("color:#2c3e50; font-weight:700; padding:0 6px;")
        self.train_params_help_label.setToolTip(
            "epoch: 训练轮数\n"
            "batch: 每卡每步样本数\n"
            "workers: DataLoader 线程数\n"
            "gpus: 使用GPU数量（0=CPU）\n"
            "w/h: 输入宽高（默认随模型带出，可编辑）\n"
            "input_size: 实际写入配置的输入尺寸\n"
            "lr: 初始学习率\n"
            "optimizer/scheduler: 优化器与学习率策略"
        )
        hp_row.addWidget(self.train_params_help_label)
        hp_row.addStretch(1)

        aug_row = QHBoxLayout()
        self.aug_affine_chk = QCheckBox("Affine")
        self.aug_affine_chk.setChecked(True)
        self.aug_affine_chk.setToolTip("BottomupRandomAffine")
        aug_row.addWidget(self.aug_affine_chk)

        self.aug_mosaic_chk = QCheckBox("Mosaic")
        self.aug_mosaic_chk.setChecked(True)
        self.aug_mosaic_chk.setToolTip("Mosaic拼接增强")
        aug_row.addWidget(self.aug_mosaic_chk)

        self.aug_mixup_chk = QCheckBox("MixUp")
        self.aug_mixup_chk.setChecked(True)
        self.aug_mixup_chk.setToolTip("YOLOXMixUp")
        aug_row.addWidget(self.aug_mixup_chk)

        self.aug_hsv_chk = QCheckBox("HSV")
        self.aug_hsv_chk.setChecked(True)
        self.aug_hsv_chk.setToolTip("YOLOXHSVRandomAug")
        aug_row.addWidget(self.aug_hsv_chk)

        self.aug_flip_chk = QCheckBox("RandomFlip")
        self.aug_flip_chk.setChecked(False)
        self.aug_flip_chk.setToolTip("默认关闭，避免左右点swap未配置时污染训练")
        aug_row.addWidget(self.aug_flip_chk)
        aug_row.addStretch(1)

        self.skeleton_edit = QPlainTextEdit()
        self.skeleton_edit.setPlaceholderText('{"0":{"link":["up","bowstring"],"id":0,"color":[100,150,200]}}')
        self.skeleton_edit.setMinimumHeight(90)

        self.joint_weights_edit = QLineEdit()
        self.joint_weights_edit.setPlaceholderText('[1.0, 1.1, 1.5, 1.0, 1.0]')

        self.sigmas_edit = QLineEdit()
        self.sigmas_edit.setPlaceholderText('[0.25, 0.25, 0.25, 0.25, 0.25]')

        config_template_row = QHBoxLayout()
        self.base_config_edit = QLineEdit()
        self.base_config_edit.setPlaceholderText("基础配置文件路径（.py）")
        config_template_row.addWidget(self.base_config_edit, stretch=1)

        choose_cfg_btn = QPushButton("配置")
        choose_cfg_btn.setFixedWidth(80)
        choose_cfg_btn.clicked.connect(self._select_base_config_file)
        config_template_row.addWidget(choose_cfg_btn)

        generated_cfg_row = QHBoxLayout()
        self.generated_config_edit = QLineEdit()
        self.generated_config_edit.setReadOnly(True)
        self.generated_config_edit.setPlaceholderText("生成后的训练配置文件")
        generated_cfg_row.addWidget(self.generated_config_edit, stretch=1)

        self.generate_cfg_btn = QPushButton("生成训练配置")
        self.generate_cfg_btn.clicked.connect(self._generate_training_config_clicked)
        generated_cfg_row.addWidget(self.generate_cfg_btn)

        preset_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("保存预设")
        self.load_preset_btn = QPushButton("加载预设")
        self.save_preset_btn.clicked.connect(self._save_preset)
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_row.addWidget(self.save_preset_btn)
        preset_row.addWidget(self.load_preset_btn)
        preset_row.addStretch(1)

        command_row = QHBoxLayout()
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("训练命令（可自动填充）")
        command_row.addWidget(self.command_edit, stretch=1)

        self.fill_command_btn = QPushButton("填充训练命令")
        self.fill_command_btn.clicked.connect(self._fill_training_command)
        command_row.addWidget(self.fill_command_btn)

        form.addRow("项目目录(固定)", fixed_project)
        form.addRow("任务/模型/目标", task_row)
        form.addRow("初始化", init_row)
        form.addRow("数据划分", split_row)
        form.addRow("数据目录", dataset_row)
        form.addRow("关键点集合", keypoint_set_row)
        form.addRow("类别名称", category_row)
        form.addRow("训练参数", hp_row)
        form.addRow("图像增强", aug_row)
        form.addRow("skeleton_info", self.skeleton_edit)
        form.addRow("joint_weights", self.joint_weights_edit)
        form.addRow("sigmas", self.sigmas_edit)
        form.addRow("基础配置", config_template_row)
        form.addRow("生成配置", generated_cfg_row)
        form.addRow("参数预设", preset_row)
        form.addRow("训练命令", command_row)
        left_layout.addWidget(config_box)

        control_row = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.stop_btn = QPushButton("停止训练")
        self.clear_btn = QPushButton("清空日志")
        self.clear_curve_btn = QPushButton("清空曲线")

        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)

        self.clear_btn.clicked.connect(self._clear_log)
        self.clear_curve_btn.clicked.connect(self._clear_curves)

        control_row.addWidget(self.start_btn)
        control_row.addWidget(self.stop_btn)
        control_row.addWidget(self.clear_btn)
        control_row.addWidget(self.clear_curve_btn)
        control_row.addStretch(1)
        left_layout.addLayout(control_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志输出区域")
        left_layout.addWidget(self.log, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        status_box = QGroupBox("训练状态")
        status_layout = QVBoxLayout(status_box)

        self.run_status_label = QLabel("状态: 未开始")
        self.progress_label = QLabel("进度: epoch=- | iter=-/- | ETA=-")
        self.metric_label = QLabel("当前指标: loss=- | mAP=- | precision=- | recall=-")
        self.best_metric_label = QLabel("最佳指标: loss=- | mAP=- | precision=- | recall=-")
        self.metric_label.setWordWrap(True)
        self.best_metric_label.setWordWrap(True)

        status_layout.addWidget(self.run_status_label)
        status_layout.addWidget(self.progress_label)
        status_layout.addWidget(self.metric_label)
        status_layout.addWidget(self.best_metric_label)

        right_layout.addWidget(status_box)

        self.loss_chart = LineChartWidget("Loss", QColor(255, 121, 97))
        self.precision_chart = LineChartWidget("Precision/mAP", QColor(88, 202, 140))

        right_layout.addWidget(self.loss_chart)
        right_layout.addWidget(self.precision_chart)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 900])

        root.addWidget(splitter)

    def _clear_log(self) -> None:
        self.log.clear()

    def _clear_curves(self) -> None:
        self._reset_metric_views(reset_status=False)

    def _reset_metric_views(self, reset_status: bool) -> None:
        self._metric_step = 0
        self.loss_chart.clear_data()
        self.precision_chart.clear_data()

        self._latest_metrics = {"loss": None, "mAP": None, "precision": None, "recall": None}
        self._best_metrics = {"loss": None, "mAP": None, "precision": None, "recall": None}
        self._latest_epoch = None
        self._latest_iter = None
        self._latest_eta = "-"

        if reset_status:
            self.run_status_label.setText("状态: 未开始")
        self._update_progress_label()
        self._update_metric_labels()

    def _update_progress_label(self) -> None:
        epoch_text = str(self._latest_epoch) if self._latest_epoch is not None else "-"
        if self._latest_iter is None:
            iter_text = "-/-"
        else:
            iter_text = f"{self._latest_iter[0]}/{self._latest_iter[1]}"
        self.progress_label.setText(f"进度: epoch={epoch_text} | iter={iter_text} | ETA={self._latest_eta}")

    def _fmt_metric(self, value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    def _update_metric_labels(self) -> None:
        self.metric_label.setText(
            "当前指标: "
            f"loss={self._fmt_metric(self._latest_metrics['loss'])} | "
            f"mAP={self._fmt_metric(self._latest_metrics['mAP'])} | "
            f"precision={self._fmt_metric(self._latest_metrics['precision'])} | "
            f"recall={self._fmt_metric(self._latest_metrics['recall'])}"
        )
        self.best_metric_label.setText(
            "最佳指标: "
            f"loss={self._fmt_metric(self._best_metrics['loss'])} | "
            f"mAP={self._fmt_metric(self._best_metrics['mAP'])} | "
            f"precision={self._fmt_metric(self._best_metrics['precision'])} | "
            f"recall={self._fmt_metric(self._best_metrics['recall'])}"
        )

    def _update_metric_value(self, name: str, value: float, lower_is_better: bool) -> None:
        self._latest_metrics[name] = value
        best = self._best_metrics.get(name)
        if best is None:
            self._best_metrics[name] = value
            return
        if lower_is_better and value < best:
            self._best_metrics[name] = value
        if (not lower_is_better) and value > best:
            self._best_metrics[name] = value

    def _current_target_key(self) -> str:
        return TARGET_ARCHERY if self.target_combo.currentIndex() == 1 else TARGET_HUMAN

    def _refresh_keypoint_set_info(self, show_errors: bool, set_category_from_set: bool) -> None:
        target = self._current_target_key()
        try:
            ensure_project_keypoint_sets(self._project_dir)
            keypoint_set = get_active_keypoint_set(self._project_dir, target)
        except Exception as exc:
            self.keypoint_set_label.setText("Keypoint Set: unavailable")
            if show_errors:
                QMessageBox.warning(self, "Keypoint Set", str(exc))
            return

        self.keypoint_set_label.setText(f"Keypoint Set: {keypoint_set.set_name} ({keypoint_set.count} pts)")

        if set_category_from_set:
            existing = self._category_name_by_target.get(target, "").strip()
            value = existing or keypoint_set.set_name
            self.category_edit.setText(value)

        self._ensure_meta_defaults(keypoint_set.names, force=False)

    def _ensure_meta_defaults(self, keypoint_names: list[str], force: bool) -> None:
        if not keypoint_names:
            return

        if force or not self.joint_weights_edit.text().strip():
            self.joint_weights_edit.setText(str(self._default_joint_weights_for_keypoints(keypoint_names)))
        if force or not self.sigmas_edit.text().strip():
            self.sigmas_edit.setText(str([0.05 for _ in keypoint_names]))

        if force or not self.skeleton_edit.toPlainText().strip():
            skeleton = {
                i: {
                    "link": [keypoint_names[i], keypoint_names[i + 1]],
                    "id": i,
                    "color": [51, 153, 255],
                }
                for i in range(max(0, len(keypoint_names) - 1))
            }
            self.skeleton_edit.setPlainText(str(skeleton))

    def _default_joint_weights_for_keypoints(self, keypoint_names: list[str]) -> list[float]:
        normalized = [str(name).strip().lower().replace("_", "") for name in keypoint_names]
        halpe26_order = [
            "nose",
            "leye",
            "reye",
            "lear",
            "rear",
            "lshoulder",
            "rshoulder",
            "lelbow",
            "relbow",
            "lwrist",
            "rwrist",
            "lhip",
            "rhip",
            "lknee",
            "rknee",
            "lankle",
            "rankle",
            "head",
            "neck",
            "hip",
            "lbigtoe",
            "rbigtoe",
            "lsmalltoe",
            "rsmalltoe",
            "lheel",
            "rheel",
        ]
        if len(normalized) == 26 and normalized == halpe26_order:
            return [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.2,
                1.2,
                1.5,
                1.5,
                1.0,
                1.0,
                1.2,
                1.2,
                1.5,
                1.5,
                1.0,
                1.0,
                1.2,
                1.5,
                1.5,
                1.5,
                1.5,
                1.5,
                1.5,
            ]
        return [1.0 for _ in keypoint_names]

    def _import_keypoint_set_for_current_target(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Import Keypoint Set (JSON)",
            str(self._project_dir),
            "JSON files (*.json)",
        )
        if not source:
            return

        target = self._current_target_key()
        try:
            imported = import_keypoint_set_for_target(self._project_dir, Path(source), target)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return

        self._category_name_by_target[target] = imported.set_name
        self._refresh_keypoint_set_info(show_errors=False, set_category_from_set=True)
        self._dataset_result = None
        self._generated_config = None
        self.generated_config_edit.clear()

        QMessageBox.information(
            self,
            "Import Success",
            f"Target: {'Human' if target == TARGET_HUMAN else 'Archery'}\nSet: {imported.set_name}\nCount: {imported.count}",
        )

    def _select_weight_file(self) -> None:
        current = self.pretrained_weight_edit.text().strip() or str(self._project_dir)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择预训练权重",
            current,
            "权重文件 (*.pth *.pt *.ckpt);;所有文件 (*)",
        )
        if selected:
            self.pretrained_weight_edit.setText(selected)

    def _select_base_config_file(self) -> None:
        current = self.base_config_edit.text().strip() or str(self._project_dir)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择基础配置文件",
            current,
            "Python配置 (*.py);;所有文件 (*)",
        )
        if selected:
            self.base_config_edit.setText(selected)

    def _select_dataset_dir(self) -> None:
        current = self.dataset_dir_edit.text().strip() or str(self._project_dir / "output")
        selected = QFileDialog.getExistingDirectory(self, "选择数据目录", current)
        if selected:
            self.dataset_dir_edit.setText(selected)

    def _build_dataset_clicked(self) -> None:
        self._build_dataset(show_message=True)

    def _build_dataset(self, show_message: bool) -> MMPoseDatasetBuildResult | None:
        preset = self._collect_preset_from_ui()

        if not self._project_dir.exists():
            QMessageBox.warning(self, "提示", f"项目目录不存在: {self._project_dir}")
            return None

        ratio_sum = preset.train_ratio + preset.val_ratio + preset.test_ratio
        if ratio_sum <= 0:
            QMessageBox.warning(self, "提示", "train/val/test 比例之和必须大于 0。")
            return None

        try:
            result = build_mmpose_coco_dataset(
                project_root=self._project_dir,
                train_ratio=preset.train_ratio,
                val_ratio=preset.val_ratio,
                test_ratio=preset.test_ratio,
                random_seed=preset.random_seed,
                active_target=self._current_target_key(),
                active_category_name=preset.category_name,
                output_dir=preset.dataset_dir,
            )
        except Exception as exc:
            self._append_log(f"[ERROR] MMPose数据构建失败: {exc}")
            QMessageBox.critical(self, "构建失败", str(exc))
            return None

        self._dataset_result = result
        self._generated_config = None
        self.generated_config_edit.clear()
        self.dataset_dir_edit.setText(str(result.output_dir))

        self._append_log(
            (
                f"[INFO] MMPose数据集构建完成: {result.output_dir} | "
                f"images(total/train/val/test)={result.total_images}/{result.train_images}/{result.val_images}/{result.test_images} | "
                f"human_ann(train/val/test)={result.human_ann_counts['train']}/{result.human_ann_counts['val']}/{result.human_ann_counts['test']} | "
                f"archery_ann(train/val/test)={result.archery_ann_counts['train']}/{result.archery_ann_counts['val']}/{result.archery_ann_counts['test']}"
            )
        )

        if show_message:
            QMessageBox.information(
                self,
                "构建完成",
                (
                    f"输出目录: {result.output_dir}\n"
                    f"总图片: {result.total_images}\n"
                    f"train/val/test: {result.train_images}/{result.val_images}/{result.test_images}"
                ),
            )

        return result

    def _generate_training_config_clicked(self) -> None:
        self._generate_config(show_message=True)

    def _check_mmpose_version_gate(self, show_message: bool) -> bool:
        ok, current = is_required_mmpose_version()
        if ok:
            return True

        current_text = current or "not installed"
        fix_cmd = f"{sys.executable} -m pip install -U mmpose=={REQUIRED_MMPOSE_VERSION}"
        msg = (
            f"Detected mmpose version: {current_text}\n"
            f"Required version: {REQUIRED_MMPOSE_VERSION}\n\n"
            f"Please run:\n{fix_cmd}"
        )
        self._append_log(f"[ERROR] {msg.replace(chr(10), ' ')}")
        if show_message:
            QMessageBox.warning(self, "Version Mismatch", msg)
        return False

    def _generate_config(self, show_message: bool) -> GeneratedTrainingConfig | None:
        if not self._check_mmpose_version_gate(show_message=show_message):
            return None

        preset = self._collect_preset_from_ui()
        dataset_result = self._dataset_result
        if dataset_result is None:
            dataset_result = self._build_dataset(show_message=False)
            if dataset_result is None:
                return None

        try:
            generated = generate_training_config(preset=preset, dataset_result=dataset_result)
        except Exception as exc:
            self._append_log(f"[ERROR] 训练配置生成失败: {exc}")
            QMessageBox.critical(self, "生成失败", str(exc))
            return None

        self._generated_config = generated
        self.generated_config_edit.setText(str(generated.config_path))
        self._append_log(
            f"[INFO] 训练配置已生成: {generated.config_path} | target={generated.target_name} | num_keypoints={generated.num_keypoints}"
        )

        if show_message:
            QMessageBox.information(
                self,
                "配置已生成",
                f"配置文件: {generated.config_path}\nwork_dir: {generated.work_dir}",
            )

        return generated

    def _fill_training_command(self) -> None:
        generated = self._generated_config
        if generated is None:
            generated = self._generate_config(show_message=False)
            if generated is None:
                return

        self.command_edit.setText(generated.command)
        self._append_log(f"[INFO] 已填充训练命令: {generated.command}")

    def _save_preset(self) -> None:
        preset = self._collect_preset_from_ui()
        default_path = self._project_dir / "output" / "train_presets" / "train_preset.json"

        selected, _ = QFileDialog.getSaveFileName(
            self,
            "保存训练预设",
            str(default_path),
            "JSON文件 (*.json)",
        )
        if not selected:
            return

        try:
            save_training_preset(Path(selected), preset)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        self._append_log(f"[INFO] 训练预设已保存: {selected}")

    def _load_preset(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "加载训练预设",
            str(self._project_dir),
            "JSON文件 (*.json)",
        )
        if not selected:
            return

        try:
            preset = load_training_preset(Path(selected), fallback_project_root=self._project_dir)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self._apply_preset_to_ui(preset)
        self._refresh_keypoint_set_info(show_errors=False, set_category_from_set=False)
        self._dataset_result = None
        self._generated_config = None
        self.generated_config_edit.clear()
        self._append_log(f"[INFO] 训练预设已加载: {selected}")

    def _collect_preset_from_ui(self) -> TrainingPreset:
        dataset_dir_text = self.dataset_dir_edit.text().strip()
        dataset_dir = Path(dataset_dir_text) if dataset_dir_text else self._project_dir / "output" / "mmpose_dataset_v1"

        target_key = self._current_target_key()
        self._category_name_by_target[target_key] = self.category_edit.text().strip()

        return TrainingPreset(
            project_dir=self._project_dir,
            dataset_dir=dataset_dir,
            task_type=self.task_combo.currentText().strip() or TRAIN_TASK_OPTIONS[0],
            model_name=self.model_combo.currentText().strip() or list_model_names()[0],
            target_name=self.target_combo.currentText().strip() or TRAIN_TARGET_OPTIONS[0],
            init_mode=self.init_mode_combo.currentText().strip() or INIT_MODE_OPTIONS[0],
            pretrained_weight_path=self.pretrained_weight_edit.text().strip(),
            base_config_path=self.base_config_edit.text().strip(),
            category_name=self.category_edit.text().strip(),
            skeleton_info=self.skeleton_edit.toPlainText().strip(),
            joint_weights=self.joint_weights_edit.text().strip(),
            sigmas=self.sigmas_edit.text().strip(),
            epochs=int(self.epochs_spin.value()),
            batch_size=int(self.batch_spin.value()),
            num_workers=int(self.workers_spin.value()),
            input_width=int(self.input_w_spin.value()),
            input_height=int(self.input_h_spin.value()),
            image_size=int(self.input_w_spin.value()),
            learning_rate=float(self.lr_spin.value()),
            aug_random_affine=bool(self.aug_affine_chk.isChecked()),
            aug_mosaic=bool(self.aug_mosaic_chk.isChecked()),
            aug_mixup=bool(self.aug_mixup_chk.isChecked()),
            aug_hsv=bool(self.aug_hsv_chk.isChecked()),
            aug_random_flip=bool(self.aug_flip_chk.isChecked()),
            optimizer=self.optimizer_combo.currentText().strip() or OPTIMIZER_OPTIONS[0],
            scheduler=self.scheduler_combo.currentText().strip() or SCHEDULER_OPTIONS[0],
            train_ratio=float(self.train_ratio_spin.value()),
            val_ratio=float(self.val_ratio_spin.value()),
            test_ratio=float(self.test_ratio_spin.value()),
            random_seed=42,
            gpus=int(self.gpus_spin.value()),
        )
    def _apply_preset_to_ui(self, preset: TrainingPreset) -> None:
        self.dataset_dir_edit.setText(str(preset.dataset_dir))

        self._set_combo(self.task_combo, preset.task_type)
        self._set_combo(self.model_combo, preset.model_name)
        self._set_combo(self.target_combo, preset.target_name)
        self._set_combo(self.init_mode_combo, preset.init_mode)
        self._set_combo(self.optimizer_combo, preset.optimizer)
        self._set_combo(self.scheduler_combo, preset.scheduler)

        self.pretrained_weight_edit.setText(str(preset.pretrained_weight_path))
        self.base_config_edit.setText(str(preset.base_config_path))

        target_key = self._current_target_key()
        self._category_name_by_target[target_key] = preset.category_name
        self.category_edit.setText(preset.category_name)

        self.skeleton_edit.setPlainText(preset.skeleton_info)
        self.joint_weights_edit.setText(preset.joint_weights)
        self.sigmas_edit.setText(preset.sigmas)

        self.epochs_spin.setValue(max(1, int(preset.epochs)))
        self.batch_spin.setValue(max(1, int(preset.batch_size)))
        self.workers_spin.setValue(max(0, int(preset.num_workers)))
        self.input_w_spin.setValue(max(64, int(getattr(preset, "input_width", preset.image_size))))
        self.input_h_spin.setValue(max(64, int(getattr(preset, "input_height", preset.image_size))))
        self.lr_spin.setValue(max(0.000001, float(preset.learning_rate)))
        self.aug_affine_chk.setChecked(bool(getattr(preset, "aug_random_affine", True)))
        self.aug_mosaic_chk.setChecked(bool(getattr(preset, "aug_mosaic", True)))
        self.aug_mixup_chk.setChecked(bool(getattr(preset, "aug_mixup", True)))
        self.aug_hsv_chk.setChecked(bool(getattr(preset, "aug_hsv", True)))
        self.aug_flip_chk.setChecked(bool(getattr(preset, "aug_random_flip", False)))
        self.gpus_spin.setValue(max(0, int(preset.gpus)))

        self.train_ratio_spin.setValue(max(0.0, float(preset.train_ratio)))
        self.val_ratio_spin.setValue(max(0.0, float(preset.val_ratio)))
        self.test_ratio_spin.setValue(max(0.0, float(preset.test_ratio)))
        self._apply_model_input_size_policy()

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_target_changed(self, _text: str) -> None:
        self._category_name_by_target[self._last_target_key] = self.category_edit.text().strip()
        new_target = self._current_target_key()
        self._last_target_key = new_target

        self._refresh_keypoint_set_info(show_errors=False, set_category_from_set=False)
        cached = self._category_name_by_target.get(new_target, "").strip()
        if cached:
            self.category_edit.setText(cached)
        else:
            self._refresh_keypoint_set_info(show_errors=False, set_category_from_set=True)

    def _on_model_changed(self, model_name: str) -> None:
        try:
            config_path, weight_path = resolve_model_assets(model_name)
        except Exception as exc:
            self._append_log(f"[WARN] Model asset resolve failed: {exc}")
            return

        self.base_config_edit.setText(str(config_path))
        if self.init_mode_combo.currentText() == INIT_MODE_OPTIONS[0]:
            self.pretrained_weight_edit.setText(str(weight_path) if weight_path and weight_path.exists() else "")
        self._apply_model_input_size_policy()

    def _on_init_mode_changed(self, mode_text: str) -> None:
        use_pretrained = mode_text == INIT_MODE_OPTIONS[0]
        self.pretrained_weight_edit.setEnabled(use_pretrained)
        self.choose_weight_btn.setEnabled(use_pretrained)

        if use_pretrained and not self.pretrained_weight_edit.text().strip():
            try:
                _, weight_path = resolve_model_assets(self.model_combo.currentText())
            except Exception:
                weight_path = None
            self.pretrained_weight_edit.setText(str(weight_path) if weight_path and weight_path.exists() else "")

    def _is_halpe26_model(self) -> bool:
        model_name = self.model_combo.currentText().strip().lower()
        return "halpe26" in model_name

    def _refresh_input_size_hint(self) -> None:
        w = int(self.input_w_spin.value())
        h = int(self.input_h_spin.value())
        model_tip = " (HALPE26默认)" if self._is_halpe26_model() else ""
        self.input_size_hint_label.setText(f"input_size={w}x{h}{model_tip}")

    def _apply_model_input_size_policy(self) -> None:
        if self._is_halpe26_model():
            if int(self.input_w_spin.value()) == 640 and int(self.input_h_spin.value()) == 640:
                self.input_w_spin.blockSignals(True)
                self.input_h_spin.blockSignals(True)
                self.input_w_spin.setValue(288)
                self.input_h_spin.setValue(384)
                self.input_w_spin.blockSignals(False)
                self.input_h_spin.blockSignals(False)
            self.input_w_spin.setToolTip("HALPE26 默认输入尺寸 288x384，可编辑")
            self.input_h_spin.setToolTip("HALPE26 默认输入尺寸 288x384，可编辑")
        else:
            self.input_w_spin.setToolTip("")
            self.input_h_spin.setToolTip("")
        self._refresh_input_size_hint()

    def _on_input_size_changed(self, _value: int) -> None:
        self._refresh_input_size_hint()

    def _start_training(self) -> None:
        if not self._check_mmpose_version_gate(show_message=True):
            return

        if self._process is not None:
            QMessageBox.information(self, "提示", "训练任务已在运行。")
            return

        command = self.command_edit.text().strip()
        if not command:
            QMessageBox.warning(self, "提示", "请先填充或输入训练命令。")
            return

        if not self._project_dir.exists():
            QMessageBox.warning(self, "提示", f"项目目录不存在: {self._project_dir}")
            return

        self._reset_metric_views(reset_status=False)
        self.run_status_label.setText("状态: 运行中")

        process = QProcess(self)
        process.setWorkingDirectory(str(self._project_dir))

        env = QProcessEnvironment.systemEnvironment()
        if os.name == "nt":
            # Do not force UTF-8 on Windows training subprocess.
            # mmengine.collect_env decodes MSVC output with locale encoding.
            env.remove("PYTHONIOENCODING")
            env.remove("PYTHONUTF8")

        python_dir = str(Path(sys.executable).resolve().parent)
        current_path = env.value("PATH")
        if python_dir:
            path_entries = current_path.split(os.pathsep) if current_path else []
            if python_dir not in path_entries:
                merged = python_dir if not current_path else python_dir + os.pathsep + current_path
                env.insert("PATH", merged)

        if os.name == "nt":
            user_profile = env.value("USERPROFILE") or os.environ.get("USERPROFILE", "")
            home_drive = env.value("HOMEDRIVE") or os.environ.get("HOMEDRIVE", "")
            home_path = env.value("HOMEPATH") or os.environ.get("HOMEPATH", "")
            system_drive = env.value("SystemDrive") or os.environ.get("SystemDrive", "")

            if not user_profile and home_drive and home_path:
                user_profile = f"{home_drive}{home_path}"
            if not user_profile:
                user_profile = str(self._project_dir.resolve())

            if not home_drive:
                drive, _ = os.path.splitdrive(user_profile)
                home_drive = drive
            if not home_path:
                _, path_part = os.path.splitdrive(user_profile)
                home_path = path_part
            if not system_drive:
                drive, _ = os.path.splitdrive(user_profile)
                system_drive = drive

            if user_profile:
                env.insert("USERPROFILE", user_profile)
                env.insert("HOME", user_profile)
            if home_drive:
                env.insert("HOMEDRIVE", home_drive)
            if home_path:
                env.insert("HOMEPATH", home_path)
            if system_drive:
                env.insert("SystemDrive", system_drive)

        process.setProcessEnvironment(env)

        resolved_command = command
        if os.name == "nt":
            stripped = command.lstrip()
            python_exe = str(Path(sys.executable).resolve())
            if stripped.lower().startswith("python "):
                resolved_command = f'"{python_exe}" {stripped[7:]}'
                program = python_exe
                args = shlex.split(stripped[7:], posix=True)
            elif stripped.lower() == "python":
                resolved_command = f'"{python_exe}"'
                program = python_exe
                args = []
            else:
                program = "cmd"
                args = ["/d", "/c", resolved_command]
        else:
            program = "/bin/bash"
            args = ["-lc", command]

        process.readyReadStandardOutput.connect(self._on_stdout)
        process.readyReadStandardError.connect(self._on_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self._append_log(f"[INFO] 工作目录: {self._project_dir}")
        self._append_log(f"[INFO] 启动命令: {resolved_command if os.name == 'nt' else command}")

        process.start(program, args)
        if not process.waitForStarted(5000):
            self._append_log("[ERROR] 启动失败：无法拉起训练进程。")
            self.run_status_label.setText("状态: 启动失败")
            QMessageBox.critical(self, "启动失败", "无法拉起训练进程，请检查命令是否正确。")
            process.deleteLater()
            return

        self._process = process
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append_log("[INFO] 训练任务已启动。")

    def _stop_training(self) -> None:
        if self._process is None:
            return

        self._append_log("[INFO] 正在停止训练任务...")
        self.run_status_label.setText("状态: 正在停止")
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()
            self._process.waitForFinished(2000)


    def _decode_process_bytes(self, raw: bytes) -> str:
        if not raw:
            return ""
        for encoding in ("utf-8", "gbk", "cp936"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        text = self._decode_process_bytes(bytes(self._process.readAllStandardOutput()))
        if text:
            self._append_log(text.rstrip("\n"))

    def _on_stderr(self) -> None:
        if self._process is None:
            return
        text = self._decode_process_bytes(bytes(self._process.readAllStandardError()))
        if text:
            self._append_log(text.rstrip("\n"))

    def _on_finished(self, exit_code: int, _exit_status) -> None:  # type: ignore[override]
        if exit_code == 0:
            self.run_status_label.setText("状态: 已完成")
        else:
            self.run_status_label.setText(f"状态: 失败(退出码={exit_code})")

        self._append_log(
            "[INFO] 训练摘要: "
            f"latest(loss/mAP/precision/recall)="
            f"{self._fmt_metric(self._latest_metrics['loss'])}/"
            f"{self._fmt_metric(self._latest_metrics['mAP'])}/"
            f"{self._fmt_metric(self._latest_metrics['precision'])}/"
            f"{self._fmt_metric(self._latest_metrics['recall'])} | "
            f"best(loss/mAP/precision/recall)="
            f"{self._fmt_metric(self._best_metrics['loss'])}/"
            f"{self._fmt_metric(self._best_metrics['mAP'])}/"
            f"{self._fmt_metric(self._best_metrics['precision'])}/"
            f"{self._fmt_metric(self._best_metrics['recall'])}"
        )
        self._append_log(f"[INFO] Training finished, exit code: {exit_code}")
        self._cleanup_process_state()

    def _on_error(self, error) -> None:  # type: ignore[override]
        self.run_status_label.setText("状态: 子进程异常")
        self._append_log(f"[ERROR] Training process error: {error}")

    def _cleanup_process_state(self) -> None:
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        for line in text.splitlines():
            self.log.append(line)
            self._parse_and_update_progress(line)
            self._parse_and_plot_metrics(line)

            low_pagefile = (
                "WinError 1455" in line
                or "pagefile" in line.lower()
            )
            if low_pagefile and not self._shown_low_pagefile_hint:
                self._shown_low_pagefile_hint = True
                self.log.append("[HINT] Not a PowerShell issue: Windows virtual memory is too small.")
                self.log.append("[HINT] Increase pagefile size and close heavy apps, then retry.")
                self.log.append("[HINT] Suggested pagefile: Initial >= 32768MB, Max >= 65536MB.")

            encoding_conflict = (
                ("UnicodeDecodeError" in line)
                or ("collect_env.py" in line)
                or (("can't decode" in line) and ("utf-8" in line))
            )
            if encoding_conflict and not self._shown_encoding_hint:
                self._shown_encoding_hint = True
                self.log.append("[HINT] Encoding conflict detected in collect_env.")
                self.log.append("[HINT] Windows subprocess now avoids forced PYTHONUTF8/PYTHONIOENCODING.")
                self.log.append("[HINT] Please restart GUI and run training again.")

    def _parse_and_update_progress(self, line: str) -> None:
        epoch_iter = re.search(r"Epoch(?:\((?:train|val|test)\))?\s*\[(\d+)\]\[(\d+)/(\d+)\]", line, flags=re.IGNORECASE)
        changed = False
        if epoch_iter:
            self._latest_epoch = int(epoch_iter.group(1))
            self._latest_iter = (int(epoch_iter.group(2)), int(epoch_iter.group(3)))
            changed = True

        epoch_only = re.search(r"(?:^|\s)epoch\s*[:=]\s*(\d+)", line, flags=re.IGNORECASE)
        if epoch_only:
            self._latest_epoch = int(epoch_only.group(1))
            changed = True

        eta_match = re.search(r"(?:^|\s)eta\s*[:=]\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})", line, flags=re.IGNORECASE)
        if eta_match:
            self._latest_eta = eta_match.group(1)
            changed = True

        if changed:
            self._update_progress_label()

    def _parse_and_plot_metrics(self, line: str) -> None:
        loss = self._extract_metric(
            line,
            [
                r"(?:^|\s)loss(?:_[A-Za-z0-9]+)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"'loss'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r'"loss"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            ],
        )
        map_value = self._extract_metric(
            line,
            [
                r"(?:mAP|coco/AP)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"'mAP'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r'"mAP"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                r"'coco/AP'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r'"coco/AP"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            ],
        )
        precision = self._extract_metric(
            line,
            [
                r"(?:precision|prec)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"'precision'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r'"precision"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            ],
        )
        recall = self._extract_metric(
            line,
            [
                r"(?:recall|coco/AR|\bAR\b)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"'recall'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r'"recall"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            ],
        )

        if loss is None and map_value is None and precision is None and recall is None:
            return

        curve_value = precision if precision is not None else map_value

        self._metric_step += 1
        if loss is not None:
            self.loss_chart.add_point(self._metric_step, loss)
            self._update_metric_value("loss", loss, lower_is_better=True)
        if map_value is not None:
            self._update_metric_value("mAP", map_value, lower_is_better=False)
        if precision is not None:
            self._update_metric_value("precision", precision, lower_is_better=False)
        if recall is not None:
            self._update_metric_value("recall", recall, lower_is_better=False)
        if curve_value is not None:
            self.precision_chart.add_point(self._metric_step, curve_value)

        self._update_metric_labels()

    def _extract_metric(self, line: str, patterns: list[str]) -> float | None:
        for pattern in patterns:
            m = re.search(pattern, line, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                return float(m.group(1))
            except Exception:
                continue
        return None


from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from archery_plus.config import BASE_DIR, DEFAULT_PROJECT_DIR, MODEL_FILES
from archery_plus.data.annotation_store import AnnotationStore
from archery_plus.pipelines.auto_annotator_halpe26 import HALPE26_NAMES, Halpe26AutoAnnotator
from archery_plus.pipelines.auto_annotator_rtmo_archery import RtmoArcheryAutoAnnotator
from archery_plus.services.model_registry import collect_artifacts
from archery_plus.ui.widgets.annotation_canvas import AnnotationCanvas

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHERY_KEYPOINTS = ["UP", "DOWN", "FL", "ST", "FS"]
HUMAN_KEYPOINTS = HALPE26_NAMES


class AnnotatePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_root = Path(DEFAULT_PROJECT_DIR)
        self._store = AnnotationStore(self._project_root)
        self._current_image_path: Path | None = None
        self._sidebar_width = 280
        self._human_auto_annotator: Halpe26AutoAnnotator | None = None
        self._archery_auto_annotator: RtmoArcheryAutoAnnotator | None = None
        self._last_archery_kp = ARCHERY_KEYPOINTS[0]
        self._last_human_kp = HUMAN_KEYPOINTS[0]
        self._build_ui()
        self._init_auto_annotators()
        self._sync_keypoint_options_by_group("archery")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QLabel("标注模块：支持手工标注、自动预标注与按开关自动保存。")
        header.setWordWrap(True)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_panel.setFixedWidth(self._sidebar_width)
        left_layout = QVBoxLayout(left_panel)

        project_box = QGroupBox("项目与数据")
        project_form = QFormLayout(project_box)
        project_sel = QHBoxLayout()
        project_sel.setSpacing(8)

        self.project_path_edit = QLineEdit(str(self._project_root))
        self.project_path_edit.setPlaceholderText("选择项目目录")
        self.project_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        project_sel.addWidget(self.project_path_edit, stretch=1)

        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._select_project_dir)
        project_sel.addWidget(browse_btn)

        self.load_images_btn = QPushButton("加载图片列表")
        self.load_images_btn.clicked.connect(self._load_images)

        project_form.addRow("项目目录", project_sel)
        project_form.addRow("", self.load_images_btn)
        left_layout.addWidget(project_box)

        model_box = QGroupBox("模型资产状态")
        model_layout = QVBoxLayout(model_box)
        self.model_list = QListWidget()
        for item in collect_artifacts():
            status = "就绪" if item.exists else "缺失"
            self.model_list.addItem(f"[{status}] {item.file_name} ({item.purpose})")
        model_layout.addWidget(self.model_list)
        left_layout.addWidget(model_box)

        left_layout.addWidget(QLabel("图片列表 (images/cam1)"))
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_image_changed)
        left_layout.addWidget(self.image_list, stretch=1)

        self.save_btn = QPushButton("保存当前图片标注")
        self.save_btn.clicked.connect(self._save_current_annotation)
        left_layout.addWidget(self.save_btn)

        self.image_count_label = QLabel("图片数: 0")
        left_layout.addWidget(self.image_count_label)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        kp_row = QHBoxLayout()
        kp_row.addWidget(QLabel("当前关键点"))
        self.kp_combo = QComboBox()
        self.kp_combo.currentTextChanged.connect(self._on_keypoint_changed)
        kp_row.addWidget(self.kp_combo)

        kp_row.addWidget(QLabel("编辑目标"))
        self.edit_group_combo = QComboBox()
        self.edit_group_combo.addItems(["弓箭关键点", "人体关键点"])
        self.edit_group_combo.currentIndexChanged.connect(self._on_edit_group_changed)
        kp_row.addWidget(self.edit_group_combo)

        self.show_human_checkbox = QCheckBox("显示人体点")
        self.show_human_checkbox.setChecked(True)
        self.show_human_checkbox.toggled.connect(self.canvas_show_human)
        kp_row.addWidget(self.show_human_checkbox)

        self.auto_save_checkbox = QCheckBox("自动保存")
        self.auto_save_checkbox.setChecked(True)
        kp_row.addWidget(self.auto_save_checkbox)

        self.points_status_label = QLabel("弓箭点数: 0")
        kp_row.addWidget(self.points_status_label)

        self.human_points_status_label = QLabel("人体点数: 0")
        kp_row.addWidget(self.human_points_status_label)

        kp_row.addWidget(QLabel("左键新增/拖拽，右键删除最近点"))
        kp_row.addStretch(1)
        right_layout.addLayout(kp_row)

        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("自动预标注"))

        auto_row.addWidget(QLabel("置信度"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        auto_row.addWidget(self.conf_spin)

        auto_row.addWidget(QLabel("NMS"))
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.50)
        auto_row.addWidget(self.nms_spin)

        self.auto_when_switch_checkbox = QCheckBox("切图自动预标注")
        auto_row.addWidget(self.auto_when_switch_checkbox)

        self.auto_human_btn = QPushButton("人体预标注")
        self.auto_human_btn.clicked.connect(self._run_human_auto_annotation_for_current)
        auto_row.addWidget(self.auto_human_btn)

        self.auto_archery_btn = QPushButton("弓箭预标注")
        self.auto_archery_btn.clicked.connect(self._run_archery_auto_annotation_for_current)
        auto_row.addWidget(self.auto_archery_btn)

        self.auto_all_btn = QPushButton("全部图片自动标注")
        self.auto_all_btn.clicked.connect(self._run_auto_annotation_for_all_images)
        auto_row.addWidget(self.auto_all_btn)

        self.auto_status_label = QLabel("模型: 未加载")
        auto_row.addWidget(self.auto_status_label)
        auto_row.addStretch(1)
        right_layout.addLayout(auto_row)

        self.canvas = AnnotationCanvas()
        self.canvas.set_current_keypoint_name(self._last_archery_kp)
        self.canvas.pointsChanged.connect(self._on_points_changed)
        self.canvas.pointAdded.connect(self._on_point_added_auto_next)
        self.canvas.pointRemoved.connect(self._on_point_removed_auto_save)
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._sidebar_width, 1700])

        root.addWidget(splitter, stretch=1)

    def _init_auto_annotators(self) -> None:
        human_model_path = BASE_DIR / MODEL_FILES["human_halpe26_onnx"]
        archery_model_path = BASE_DIR / MODEL_FILES["archery_keypoints_onnx"]

        self._human_auto_annotator = Halpe26AutoAnnotator(model_path=human_model_path)
        self._archery_auto_annotator = RtmoArcheryAutoAnnotator(model_path=archery_model_path)

        human_status = "人体:就绪" if self._human_auto_annotator.is_ready() else f"人体:缺失 {human_model_path.name}"
        archery_status = (
            "弓箭:就绪"
            if self._archery_auto_annotator.is_ready()
            else f"弓箭:缺失 {archery_model_path.name}"
        )
        self.auto_status_label.setText(f"{human_status} | {archery_status}")

    def canvas_show_human(self, show: bool) -> None:
        self.canvas.set_show_human_points(show)

    def _on_edit_group_changed(self, index: int) -> None:
        group = "human" if index == 1 else "archery"
        self.canvas.set_edit_group(group)
        self._sync_keypoint_options_by_group(group)

    def _sync_keypoint_options_by_group(self, group: str) -> None:
        options = HUMAN_KEYPOINTS if group == "human" else ARCHERY_KEYPOINTS
        current_name = self._last_human_kp if group == "human" else self._last_archery_kp
        if current_name not in options:
            current_name = options[0]

        self.kp_combo.blockSignals(True)
        self.kp_combo.clear()
        self.kp_combo.addItems(options)
        self.kp_combo.setCurrentText(current_name)
        self.kp_combo.blockSignals(False)

        self.canvas.set_current_keypoint_name(current_name)

    def _select_project_dir(self) -> None:
        current = self.project_path_edit.text().strip() or str(self._project_root)
        selected = QFileDialog.getExistingDirectory(self, "选择项目目录", current)
        if selected:
            self.project_path_edit.setText(selected)

    def _load_images(self) -> None:
        root_text = self.project_path_edit.text().strip()
        if not root_text:
            QMessageBox.information(self, "提示", "请先选择项目目录。")
            return

        project_root = Path(root_text)
        cam1_dir = project_root / "images" / "cam1"
        if not cam1_dir.exists():
            QMessageBox.warning(self, "目录不存在", f"未找到目录: {cam1_dir}")
            return

        image_paths = sorted(
            [p for p in cam1_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
        )

        self._project_root = project_root
        self._store = AnnotationStore(self._project_root)

        self.image_list.clear()
        for p in image_paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.image_list.addItem(item)

        self.image_count_label.setText(f"图片数: {len(image_paths)}")

        if image_paths:
            self.image_list.setCurrentRow(0)
        else:
            self.canvas.clear_image()
            self.points_status_label.setText("弓箭点数: 0")
            self.human_points_status_label.setText("人体点数: 0")

    def _on_image_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        _ = previous
        if current is None:
            self._current_image_path = None
            self.canvas.clear_image()
            self.points_status_label.setText("弓箭点数: 0")
            self.human_points_status_label.setText("人体点数: 0")
            return

        path = Path(current.data(Qt.ItemDataRole.UserRole))
        image = QImage(str(path))
        if image.isNull():
            QMessageBox.warning(self, "图片加载失败", f"无法读取图片: {path}")
            return

        self._current_image_path = path
        self.canvas.set_image(image)

        annotation = self._store.get_image_annotation(path.name)
        archery_points = annotation["archery_keypoints"]
        human_points = annotation["human_keypoints"]

        self.canvas.set_archery_points(archery_points)
        self.canvas.set_human_points(human_points)
        self.points_status_label.setText(f"弓箭点数: {len(archery_points)}")
        self.human_points_status_label.setText(f"人体点数: {len(human_points)}")

        if self.auto_when_switch_checkbox.isChecked():
            if not human_points:
                self._run_human_auto_annotation_for_current(show_message=False)
            if not archery_points:
                self._run_archery_auto_annotation_for_current(show_message=False)

    def _on_keypoint_changed(self, name: str) -> None:
        group = self.canvas.get_edit_group()
        if group == "human":
            self._last_human_kp = name
        else:
            self._last_archery_kp = name
        self.canvas.set_current_keypoint_name(name)

    def _on_points_changed(self) -> None:
        self.points_status_label.setText(f"弓箭点数: {len(self.canvas.get_archery_points())}")
        self.human_points_status_label.setText(f"人体点数: {len(self.canvas.get_human_points())}")

    def _on_point_added_auto_next(self) -> None:
        if not self.auto_save_checkbox.isChecked():
            return
        self._save_current_annotation(show_message=False)
        if self.canvas.get_edit_group() == "archery":
            self._goto_next_image()

    def _on_point_removed_auto_save(self) -> None:
        if not self.auto_save_checkbox.isChecked():
            return
        self._save_current_annotation(show_message=False)

    def _goto_next_image(self) -> None:
        row = self.image_list.currentRow()
        if row < 0:
            return

        next_row = row + 1
        if next_row < self.image_list.count():
            self.image_list.setCurrentRow(next_row)
        else:
            self.points_status_label.setText(
                f"弓箭点数: {len(self.canvas.get_archery_points())}（已到最后一张）"
            )

    def _save_current_annotation(self, show_message: bool = True) -> None:
        if self._current_image_path is None:
            if show_message:
                QMessageBox.information(self, "提示", "请先选择图片。")
            return

        archery_points = self.canvas.get_archery_points()
        human_points = self.canvas.get_human_points()
        payload = {
            "human_keypoints": human_points,
            "archery_keypoints": archery_points,
        }
        self._store.set_image_annotation(self._current_image_path.name, payload)

        if show_message:
            QMessageBox.information(
                self,
                "保存成功",
                (
                    f"已保存 {self._current_image_path.name} 的标注。\n"
                    f"人体关键点: {len(human_points)}\n"
                    f"弓箭关键点: {len(archery_points)}"
                ),
            )

    def _run_human_auto_annotation_for_current(self, show_message: bool = True) -> bool:
        if self._current_image_path is None:
            if show_message:
                QMessageBox.information(self, "提示", "请先选择图片。")
            return False

        if self._human_auto_annotator is None or not self._human_auto_annotator.is_ready():
            if show_message:
                QMessageBox.warning(self, "模型不可用", "未找到 Halpe26 ONNX 模型，请检查模型文件。")
            return False

        self._human_auto_annotator.set_thresholds(
            confidence_threshold=self.conf_spin.value(),
            nms_threshold=self.nms_spin.value(),
        )

        try:
            result = self._human_auto_annotator.predict_image(self._current_image_path)
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "人体预标注失败", str(exc))
            return False

        self.canvas.set_human_points(result.points)
        self.human_points_status_label.setText(f"人体点数: {len(result.points)}")
        self._save_current_annotation(show_message=False)

        if show_message:
            QMessageBox.information(
                self,
                "人体预标注完成",
                (
                    f"图片: {self._current_image_path.name}\n"
                    f"原始关键点: {result.raw_count}\n"
                    f"保留关键点: {len(result.points)}\n"
                    f"置信度阈值: {self.conf_spin.value():.2f}\n"
                    f"NMS阈值: {self.nms_spin.value():.2f}"
                ),
            )

        return True

    def _run_archery_auto_annotation_for_current(self, show_message: bool = True) -> bool:
        if self._current_image_path is None:
            if show_message:
                QMessageBox.information(self, "提示", "请先选择图片。")
            return False

        if self._archery_auto_annotator is None or not self._archery_auto_annotator.is_ready():
            if show_message:
                QMessageBox.warning(self, "模型不可用", "未找到 RTMO ONNX 模型，请检查模型文件。")
            return False

        self._archery_auto_annotator.set_thresholds(
            confidence_threshold=self.conf_spin.value(),
            nms_threshold=self.nms_spin.value(),
        )

        try:
            result = self._archery_auto_annotator.predict_image(self._current_image_path)
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "弓箭预标注失败", str(exc))
            return False

        self.canvas.set_archery_points(result.points)
        self.points_status_label.setText(f"弓箭点数: {len(result.points)}")
        self._save_current_annotation(show_message=False)

        if show_message:
            QMessageBox.information(
                self,
                "弓箭预标注完成",
                (
                    f"图片: {self._current_image_path.name}\n"
                    f"原始检测数: {result.raw_detection_count}\n"
                    f"NMS后检测数: {result.kept_detection_count}\n"
                    f"输出关键点: {len(result.points)}\n"
                    f"置信度阈值: {self.conf_spin.value():.2f}\n"
                    f"NMS阈值: {self.nms_spin.value():.2f}"
                ),
            )

        return True

    def _run_auto_annotation_for_all_images(self) -> None:
        total = self.image_list.count()
        if total <= 0:
            QMessageBox.information(self, "提示", "请先加载图片列表。")
            return

        ok_human = 0
        ok_archery = 0
        fail_images = 0

        for row in range(total):
            self.image_list.setCurrentRow(row)
            QApplication.processEvents()

            if self._current_image_path is None:
                fail_images += 1
                continue

            human_ok = self._run_human_auto_annotation_for_current(show_message=False)
            archery_ok = self._run_archery_auto_annotation_for_current(show_message=False)

            ok_human += 1 if human_ok else 0
            ok_archery += 1 if archery_ok else 0
            if not human_ok or not archery_ok:
                fail_images += 1

        QMessageBox.information(
            self,
            "全部自动标注完成",
            (
                f"图片总数: {total}\n"
                f"人体预标注成功: {ok_human}\n"
                f"弓箭预标注成功: {ok_archery}\n"
                f"存在失败的图片: {fail_images}"
            ),
        )





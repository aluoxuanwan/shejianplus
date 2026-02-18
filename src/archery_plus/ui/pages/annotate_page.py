
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    QInputDialog,
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
from archery_plus.core.keypoint_schema import TARGET_ARCHERY, TARGET_HUMAN
from archery_plus.data.annotation_store import AnnotationStore
from archery_plus.pipelines.auto_annotator_halpe26 import Halpe26AutoAnnotator
from archery_plus.pipelines.auto_annotator_rtmo_archery import RtmoArcheryAutoAnnotator
from archery_plus.ui.widgets.annotation_canvas import AnnotationCanvas

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class AnnotatePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_root = Path(DEFAULT_PROJECT_DIR)
        self._store = AnnotationStore(self._project_root)
        self._current_image_path: Path | None = None
        self._sidebar_width = 320

        self._human_auto_annotator: Halpe26AutoAnnotator | None = None
        self._archery_auto_annotator: RtmoArcheryAutoAnnotator | None = None
        self._model_status_base = "模型: 未加载"

        self._human_keypoint_labels: list[str] = []
        self._archery_keypoint_labels: list[str] = []
        self._bbox_labels: list[str] = []

        self._last_archery_kp = "UP"
        self._last_human_kp = "Nose"
        self._last_bbox_label = "target"

        self._build_ui()
        self._reload_label_schema(show_errors=False)
        self._init_auto_annotators()
        self._sync_editor_controls()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QLabel("标注模块：支持关键点+目标框标注、自动预标注与自动保存。")
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

        label_box = QGroupBox("标签")
        label_layout = QVBoxLayout(label_box)

        label_btn_row = QHBoxLayout()
        self.import_label_btn = QPushButton("导入标签JSON")
        self.import_label_btn.clicked.connect(self._import_label_json)
        label_btn_row.addWidget(self.import_label_btn)

        self.add_kp_label_btn = QPushButton("添加关键点")
        self.add_kp_label_btn.clicked.connect(self._add_keypoint_label)
        label_btn_row.addWidget(self.add_kp_label_btn)

        self.add_box_label_btn = QPushButton("添加矩形")
        self.add_box_label_btn.clicked.connect(self._add_bbox_label)
        label_btn_row.addWidget(self.add_box_label_btn)
        label_layout.addLayout(label_btn_row)

        self.label_summary_label = QLabel("人体KP:0 | 弓箭KP:0 | 矩形标签:0")
        label_layout.addWidget(self.label_summary_label)

        label_layout.addWidget(QLabel("标签列表"))
        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(100)
        label_layout.addWidget(self.label_list)

        label_layout.addWidget(QLabel("当前图片标注（关键点+矩形）"))
        self.annotation_item_list = QListWidget()
        self.annotation_item_list.setMinimumHeight(120)
        label_layout.addWidget(self.annotation_item_list)

        left_layout.addWidget(label_box)

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
        kp_row.addWidget(QLabel("标注模式"))
        self.edit_mode_combo = QComboBox()
        self.edit_mode_combo.addItems(["关键点标注", "矩形标注"])
        self.edit_mode_combo.currentIndexChanged.connect(self._on_edit_mode_changed)
        kp_row.addWidget(self.edit_mode_combo)

        kp_row.addWidget(QLabel("编辑目标"))
        self.edit_group_combo = QComboBox()
        self.edit_group_combo.addItems(["弓箭关键点", "人体关键点"])
        self.edit_group_combo.currentIndexChanged.connect(self._on_edit_group_changed)
        kp_row.addWidget(self.edit_group_combo)

        kp_row.addWidget(QLabel("当前关键点"))
        self.kp_combo = QComboBox()
        self.kp_combo.currentTextChanged.connect(self._on_keypoint_changed)
        kp_row.addWidget(self.kp_combo)

        kp_row.addWidget(QLabel("当前矩形标签"))
        self.bbox_label_combo = QComboBox()
        self.bbox_label_combo.currentTextChanged.connect(self._on_bbox_label_changed)
        kp_row.addWidget(self.bbox_label_combo)

        self.auto_save_checkbox = QCheckBox("自动保存")
        self.auto_save_checkbox.setChecked(True)
        kp_row.addWidget(self.auto_save_checkbox)

        self.points_status_label = QLabel("弓箭点数: 0")
        kp_row.addWidget(self.points_status_label)

        self.human_points_status_label = QLabel("人体点数: 0")
        kp_row.addWidget(self.human_points_status_label)

        self.bbox_status_label = QLabel("框数: 0")
        kp_row.addWidget(self.bbox_status_label)

        self.mismatch_status_label = QLabel("历史不匹配: 人体0 弓箭0 框0")
        kp_row.addWidget(self.mismatch_status_label)

        kp_row.addWidget(QLabel("左键新增/拖拽，右键删除"))
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

        auto_row.addWidget(QLabel("IoU"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.50)
        auto_row.addWidget(self.iou_spin)

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
        self.canvas.set_current_bbox_name(self._last_bbox_label)
        self.canvas.pointsChanged.connect(self._on_annotations_changed)
        self.canvas.pointAdded.connect(self._on_point_added_auto_next)
        self.canvas.pointRemoved.connect(self._on_removed_auto_save)
        self.canvas.boxAdded.connect(self._on_box_added_auto_save)
        self.canvas.boxRemoved.connect(self._on_removed_auto_save)
        self.canvas.imageCursorMoved.connect(self._on_canvas_cursor_moved)
        right_layout.addWidget(self.canvas, stretch=1)

        self.cursor_status_label = QLabel("(X:-,Y:-)[图片名称:- 0/0]")
        right_layout.addWidget(self.cursor_status_label)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._sidebar_width, 1700])

        root.addWidget(splitter, stretch=1)

    def _select_project_dir(self) -> None:
        current = self.project_path_edit.text().strip() or str(self._project_root)
        selected = QFileDialog.getExistingDirectory(self, "选择项目目录", current)
        if not selected:
            return

        self.project_path_edit.setText(selected)
        self._project_root = Path(selected)
        self._store = AnnotationStore(self._project_root)
        self._reload_label_schema(show_errors=True)
        self._sync_editor_controls()

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

        image_paths = sorted([p for p in cam1_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])

        self._project_root = project_root
        self._store = AnnotationStore(self._project_root)
        self._reload_label_schema(show_errors=True)
        self._sync_editor_controls()

        self.image_list.clear()
        for p in image_paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.image_list.addItem(item)

        self.image_count_label.setText(f"图片数: {len(image_paths)}")

        if image_paths:
            self.image_list.setCurrentRow(0)
        else:
            self._current_image_path = None
            self.canvas.clear_image()
            self._refresh_counts_and_lists()
            self._update_cursor_status(None, None)
    def _on_image_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        _ = previous
        if current is None:
            self._current_image_path = None
            self.canvas.clear_image()
            self._refresh_counts_and_lists()
            self._update_cursor_status(None, None)
            return

        path = Path(current.data(Qt.ItemDataRole.UserRole))
        image = QImage(str(path))
        if image.isNull():
            QMessageBox.warning(self, "图片加载失败", f"无法读取图片: {path}")
            return

        self._current_image_path = path
        self.canvas.set_image(image)

        annotation = self._store.get_image_annotation(path.name)
        self.canvas.set_archery_points(annotation.get("archery_keypoints", []))
        self.canvas.set_human_points(annotation.get("human_keypoints", []))
        self.canvas.set_bboxes(annotation.get("bboxes", []))

        self._refresh_counts_and_lists()
        self._update_mismatch_status_for_image(path.name)
        self._update_cursor_status(None, None)

        if self.auto_when_switch_checkbox.isChecked():
            if not annotation.get("human_keypoints", []) and self.auto_human_btn.isEnabled():
                self._run_human_auto_annotation_for_current(show_message=False)
            if not annotation.get("archery_keypoints", []) and self.auto_archery_btn.isEnabled():
                self._run_archery_auto_annotation_for_current(show_message=False)

    def _save_current_annotation(self, show_message: bool = True) -> None:
        if self._current_image_path is None:
            if show_message:
                QMessageBox.information(self, "提示", "请先选择图片。")
            return

        payload = {
            "human_keypoints": self.canvas.get_human_points(),
            "archery_keypoints": self.canvas.get_archery_points(),
            "bboxes": self.canvas.get_bboxes(),
        }
        self._store.set_image_annotation(self._current_image_path.name, payload)

        if show_message:
            QMessageBox.information(
                self,
                "保存成功",
                (
                    f"已保存 {self._current_image_path.name} 的标注。\n"
                    f"人体关键点: {len(payload['human_keypoints'])}\n"
                    f"弓箭关键点: {len(payload['archery_keypoints'])}\n"
                    f"矩形框: {len(payload['bboxes'])}"
                ),
            )

    def _on_annotations_changed(self) -> None:
        self._refresh_counts_and_lists()
        if self._current_image_path is not None:
            self._update_mismatch_status_for_image(self._current_image_path.name)

    def _on_point_added_auto_next(self) -> None:
        if self.auto_save_checkbox.isChecked():
            self._save_current_annotation(show_message=False)

        if self.canvas.get_edit_mode() != "keypoint":
            return

        if self.canvas.get_edit_group() == TARGET_ARCHERY:
            self._goto_next_image()

    def _on_box_added_auto_save(self) -> None:
        if self.auto_save_checkbox.isChecked():
            self._save_current_annotation(show_message=False)

    def _on_removed_auto_save(self) -> None:
        if self.auto_save_checkbox.isChecked():
            self._save_current_annotation(show_message=False)

    def _goto_next_image(self) -> None:
        row = self.image_list.currentRow()
        if row < 0:
            return
        next_row = row + 1
        if next_row < self.image_list.count():
            self.image_list.setCurrentRow(next_row)

    def _on_edit_group_changed(self, _index: int) -> None:
        self._sync_editor_controls()

    def _on_edit_mode_changed(self, _index: int) -> None:
        self._sync_editor_controls()

    def _on_keypoint_changed(self, name: str) -> None:
        if not name:
            return
        group = self._current_target()
        if group == TARGET_HUMAN:
            self._last_human_kp = name
        else:
            self._last_archery_kp = name
        self.canvas.set_current_keypoint_name(name)

    def _on_bbox_label_changed(self, name: str) -> None:
        if not name:
            return
        self._last_bbox_label = name
        self.canvas.set_current_bbox_name(name)

    def _current_target(self) -> str:
        return TARGET_HUMAN if self.edit_group_combo.currentIndex() == 1 else TARGET_ARCHERY

    def _current_mode(self) -> str:
        return "bbox" if self.edit_mode_combo.currentIndex() == 1 else "keypoint"

    def _sync_editor_controls(self) -> None:
        target = self._current_target()
        mode = self._current_mode()

        self.canvas.set_edit_group(target)
        self.canvas.set_edit_mode(mode)

        kp_options = self._human_keypoint_labels if target == TARGET_HUMAN else self._archery_keypoint_labels
        current_kp = self._last_human_kp if target == TARGET_HUMAN else self._last_archery_kp

        self.kp_combo.blockSignals(True)
        self.kp_combo.clear()
        self.kp_combo.addItems(kp_options)
        if current_kp in kp_options:
            self.kp_combo.setCurrentText(current_kp)
        elif kp_options:
            self.kp_combo.setCurrentText(kp_options[0])
            if target == TARGET_HUMAN:
                self._last_human_kp = kp_options[0]
            else:
                self._last_archery_kp = kp_options[0]
        self.kp_combo.blockSignals(False)

        self.bbox_label_combo.blockSignals(True)
        self.bbox_label_combo.clear()
        self.bbox_label_combo.addItems(self._bbox_labels)
        if self._last_bbox_label in self._bbox_labels:
            self.bbox_label_combo.setCurrentText(self._last_bbox_label)
        elif self._bbox_labels:
            self.bbox_label_combo.setCurrentText(self._bbox_labels[0])
            self._last_bbox_label = self._bbox_labels[0]
        self.bbox_label_combo.blockSignals(False)
        if self.kp_combo.currentText():
            self.canvas.set_current_keypoint_name(self.kp_combo.currentText())
        if self.bbox_label_combo.currentText():
            self.canvas.set_current_bbox_name(self.bbox_label_combo.currentText())

        is_kp = mode == "keypoint"
        self.kp_combo.setEnabled(is_kp)
        self.edit_group_combo.setEnabled(is_kp)
        self.bbox_label_combo.setEnabled(not is_kp)

        self._update_annotation_enable_state()

    def _update_annotation_enable_state(self) -> None:
        mode = self._current_mode()
        if mode == "bbox":
            enabled = len(self._bbox_labels) > 0
        else:
            target = self._current_target()
            labels = self._human_keypoint_labels if target == TARGET_HUMAN else self._archery_keypoint_labels
            enabled = len(labels) > 0

        self.canvas.set_annotation_enabled(enabled)

    def _reload_label_schema(self, show_errors: bool) -> None:
        try:
            schema = self._store.ensure_label_schema()
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "标签读取失败", str(exc))
            schema = {
                "human_keypoint_labels": [],
                "archery_keypoint_labels": [],
                "bbox_labels": [],
            }

        self._human_keypoint_labels = list(schema.get("human_keypoint_labels", []))
        self._archery_keypoint_labels = list(schema.get("archery_keypoint_labels", []))
        self._bbox_labels = list(schema.get("bbox_labels", []))

        self._refresh_label_views()
        self._update_auto_annotation_availability()

    def _refresh_label_views(self) -> None:
        self.label_summary_label.setText(
            f"人体KP:{len(self._human_keypoint_labels)} | 弓箭KP:{len(self._archery_keypoint_labels)} | 矩形标签:{len(self._bbox_labels)}"
        )

        self.label_list.clear()
        for name in self._human_keypoint_labels:
            self.label_list.addItem(f"[人体KP] {name}")
        for name in self._archery_keypoint_labels:
            self.label_list.addItem(f"[弓箭KP] {name}")
        for name in self._bbox_labels:
            self.label_list.addItem(f"[矩形] {name}")

    def _refresh_counts_and_lists(self) -> None:
        archery_points = self.canvas.get_archery_points()
        human_points = self.canvas.get_human_points()
        bboxes = self.canvas.get_bboxes()

        self.points_status_label.setText(f"弓箭点数: {len(archery_points)}")
        self.human_points_status_label.setText(f"人体点数: {len(human_points)}")
        self.bbox_status_label.setText(f"框数: {len(bboxes)}")

        self.annotation_item_list.clear()
        for p in archery_points:
            self.annotation_item_list.addItem(f"[弓箭KP] {p.get('name', '')} ({p.get('x', 0):.1f}, {p.get('y', 0):.1f})")
        for p in human_points:
            self.annotation_item_list.addItem(f"[人体KP] {p.get('name', '')} ({p.get('x', 0):.1f}, {p.get('y', 0):.1f})")
        for b in bboxes:
            self.annotation_item_list.addItem(
                f"[矩形] {b.get('name', '')} [{b.get('x1', 0):.1f}, {b.get('y1', 0):.1f}, {b.get('x2', 0):.1f}, {b.get('y2', 0):.1f}]"
            )

    def _update_mismatch_status_for_image(self, image_name: str) -> None:
        mismatch = self._store.count_mismatched_names(
            image_name,
            valid_human_names=self._human_keypoint_labels,
            valid_archery_names=self._archery_keypoint_labels,
            valid_bbox_names=self._bbox_labels,
        )
        h = int(mismatch.get("human", 0))
        a = int(mismatch.get("archery", 0))
        b = int(mismatch.get("bbox", 0))
        self.mismatch_status_label.setText(f"历史不匹配: 人体{h} 弓箭{a} 框{b}")

        if h > 0 or a > 0 or b > 0:
            self.mismatch_status_label.setStyleSheet("color: #d35400;")
            self.mismatch_status_label.setToolTip("存在历史标签名称不在当前标签集合中，已保留原数据。")
        else:
            self.mismatch_status_label.setStyleSheet("")
            self.mismatch_status_label.setToolTip("")

    def _import_label_json(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "导入标签JSON",
            str(self._project_root),
            "JSON文件 (*.json)",
        )
        if not source:
            return

        try:
            with Path(source).open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", f"读取JSON失败: {exc}")
            return

        try:
            self._apply_label_payload(raw)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return

        self._reload_label_schema(show_errors=False)
        self._sync_editor_controls()
        if self._current_image_path is not None:
            self._update_mismatch_status_for_image(self._current_image_path.name)

        QMessageBox.information(self, "导入成功", "标签JSON已导入并保存到项目 annotations/label_schema.json")

    def _apply_label_payload(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise RuntimeError("标签JSON根节点必须是对象。")

        schema = self._store.ensure_label_schema()

        if isinstance(raw.get("keypoints"), list):
            keypoints = raw.get("keypoints", [])
            names: list[str] = []
            for item in keypoints:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                else:
                    name = str(item).strip()
                if name and name not in names:
                    names.append(name)
            if not names:
                raise RuntimeError("keypoints 为空或无有效名称。")
            self._store.set_keypoint_labels(self._current_target(), names)
            return

        any_applied = False
        if "human_keypoint_labels" in raw:
            schema["human_keypoint_labels"] = self._normalize_name_list(raw.get("human_keypoint_labels", []))
            any_applied = True
        if "archery_keypoint_labels" in raw:
            schema["archery_keypoint_labels"] = self._normalize_name_list(raw.get("archery_keypoint_labels", []))
            any_applied = True
        if "bbox_labels" in raw:
            schema["bbox_labels"] = self._normalize_name_list(raw.get("bbox_labels", []))
            any_applied = True

        if not any_applied:
            raise RuntimeError(
                "未识别到标签字段，请使用 keypoints / human_keypoint_labels / archery_keypoint_labels / bbox_labels。"
            )

        self._store.save_label_schema(schema)

    def _normalize_name_list(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        names: list[str] = []
        for item in raw:
            name = str(item).strip()
            if name and name not in names:
                names.append(name)
        return names

    def _add_keypoint_label(self) -> None:
        target = self._current_target()
        title = "添加人体关键点标签" if target == TARGET_HUMAN else "添加弓箭关键点标签"
        text, ok = QInputDialog.getText(self, title, "请输入关键点标签名称")
        if not ok:
            return

        name = text.strip()
        if not name:
            QMessageBox.information(self, "提示", "标签名称不能为空。")
            return

        self._store.add_keypoint_label(target, name)
        self._reload_label_schema(show_errors=False)
        self._sync_editor_controls()

    def _add_bbox_label(self) -> None:
        text, ok = QInputDialog.getText(self, "添加矩形标签", "请输入目标检测标签名称")
        if not ok:
            return

        name = text.strip()
        if not name:
            QMessageBox.information(self, "提示", "标签名称不能为空。")
            return

        self._store.add_bbox_label(name)
        self._reload_label_schema(show_errors=False)
        self._sync_editor_controls()

    def _init_auto_annotators(self) -> None:
        human_model_path = BASE_DIR / MODEL_FILES["human_halpe26_onnx"]
        archery_model_path = BASE_DIR / MODEL_FILES["archery_keypoints_onnx"]

        self._human_auto_annotator = Halpe26AutoAnnotator(model_path=human_model_path)
        self._archery_auto_annotator = RtmoArcheryAutoAnnotator(model_path=archery_model_path)

        human_status = "人体:就绪" if self._human_auto_annotator.is_ready() else f"人体:缺失 {human_model_path.name}"
        archery_status = "弓箭:就绪" if self._archery_auto_annotator.is_ready() else f"弓箭:缺失 {archery_model_path.name}"
        self._model_status_base = f"{human_status} | {archery_status}"
        self._update_auto_annotation_availability()

    def _update_auto_annotation_availability(self) -> None:
        human_compatible = len(self._human_keypoint_labels) == 26
        archery_compatible = len(self._archery_keypoint_labels) == 5

        self.auto_human_btn.setEnabled(human_compatible)
        self.auto_archery_btn.setEnabled(archery_compatible)
        self.auto_all_btn.setEnabled(human_compatible and archery_compatible)

        self.auto_human_btn.setToolTip("" if human_compatible else "人体自动预标注要求人体关键点标签数量为26")
        self.auto_archery_btn.setToolTip("" if archery_compatible else "弓箭自动预标注要求弓箭关键点标签数量为5")
        self.auto_all_btn.setToolTip("" if (human_compatible and archery_compatible) else "需同时满足人体=26点，弓箭=5点")

        extra = []
        if not human_compatible:
            extra.append(f"人体标签{len(self._human_keypoint_labels)}点")
        if not archery_compatible:
            extra.append(f"弓箭标签{len(self._archery_keypoint_labels)}点")

        if extra:
            self.auto_status_label.setText(f"{self._model_status_base} | 自动标注受限: {'/'.join(extra)}")
        else:
            self.auto_status_label.setText(self._model_status_base)

    def _map_points_to_active_names(self, points: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for idx, point in enumerate(points):
            item = dict(point)
            if idx < len(names):
                item["name"] = names[idx]
            mapped.append(item)
        return mapped

    def _run_human_auto_annotation_for_current(self, show_message: bool = True) -> bool:
        if len(self._human_keypoint_labels) != 26:
            if show_message:
                QMessageBox.warning(self, "不可用", "人体自动预标注要求人体关键点标签数量为26。")
            return False

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
            iou_threshold=self.iou_spin.value(),
        )

        try:
            result = self._human_auto_annotator.predict_image(self._current_image_path)
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "人体预标注失败", str(exc))
            return False

        mapped = self._map_points_to_active_names(result.points, self._human_keypoint_labels)
        self.canvas.set_human_points(mapped)
        self._refresh_counts_and_lists()
        self._save_current_annotation(show_message=False)
        self._update_mismatch_status_for_image(self._current_image_path.name)

        if show_message:
            QMessageBox.information(
                self,
                "人体预标注完成",
                (
                    f"图片: {self._current_image_path.name}\n"
                    f"原始关键点: {result.raw_count}\n"
                    f"保留关键点: {len(mapped)}\n"
                    f"置信度阈值: {self.conf_spin.value():.2f}\n"
                    f"IoU阈值: {self.iou_spin.value():.2f}"
                ),
            )

        return True

    def _run_archery_auto_annotation_for_current(self, show_message: bool = True) -> bool:
        if len(self._archery_keypoint_labels) != 5:
            if show_message:
                QMessageBox.warning(self, "不可用", "弓箭自动预标注要求弓箭关键点标签数量为5。")
            return False

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
            iou_threshold=self.iou_spin.value(),
        )

        try:
            result = self._archery_auto_annotator.predict_image(self._current_image_path)
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "弓箭预标注失败", str(exc))
            return False

        mapped = self._map_points_to_active_names(result.points, self._archery_keypoint_labels)
        self.canvas.set_archery_points(mapped)
        self._refresh_counts_and_lists()
        self._save_current_annotation(show_message=False)
        self._update_mismatch_status_for_image(self._current_image_path.name)

        if show_message:
            QMessageBox.information(
                self,
                "弓箭预标注完成",
                (
                    f"图片: {self._current_image_path.name}\n"
                    f"原始检测数: {result.raw_detection_count}\n"
                    f"IoU后检测数: {result.kept_detection_count}\n"
                    f"输出关键点: {len(mapped)}\n"
                    f"置信度阈值: {self.conf_spin.value():.2f}\n"
                    f"IoU阈值: {self.iou_spin.value():.2f}"
                ),
            )

        return True

    def _run_auto_annotation_for_all_images(self) -> None:
        if not self.auto_all_btn.isEnabled():
            QMessageBox.warning(self, "不可用", "当前标签集合与自动预标注模型不兼容。")
            return

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

    def _on_canvas_cursor_moved(self, x: float, y: float, inside: bool) -> None:
        if inside:
            self._update_cursor_status(int(round(x)), int(round(y)))
        else:
            self._update_cursor_status(None, None)

    def _update_cursor_status(self, x: int | None, y: int | None) -> None:
        x_text = "-" if x is None else str(x)
        y_text = "-" if y is None else str(y)

        name = "-"
        idx = 0
        total = self.image_list.count()
        if self._current_image_path is not None:
            name = self._current_image_path.name
            idx = self.image_list.currentRow() + 1 if self.image_list.currentRow() >= 0 else 0

        self.cursor_status_label.setText(f"(X:{x_text},Y:{y_text})[图片名称:{name} {idx}/{total}]")

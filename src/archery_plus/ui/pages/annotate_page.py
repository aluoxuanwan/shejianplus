from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
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

from archery_plus.config import DEFAULT_PROJECT_DIR
from archery_plus.data.annotation_store import AnnotationStore
from archery_plus.services.model_registry import collect_artifacts
from archery_plus.ui.widgets.annotation_canvas import AnnotationCanvas

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHERY_KEYPOINTS = ["UP", "DOWN", "FL", "ST", "FS"]


class AnnotatePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_root = Path(DEFAULT_PROJECT_DIR)
        self._store = AnnotationStore(self._project_root)
        self._current_image_path: Path | None = None
        self._sidebar_width = 280
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QLabel("标注模块：点击一个点后自动保存，并自动跳转到下一张图片。")
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

        left_layout.addWidget(QLabel("图片列表（images/cam1）"))
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
        kp_row.addWidget(QLabel("当前关键点:"))
        self.kp_combo = QComboBox()
        self.kp_combo.addItems(ARCHERY_KEYPOINTS)
        self.kp_combo.currentTextChanged.connect(self._on_keypoint_changed)
        kp_row.addWidget(self.kp_combo)

        self.points_status_label = QLabel("当前点数: 0")
        kp_row.addWidget(self.points_status_label)

        kp_row.addWidget(QLabel("左键新增/拖拽，右键删除最近点"))
        kp_row.addStretch(1)
        right_layout.addLayout(kp_row)

        self.canvas = AnnotationCanvas()
        self.canvas.set_current_keypoint_name(self.kp_combo.currentText())
        self.canvas.pointsChanged.connect(self._on_points_changed)
        self.canvas.pointAdded.connect(self._on_point_added_auto_next)
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self._sidebar_width, 1700])

        root.addWidget(splitter, stretch=1)

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
            self.points_status_label.setText("当前点数: 0")

    def _on_image_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        _ = previous
        if current is None:
            self._current_image_path = None
            self.canvas.clear_image()
            self.points_status_label.setText("当前点数: 0")
            return

        path = Path(current.data(Qt.ItemDataRole.UserRole))
        image = QImage(str(path))
        if image.isNull():
            QMessageBox.warning(self, "图片加载失败", f"无法读取图片: {path}")
            return

        self._current_image_path = path
        self.canvas.set_image(image)
        points = self._store.get_image_points(path.name)
        self.canvas.set_points(points)
        self.points_status_label.setText(f"当前点数: {len(points)}")

    def _on_keypoint_changed(self, name: str) -> None:
        self.canvas.set_current_keypoint_name(name)

    def _on_points_changed(self) -> None:
        self.points_status_label.setText(f"当前点数: {len(self.canvas.get_points())}")

    def _on_point_added_auto_next(self) -> None:
        self._save_current_annotation(show_message=False)
        self._goto_next_image()

    def _goto_next_image(self) -> None:
        row = self.image_list.currentRow()
        if row < 0:
            return

        next_row = row + 1
        if next_row < self.image_list.count():
            self.image_list.setCurrentRow(next_row)
        else:
            self.points_status_label.setText(
                f"当前点数: {len(self.canvas.get_points())}（已到最后一张）"
            )

    def _save_current_annotation(self, show_message: bool = True) -> None:
        if self._current_image_path is None:
            if show_message:
                QMessageBox.information(self, "提示", "请先选择图片。")
            return

        points = self.canvas.get_points()
        self._store.set_image_points(self._current_image_path.name, points)

        if show_message:
            QMessageBox.information(
                self,
                "保存成功",
                f"已保存 {self._current_image_path.name} 的标注。\n关键点数量: {len(points)}",
            )

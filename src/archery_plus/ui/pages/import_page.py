from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from archery_plus.config import DEFAULT_PROJECT_DIR
from archery_plus.core.project_initializer import initialize_project_structure
from archery_plus.pipelines.frame_extractor import extract_video_frames


class ImportPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        project_box = QGroupBox("项目初始化")
        project_form = QFormLayout(project_box)

        project_selector = QHBoxLayout()
        project_selector.setSpacing(8)

        self.project_path_edit = QLineEdit(str(DEFAULT_PROJECT_DIR))
        self.project_path_edit.setPlaceholderText("选择或输入项目目录")
        self.project_path_edit.setMinimumWidth(520)
        self.project_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        project_selector.addWidget(self.project_path_edit, stretch=1)

        choose_project_button = QPushButton("浏览")
        choose_project_button.setFixedWidth(88)
        choose_project_button.clicked.connect(self._select_project_dir)
        project_selector.addWidget(choose_project_button)

        self.init_button = QPushButton("创建项目目录结构")
        self.init_button.clicked.connect(self._initialize_project)

        project_form.addRow("项目目录", project_selector)
        project_form.addRow("", self.init_button)

        import_box = QGroupBox("导入视频并抽帧")
        import_form = QFormLayout(import_box)
        video_selector = QHBoxLayout()
        video_selector.setSpacing(8)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择 MP4/AVI 文件")
        self.video_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        video_selector.addWidget(self.video_path_edit, stretch=1)

        choose_video_button = QPushButton("选择视频")
        choose_video_button.setFixedWidth(88)
        choose_video_button.clicked.connect(self._select_video_file)
        video_selector.addWidget(choose_video_button)

        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 30)
        self.frame_interval_spin.setValue(5)

        self.extract_button = QPushButton("开始抽帧")
        self.extract_button.clicked.connect(self._extract_frames)

        import_form.addRow("视频文件", video_selector)
        import_form.addRow("抽帧间隔", self.frame_interval_spin)
        import_form.addRow("", self.extract_button)

        self.log_label = QLabel("操作日志")
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("这里会显示导入/初始化日志。")

        root_layout.addWidget(project_box)
        root_layout.addWidget(import_box)
        root_layout.addWidget(self.log_label)
        root_layout.addWidget(self.log_edit)

    def _append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    def _select_project_dir(self) -> None:
        current = self.project_path_edit.text().strip() or str(DEFAULT_PROJECT_DIR)
        selected = QFileDialog.getExistingDirectory(self, "选择项目目录", current)
        if selected:
            self.project_path_edit.setText(selected)

    def _select_video_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi);;All Files (*)",
        )
        if selected:
            self.video_path_edit.setText(selected)
            self._append_log(f"已选择视频: {selected}")

    def _initialize_project(self) -> None:
        raw_path = self.project_path_edit.text().strip()
        if not raw_path:
            QMessageBox.warning(self, "路径为空", "请先填写项目目录。")
            return

        project_root = Path(raw_path)
        created_dirs = initialize_project_structure(project_root)
        self._append_log(f"项目初始化完成: {project_root}")
        for path in created_dirs:
            self._append_log(f"  - {path}")

    def _extract_frames(self) -> None:
        project_root_text = self.project_path_edit.text().strip()
        video_path_text = self.video_path_edit.text().strip()
        interval = self.frame_interval_spin.value()

        if not project_root_text:
            QMessageBox.information(self, "提示", "请先设置项目目录。")
            return
        if not video_path_text:
            QMessageBox.information(self, "提示", "请先选择视频文件。")
            return

        project_root = Path(project_root_text)
        video_path = Path(video_path_text)

        initialize_project_structure(project_root)
        output_dir = project_root / "images" / "cam1"

        self._append_log(f"准备抽帧: {video_path}")
        self._append_log(f"输出目录: {output_dir}")
        self._append_log(f"抽帧间隔: 每 {interval} 帧")

        try:
            result = extract_video_frames(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=interval,
            )
        except Exception as exc:
            self._append_log(f"抽帧失败: {exc}")
            QMessageBox.critical(self, "抽帧失败", str(exc))
            return

        self._append_log(
            f"抽帧完成: 总帧数={result.total_frames}, 保存帧数={result.saved_frames}"
        )
        QMessageBox.information(
            self,
            "抽帧完成",
            f"已完成抽帧。\n总帧数: {result.total_frames}\n保存帧数: {result.saved_frames}",
        )

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExportPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        desc = QLabel("导出模块 MVP：先定义导出选项和流程入口。")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        options_box = QGroupBox("导出选项")
        options_layout = QVBoxLayout(options_box)
        self.raw_2d = QCheckBox("原始 2D 坐标 CSV")
        self.raw_2d.setChecked(True)
        self.filtered_2d = QCheckBox("滤波后 2D 坐标 CSV")
        self.filtered_2d.setChecked(True)
        self.coords_3d = QCheckBox("3D 坐标 CSV（下一阶段）")
        self.overlay_video = QCheckBox("关键点叠加视频（下一阶段）")
        options_layout.addWidget(self.raw_2d)
        options_layout.addWidget(self.filtered_2d)
        options_layout.addWidget(self.coords_3d)
        options_layout.addWidget(self.overlay_video)
        layout.addWidget(options_box)

        self.start_export = QPushButton("开始导出（待接入）")
        self.start_export.setEnabled(False)
        layout.addWidget(self.start_export)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("导出日志（占位）。")
        layout.addWidget(self.log)

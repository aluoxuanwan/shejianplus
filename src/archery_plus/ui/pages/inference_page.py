from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class InferencePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        desc = QLabel("应用模块 MVP：接入实时推理前，先固定参数面板与实时状态区。")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        param_box = QGroupBox("推理参数")
        form = QFormLayout(param_box)

        self.height_cm = QDoubleSpinBox()
        self.height_cm.setRange(100, 230)
        self.height_cm.setValue(175.0)
        self.height_cm.setSuffix(" cm")

        self.ema_alpha = QDoubleSpinBox()
        self.ema_alpha.setRange(0.0, 1.0)
        self.ema_alpha.setSingleStep(0.05)
        self.ema_alpha.setValue(0.70)

        form.addRow("受试者身高", self.height_cm)
        form.addRow("EMA 平滑系数", self.ema_alpha)
        layout.addWidget(param_box)

        self.start_btn = QPushButton("开始检测（待接入）")
        self.stop_btn = QPushButton("停止检测（待接入）")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.monitor = QTextEdit()
        self.monitor.setReadOnly(True)
        self.monitor.setPlaceholderText("推理状态/FPS/关键点日志（占位）。")
        layout.addWidget(self.monitor)

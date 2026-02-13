from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel("训练模块 MVP：先提供参数入口与命令占位，下一步接入 MMPose 训练脚本。")
        info.setWordWrap(True)
        layout.addWidget(info)

        config_box = QGroupBox("训练配置")
        form = QFormLayout(config_box)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2000)
        self.epochs.setValue(600)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(32)

        form.addRow("Epochs", self.epochs)
        form.addRow("Batch Size", self.batch_size)
        layout.addWidget(config_box)

        self.start_btn = QPushButton("开始训练（待接入）")
        self.stop_btn = QPushButton("停止训练（待接入）")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志输出区域（占位）。")
        layout.addWidget(self.log)

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from shejianplus.config import APP_NAME, APP_VERSION
from shejianplus.services.model_registry import count_ready_models
from shejianplus.services.runtime_probe import probe_runtime
from shejianplus.ui.pages import (
    AnnotatePage,
    ExportPage,
    ImportPage,
    InferencePage,
    TrainPage,
)


class MainWindow(QMainWindow):
    MODULE_NAMES = ["导入", "标注", "训练", "应用", "导出"]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        # 2560x1440 同比例（16:9）
        self.resize(1600, 900)

        self._runtime_info = probe_runtime()
        self._build_ui()
        self._apply_status()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        self.header_label = QLabel(self.MODULE_NAMES[0])
        self.header_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(self.header_label)

        content = QHBoxLayout()
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(50)
        self.nav_list.addItems(self.MODULE_NAMES)
        self.nav_list.currentRowChanged.connect(self._on_page_changed)

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(ImportPage())
        self.page_stack.addWidget(AnnotatePage())
        self.page_stack.addWidget(TrainPage())
        self.page_stack.addWidget(InferencePage())
        self.page_stack.addWidget(ExportPage())

        content.addWidget(self.nav_list)
        content.addWidget(self.page_stack, stretch=1)
        root.addLayout(content)

        self.setCentralWidget(central)
        self.nav_list.setCurrentRow(0)

        status_bar = QStatusBar()
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)

    def _on_page_changed(self, index: int) -> None:
        if index < 0:
            return
        self.page_stack.setCurrentIndex(index)
        self.header_label.setText(self.MODULE_NAMES[index])

    def _apply_status(self) -> None:
        ready, total = count_ready_models()
        providers = self._runtime_info.onnx_providers
        provider_text = ",".join(providers) if providers else "无"
        status_text = (
            f"ONNX Providers: {provider_text} | "
            f"OpenCV CUDA设备数: {self._runtime_info.cuda_device_count} | "
            f"ONNX模型就绪: {ready}/{total}"
        )
        self.statusBar().showMessage(status_text)

        if self._runtime_info.notes:
            note_label = QLabel(" | ".join(self._runtime_info.notes))
            note_label.setTextFormat(Qt.TextFormat.PlainText)
            self.statusBar().addPermanentWidget(note_label)

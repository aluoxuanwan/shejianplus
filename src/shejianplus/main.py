from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from shejianplus.config import APP_NAME
from shejianplus.ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    win = MainWindow()
    win.setWindowState(win.windowState() | Qt.WindowState.WindowMaximized)
    win.show()
    return app.exec()

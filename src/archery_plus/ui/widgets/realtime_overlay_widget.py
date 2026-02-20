from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QSize
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class RealtimeOverlayWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._frame_rgb: np.ndarray | None = None
        self._qimage: QImage | None = None
        self._points: list[dict[str, Any]] = []
        self._status_text: str = ""
        self._highlight_names = {"UP", "DOWN"}

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return int(width * 9 / 16)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(1280, 720)

    def set_frame(self, frame_bgr: np.ndarray, points: list[dict[str, Any]]) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            self.clear()
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_rgb = rgb
        h, w = rgb.shape[:2]
        bytes_per_line = int(3 * w)
        self._qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._points = [dict(p) for p in points if isinstance(p, dict)]
        self.update()

    def set_overlay_status(self, text: str) -> None:
        self._status_text = str(text).strip()
        self.update()

    def clear(self) -> None:
        self._frame_rgb = None
        self._qimage = None
        self._points = []
        self._status_text = ""
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor(12, 12, 12))
        viewport = self._fit_rect_16_9()
        painter.fillRect(viewport, QColor(0, 0, 0))

        if self._qimage is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(viewport, Qt.AlignmentFlag.AlignCenter, "请选择视频并开始推理")
            return

        image_rect = self._fit_image_in_viewport(viewport, self._qimage.width(), self._qimage.height())
        pix = QPixmap.fromImage(self._qimage)
        painter.drawPixmap(image_rect, pix, QRectF(self._qimage.rect()))

        self._draw_points(painter, image_rect)
        self._draw_top_status(painter, viewport)

    def _draw_points(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._qimage is None:
            return

        iw = float(max(self._qimage.width(), 1))
        ih = float(max(self._qimage.height(), 1))

        ring_color = QColor(255, 176, 59)
        fill_color = QColor(255, 140, 48)

        for point in self._points:
            try:
                x = float(point.get("x", 0.0))
                y = float(point.get("y", 0.0))
            except Exception:
                continue

            px = image_rect.x() + (x / iw) * image_rect.width()
            py = image_rect.y() + (y / ih) * image_rect.height()
            center = QPointF(px, py)

            painter.setPen(QPen(QColor(18, 18, 18, 220), 2.4))
            painter.setBrush(QColor(18, 18, 18, 80))
            painter.drawEllipse(center, 6.0, 6.0)

            painter.setPen(QPen(ring_color, 1.2))
            painter.setBrush(fill_color)
            painter.drawEllipse(center, 4.0, 4.0)

            name = str(point.get("name", "")).strip().upper()
            if name in self._highlight_names:
                self._draw_label_chip(painter, px + 6.0, py - 8.0, name)

    def _draw_label_chip(self, painter: QPainter, x: float, y: float, text: str) -> None:
        font = QFont(painter.font())
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        pad_x = 5
        pad_y = 2
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()

        rect = QRectF(x, y - text_h, text_w + pad_x * 2, text_h + pad_y * 2)
        painter.setPen(QPen(QColor(255, 176, 59), 1.0))
        painter.setBrush(QColor(16, 18, 22, 190))
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QColor(244, 246, 250))
        painter.drawText(rect.adjusted(pad_x, pad_y, -pad_x, -pad_y), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

    def _draw_top_status(self, painter: QPainter, viewport: QRectF) -> None:
        if not self._status_text:
            return

        rect = QRectF(viewport.x() + 8, viewport.y() + 8, min(620.0, viewport.width() - 16), 30.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 28, 32, 200))
        painter.drawRoundedRect(rect, 6, 6)

        painter.setPen(QColor(236, 245, 249))
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._status_text)

    def _fit_rect_16_9(self) -> QRectF:
        w = float(max(self.width(), 1))
        h = float(max(self.height(), 1))
        target_ratio = 16.0 / 9.0
        current_ratio = w / h

        if current_ratio >= target_ratio:
            draw_h = h
            draw_w = h * target_ratio
        else:
            draw_w = w
            draw_h = w / target_ratio

        cx = w * 0.5
        cy = h * 0.5
        return QRectF(cx - draw_w * 0.5, cy - draw_h * 0.5, draw_w, draw_h)

    def _fit_image_in_viewport(self, viewport: QRectF, img_w: int, img_h: int) -> QRectF:
        vw = viewport.width()
        vh = viewport.height()
        iw = float(max(img_w, 1))
        ih = float(max(img_h, 1))

        image_ratio = iw / ih
        viewport_ratio = vw / vh

        if viewport_ratio >= image_ratio:
            draw_h = vh
            draw_w = vh * image_ratio
        else:
            draw_w = vw
            draw_h = vw / image_ratio

        cx = viewport.x() + vw * 0.5
        cy = viewport.y() + vh * 0.5
        return QRectF(cx - draw_w * 0.5, cy - draw_h * 0.5, draw_w, draw_h)

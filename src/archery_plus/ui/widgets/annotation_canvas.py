from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class AnnotationCanvas(QWidget):
    pointsChanged = Signal()
    pointAdded = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(1280, 720)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._viewport_rect: QRectF | None = None
        self._image_draw_rect: QRectF | None = None
        self._points: list[dict[str, Any]] = []
        self._selected_index: int | None = None
        self._dragging = False
        self._current_keypoint_name = "UP"
        self._point_radius = 6.0

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return int(width * 9 / 16)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(1280, 720)

    def set_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._selected_index = None
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self._points = []
        self._selected_index = None
        self.update()

    def set_points(self, points: list[dict[str, Any]]) -> None:
        self._points = points.copy()
        self._selected_index = None
        self.update()

    def get_points(self) -> list[dict[str, Any]]:
        return self._points.copy()

    def set_current_keypoint_name(self, name: str) -> None:
        self._current_keypoint_name = name

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 12))

        self._viewport_rect = self._fit_rect_16_9()
        painter.fillRect(self._viewport_rect, QColor(0, 0, 0))

        if self._pixmap is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self._viewport_rect, Qt.AlignCenter, "请先加载图片")
            return

        self._image_draw_rect = self._fit_image_in_viewport(self._viewport_rect)
        painter.drawPixmap(self._image_draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        for idx, pt in enumerate(self._points):
            x_disp, y_disp = self._image_to_display(float(pt["x"]), float(pt["y"]))
            is_selected = idx == self._selected_index
            color = QColor(255, 80, 80) if is_selected else QColor(80, 220, 120)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x_disp, y_disp), self._point_radius, self._point_radius)
            painter.setPen(QColor(240, 240, 240))
            painter.drawText(int(x_disp + 8), int(y_disp - 8), str(pt.get("name", "KP")))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._pixmap is None or self._image_draw_rect is None:
            return

        pos = event.position()

        if event.button() == Qt.LeftButton:
            hit = self._find_nearest_display_point(pos, threshold=10.0)
            if hit is not None:
                self._selected_index = hit
                self._dragging = True
                self.update()
                return

            image_pos = self._display_to_image(pos.x(), pos.y())
            if image_pos is None:
                return

            x_img, y_img = image_pos
            self._points.append(
                {
                    "name": self._current_keypoint_name,
                    "x": round(x_img, 2),
                    "y": round(y_img, 2),
                    "v": 2,
                }
            )
            self._selected_index = len(self._points) - 1
            self.pointsChanged.emit()
            self.pointAdded.emit()
            self.update()

        elif event.button() == Qt.RightButton:
            hit = self._find_nearest_display_point(pos, threshold=10.0)
            if hit is not None:
                self._points.pop(hit)
                self._selected_index = None
                self.pointsChanged.emit()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._dragging or self._selected_index is None:
            return

        image_pos = self._display_to_image(event.position().x(), event.position().y())
        if image_pos is None:
            return

        x_img, y_img = image_pos
        self._points[self._selected_index]["x"] = round(x_img, 2)
        self._points[self._selected_index]["y"] = round(y_img, 2)
        self.pointsChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._dragging = False

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

        cx = w / 2.0
        cy = h / 2.0
        x = cx - draw_w / 2.0
        y = cy - draw_h / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def _fit_image_in_viewport(self, viewport: QRectF) -> QRectF:
        assert self._pixmap is not None

        vw = viewport.width()
        vh = viewport.height()
        iw = float(self._pixmap.width())
        ih = float(self._pixmap.height())

        image_ratio = iw / ih
        viewport_ratio = vw / vh

        if viewport_ratio >= image_ratio:
            draw_h = vh
            draw_w = vh * image_ratio
        else:
            draw_w = vw
            draw_h = vw / image_ratio

        cx = viewport.x() + vw / 2.0
        cy = viewport.y() + vh / 2.0
        x = cx - draw_w / 2.0
        y = cy - draw_h / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def _display_to_image(self, x_disp: float, y_disp: float) -> tuple[float, float] | None:
        if self._pixmap is None or self._image_draw_rect is None:
            return None

        rect = self._image_draw_rect
        if not rect.contains(QPointF(x_disp, y_disp)):
            return None

        rel_x = (x_disp - rect.x()) / rect.width()
        rel_y = (y_disp - rect.y()) / rect.height()
        return rel_x * self._pixmap.width(), rel_y * self._pixmap.height()

    def _image_to_display(self, x_img: float, y_img: float) -> tuple[float, float]:
        assert self._pixmap is not None
        assert self._image_draw_rect is not None

        rel_x = x_img / self._pixmap.width()
        rel_y = y_img / self._pixmap.height()

        x_disp = self._image_draw_rect.x() + rel_x * self._image_draw_rect.width()
        y_disp = self._image_draw_rect.y() + rel_y * self._image_draw_rect.height()
        return x_disp, y_disp

    def _find_nearest_display_point(self, pos: QPointF, threshold: float) -> int | None:
        if not self._points or self._pixmap is None or self._image_draw_rect is None:
            return None

        best_idx: int | None = None
        best_dist2 = threshold * threshold

        for idx, pt in enumerate(self._points):
            x_disp, y_disp = self._image_to_display(float(pt["x"]), float(pt["y"]))
            dx = x_disp - pos.x()
            dy = y_disp - pos.y()
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist2:
                best_dist2 = dist2
                best_idx = idx

        return best_idx

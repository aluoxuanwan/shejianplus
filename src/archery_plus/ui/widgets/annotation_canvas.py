from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class AnnotationCanvas(QWidget):
    pointsChanged = Signal()
    pointAdded = Signal()
    pointRemoved = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(1280, 720)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._viewport_rect: QRectF | None = None
        self._image_draw_rect: QRectF | None = None

        self._archery_points: list[dict[str, Any]] = []
        self._human_points: list[dict[str, Any]] = []
        self._show_human_points = True
        self._edit_group = "archery"

        self._selected_index: int | None = None
        self._dragging = False
        self._current_keypoint_name = "UP"
        self._point_radius = 4.0

        self._hovered_group: str | None = None
        self._hovered_index: int | None = None

        self.setMouseTracking(True)

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return int(width * 9 / 16)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(1280, 720)

    def set_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._selected_index = None
        self._clear_hover()
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self._archery_points = []
        self._human_points = []
        self._selected_index = None
        self._clear_hover()
        self.update()

    def set_points(self, points: list[dict[str, Any]]) -> None:
        self.set_archery_points(points)

    def get_points(self) -> list[dict[str, Any]]:
        return self.get_archery_points()

    def set_archery_points(self, points: list[dict[str, Any]]) -> None:
        self._archery_points = points.copy()
        self._selected_index = None
        self.update()

    def get_archery_points(self) -> list[dict[str, Any]]:
        return self._archery_points.copy()

    def set_human_points(self, points: list[dict[str, Any]]) -> None:
        self._human_points = points.copy()
        self._selected_index = None
        self.update()

    def get_human_points(self) -> list[dict[str, Any]]:
        return self._human_points.copy()

    def set_show_human_points(self, show: bool) -> None:
        self._show_human_points = bool(show)
        self.update()

    def set_edit_group(self, group: str) -> None:
        normalized = "human" if group == "human" else "archery"
        if self._edit_group != normalized:
            self._edit_group = normalized
            self._selected_index = None
            self._dragging = False
            self.update()

    def get_edit_group(self) -> str:
        return self._edit_group

    def set_current_keypoint_name(self, name: str) -> None:
        self._current_keypoint_name = name

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor(12, 12, 12))

        self._viewport_rect = self._fit_rect_16_9()
        painter.fillRect(self._viewport_rect, QColor(0, 0, 0))

        if self._pixmap is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self._viewport_rect, Qt.AlignCenter, "请先加载图片")
            return

        self._image_draw_rect = self._fit_image_in_viewport(self._viewport_rect)
        painter.drawPixmap(self._image_draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        self._draw_points(painter, self._archery_points, base=QColor(255, 190, 70), group="archery")
        if self._show_human_points:
            self._draw_points(painter, self._human_points, base=QColor(90, 190, 255), group="human")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._pixmap is None or self._image_draw_rect is None:
            return

        pos = event.position()
        editable_points = self._editable_points()

        if event.button() == Qt.LeftButton:
            hit = self._find_nearest_display_point(pos, editable_points, threshold=12.0)
            if hit is not None:
                self._selected_index = hit
                self._dragging = True
                self.update()
                return

            image_pos = self._display_to_image(pos.x(), pos.y())
            if image_pos is None:
                return

            x_img, y_img = image_pos
            editable_points.append(
                {
                    "name": self._current_keypoint_name,
                    "x": round(x_img, 2),
                    "y": round(y_img, 2),
                    "v": 2,
                    "score": 0.0,
                }
            )
            self._selected_index = len(editable_points) - 1
            self.pointsChanged.emit()
            self.pointAdded.emit()
            self.update()

        elif event.button() == Qt.RightButton:
            hit_group: str | None = self._hovered_group
            hit_idx: int | None = self._hovered_index
            if hit_group is None or hit_idx is None:
                hit_group, hit_idx = self._find_nearest_visible_point(pos, threshold=12.0)

            if hit_group is not None and hit_idx is not None:
                target_points = self._points_of_group(hit_group)
                if 0 <= hit_idx < len(target_points):
                    target_points.pop(hit_idx)
                    if self._edit_group == hit_group:
                        self._selected_index = None
                    self._clear_hover()
                    self.pointsChanged.emit()
                    self.pointRemoved.emit()
                    self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and self._selected_index is not None:
            image_pos = self._display_to_image(event.position().x(), event.position().y())
            if image_pos is None:
                return

            editable_points = self._editable_points()
            if self._selected_index < 0 or self._selected_index >= len(editable_points):
                return

            x_img, y_img = image_pos
            editable_points[self._selected_index]["x"] = round(x_img, 2)
            editable_points[self._selected_index]["y"] = round(y_img, 2)
            self.pointsChanged.emit()
            self.update()
            return

        self._update_hover(event.position())

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        self._clear_hover()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def _draw_points(self, painter: QPainter, points: list[dict[str, Any]], base: QColor, group: str) -> None:
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)

        for idx, pt in enumerate(points):
            x_disp, y_disp = self._image_to_display(float(pt["x"]), float(pt["y"]))
            center = QPointF(x_disp, y_disp)
            selected = self._edit_group == group and idx == self._selected_index
            hovered = self._hovered_group == group and idx == self._hovered_index

            ring_radius = self._point_radius + (2.0 if selected else 0.0)
            fill_radius = self._point_radius - 2.2 + (1.0 if selected else 0.0)

            ring_color = QColor(255, 95, 75) if selected else base
            fill_color = QColor(ring_color)
            fill_color.setAlpha(220)

            painter.setPen(QPen(QColor(20, 20, 20, 220), 3.0))
            painter.setBrush(QColor(20, 20, 20, 90))
            painter.drawEllipse(center, ring_radius + 2.0, ring_radius + 2.0)

            painter.setPen(QPen(QColor(250, 250, 250, 230), 1.4))
            painter.setBrush(ring_color)
            painter.drawEllipse(center, ring_radius, ring_radius)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawEllipse(center, fill_radius, fill_radius)

            if hovered:
                label_text = str(pt.get("name", "KP"))
                self._draw_label_chip(
                    painter,
                    x_disp + ring_radius + 6.0,
                    y_disp - ring_radius - 4.0,
                    label_text,
                    ring_color,
                )

    def _draw_label_chip(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        border_color: QColor,
    ) -> None:
        fm = painter.fontMetrics()
        pad_x = 6
        pad_y = 3
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()

        rect = QRectF(x, y - text_h, text_w + pad_x * 2, text_h + pad_y * 2)

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QColor(16, 18, 22, 185))
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QColor(244, 246, 250))
        painter.drawText(rect.adjusted(pad_x, pad_y, -pad_x, -pad_y), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _editable_points(self) -> list[dict[str, Any]]:
        return self._human_points if self._edit_group == "human" else self._archery_points

    def _points_of_group(self, group: str) -> list[dict[str, Any]]:
        return self._human_points if group == "human" else self._archery_points

    def _find_nearest_visible_point(self, pos: QPointF, threshold: float) -> tuple[str | None, int | None]:
        best_group: str | None = None
        best_idx: int | None = None
        best_dist2 = threshold * threshold

        candidates: list[tuple[str, list[dict[str, Any]]]] = [("archery", self._archery_points)]
        if self._show_human_points:
            candidates.append(("human", self._human_points))

        for group, points in candidates:
            idx = self._find_nearest_display_point(pos, points, threshold=threshold)
            if idx is None:
                continue
            x_disp, y_disp = self._image_to_display(float(points[idx]["x"]), float(points[idx]["y"]))
            dx = x_disp - pos.x()
            dy = y_disp - pos.y()
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist2:
                best_dist2 = dist2
                best_group = group
                best_idx = idx

        return best_group, best_idx

    def _update_hover(self, pos: QPointF) -> None:
        best_group, best_idx = self._find_nearest_visible_point(pos, threshold=14.0)

        if best_group != self._hovered_group or best_idx != self._hovered_index:
            self._hovered_group = best_group
            self._hovered_index = best_idx
            self.update()

    def _clear_hover(self) -> None:
        self._hovered_group = None
        self._hovered_index = None

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

    def _find_nearest_display_point(
        self,
        pos: QPointF,
        points: list[dict[str, Any]],
        threshold: float,
    ) -> int | None:
        if not points or self._pixmap is None or self._image_draw_rect is None:
            return None

        best_idx: int | None = None
        best_dist2 = threshold * threshold

        for idx, pt in enumerate(points):
            x_disp, y_disp = self._image_to_display(float(pt["x"]), float(pt["y"]))
            dx = x_disp - pos.x()
            dy = y_disp - pos.y()
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist2:
                best_dist2 = dist2
                best_idx = idx

        return best_idx


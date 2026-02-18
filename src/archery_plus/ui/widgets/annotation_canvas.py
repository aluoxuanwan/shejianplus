from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class AnnotationCanvas(QWidget):
    pointsChanged = Signal()
    pointAdded = Signal()
    pointRemoved = Signal()
    boxAdded = Signal()
    boxRemoved = Signal()
    imageCursorMoved = Signal(float, float, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(1280, 720)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._viewport_rect: QRectF | None = None
        self._image_draw_rect: QRectF | None = None

        self._archery_points: list[dict[str, Any]] = []
        self._human_points: list[dict[str, Any]] = []
        self._bboxes: list[dict[str, Any]] = []

        self._edit_group = "archery"
        self._edit_mode = "keypoint"  # keypoint | bbox
        self._annotation_enabled = True

        self._selected_index: int | None = None
        self._dragging = False
        self._current_keypoint_name = "UP"
        self._point_radius = 4.0

        self._current_bbox_name = "object"
        self._selected_box_index: int | None = None
        self._dragging_box = False
        self._box_drag_anchor: tuple[float, float] | None = None

        self._drawing_box = False
        self._new_box_start_img: tuple[float, float] | None = None
        self._new_box_current_img: tuple[float, float] | None = None

        self._hovered_group: str | None = None
        self._hovered_index: int | None = None
        self._hovered_box_index: int | None = None

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
        self._selected_box_index = None
        self._clear_hover()
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self._archery_points = []
        self._human_points = []
        self._bboxes = []
        self._selected_index = None
        self._selected_box_index = None
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

    def set_bboxes(self, bboxes: list[dict[str, Any]]) -> None:
        self._bboxes = bboxes.copy()
        self._selected_box_index = None
        self.update()

    def get_bboxes(self) -> list[dict[str, Any]]:
        return self._bboxes.copy()

    def set_edit_group(self, group: str) -> None:
        normalized = "human" if group == "human" else "archery"
        if self._edit_group != normalized:
            self._edit_group = normalized
            self._selected_index = None
            self._dragging = False
            self.update()

    def get_edit_group(self) -> str:
        return self._edit_group

    def set_edit_mode(self, mode: str) -> None:
        normalized = "bbox" if mode == "bbox" else "keypoint"
        if self._edit_mode != normalized:
            self._edit_mode = normalized
            self._selected_index = None
            self._selected_box_index = None
            self._dragging = False
            self._dragging_box = False
            self._drawing_box = False
            self._new_box_start_img = None
            self._new_box_current_img = None
            self.update()

    def get_edit_mode(self) -> str:
        return self._edit_mode

    def set_annotation_enabled(self, enabled: bool) -> None:
        self._annotation_enabled = bool(enabled)

    def set_current_keypoint_name(self, name: str) -> None:
        self._current_keypoint_name = name

    def set_current_bbox_name(self, name: str) -> None:
        self._current_bbox_name = name

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

        self._draw_bboxes(painter)
        self._draw_points(painter, self._archery_points, base=QColor(255, 190, 70), group="archery")
        self._draw_points(painter, self._human_points, base=QColor(90, 190, 255), group="human")
        self._draw_temp_bbox(painter)

        if not self._annotation_enabled:
            painter.fillRect(self._viewport_rect, QColor(20, 20, 20, 70))
            painter.setPen(QColor(255, 245, 210))
            painter.drawText(self._viewport_rect, Qt.AlignCenter, "请先导入或新增标签，再进行标注")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._pixmap is None or self._image_draw_rect is None:
            return

        pos = event.position()
        self._emit_cursor(pos)

        if not self._annotation_enabled:
            return

        if self._edit_mode == "bbox":
            self._mouse_press_bbox(event)
        else:
            self._mouse_press_keypoint(event)

    def _mouse_press_keypoint(self, event: QMouseEvent) -> None:
        if self._image_draw_rect is None:
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

    def _mouse_press_bbox(self, event: QMouseEvent) -> None:
        pos = event.position()
        if event.button() == Qt.LeftButton:
            hit = self._find_bbox_hit(pos)
            if hit is not None:
                self._selected_box_index = hit
                self._dragging_box = True
                image_pos = self._display_to_image(pos.x(), pos.y())
                if image_pos is not None:
                    box = self._bboxes[hit]
                    self._box_drag_anchor = (image_pos[0] - float(box["x1"]), image_pos[1] - float(box["y1"]))
                self.update()
                return

            image_pos = self._display_to_image(pos.x(), pos.y())
            if image_pos is None:
                return
            self._drawing_box = True
            self._new_box_start_img = image_pos
            self._new_box_current_img = image_pos
            self._selected_box_index = None
            self.update()

        elif event.button() == Qt.RightButton:
            hit = self._hovered_box_index
            if hit is None:
                hit = self._find_bbox_hit(pos)
            if hit is not None and 0 <= hit < len(self._bboxes):
                self._bboxes.pop(hit)
                self._selected_box_index = None
                self._hovered_box_index = None
                self.pointsChanged.emit()
                self.boxRemoved.emit()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        pos = event.position()
        self._emit_cursor(pos)

        if self._edit_mode == "bbox":
            self._mouse_move_bbox(event)
            return

        if self._dragging and self._selected_index is not None:
            image_pos = self._display_to_image(pos.x(), pos.y())
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

        self._update_hover(pos)

    def _mouse_move_bbox(self, event: QMouseEvent) -> None:
        pos = event.position()

        if self._drawing_box and self._new_box_start_img is not None:
            image_pos = self._display_to_image(pos.x(), pos.y())
            if image_pos is not None:
                self._new_box_current_img = image_pos
                self.update()
            return

        if self._dragging_box and self._selected_box_index is not None and self._box_drag_anchor is not None:
            image_pos = self._display_to_image(pos.x(), pos.y())
            if image_pos is None:
                return

            idx = self._selected_box_index
            if idx < 0 or idx >= len(self._bboxes):
                return

            box = self._bboxes[idx]
            bw = float(box["x2"]) - float(box["x1"])
            bh = float(box["y2"]) - float(box["y1"])

            x1 = image_pos[0] - self._box_drag_anchor[0]
            y1 = image_pos[1] - self._box_drag_anchor[1]
            x2 = x1 + bw
            y2 = y1 + bh

            x1, y1, x2, y2 = self._clamp_box_to_image(x1, y1, x2, y2)
            box["x1"] = round(x1, 2)
            box["y1"] = round(y1, 2)
            box["x2"] = round(x2, 2)
            box["y2"] = round(y2, 2)
            self.pointsChanged.emit()
            self.update()
            return

        self._update_hover(pos)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        self._clear_hover()
        self.imageCursorMoved.emit(0.0, 0.0, False)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if self._edit_mode == "bbox":
                if self._drawing_box and self._new_box_start_img is not None and self._new_box_current_img is not None:
                    x1 = self._new_box_start_img[0]
                    y1 = self._new_box_start_img[1]
                    x2 = self._new_box_current_img[0]
                    y2 = self._new_box_current_img[1]
                    left = min(x1, x2)
                    top = min(y1, y2)
                    right = max(x1, x2)
                    bottom = max(y1, y2)
                    if (right - left) >= 2.0 and (bottom - top) >= 2.0:
                        self._bboxes.append(
                            {
                                "name": self._current_bbox_name,
                                "x1": round(left, 2),
                                "y1": round(top, 2),
                                "x2": round(right, 2),
                                "y2": round(bottom, 2),
                                "score": 0.0,
                            }
                        )
                        self.pointsChanged.emit()
                        self.boxAdded.emit()

                self._drawing_box = False
                self._new_box_start_img = None
                self._new_box_current_img = None
                self._dragging_box = False
                self._box_drag_anchor = None
                self.update()
                return

            self._dragging = False

    def _draw_points(self, painter: QPainter, points: list[dict[str, Any]], base: QColor, group: str) -> None:
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)

        for idx, pt in enumerate(points):
            x_disp, y_disp = self._image_to_display(float(pt["x"]), float(pt["y"]))
            center = QPointF(x_disp, y_disp)
            selected = self._edit_group == group and self._edit_mode == "keypoint" and idx == self._selected_index
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

    def _draw_bboxes(self, painter: QPainter) -> None:
        for idx, box in enumerate(self._bboxes):
            x1, y1 = self._image_to_display(float(box["x1"]), float(box["y1"]))
            x2, y2 = self._image_to_display(float(box["x2"]), float(box["y2"]))
            rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

            selected = self._edit_mode == "bbox" and idx == self._selected_box_index
            hovered = idx == self._hovered_box_index
            color = QColor(72, 208, 120)
            if selected:
                color = QColor(255, 120, 80)
            elif hovered:
                color = QColor(120, 220, 140)

            painter.setPen(QPen(color, 2.0))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 24))
            painter.drawRect(rect)

            if hovered or selected:
                name = str(box.get("name", "box"))
                self._draw_label_chip(painter, rect.x() + 4.0, rect.y() - 6.0, name, color)

    def _draw_temp_bbox(self, painter: QPainter) -> None:
        if not self._drawing_box or self._new_box_start_img is None or self._new_box_current_img is None:
            return

        x1, y1 = self._image_to_display(self._new_box_start_img[0], self._new_box_start_img[1])
        x2, y2 = self._image_to_display(self._new_box_current_img[0], self._new_box_current_img[1])
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        painter.setPen(QPen(QColor(72, 208, 120, 220), 1.6, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(72, 208, 120, 24))
        painter.drawRect(rect)

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

        candidates: list[tuple[str, list[dict[str, Any]]]] = [
            ("archery", self._archery_points),
            ("human", self._human_points),
        ]

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

    def _find_bbox_hit(self, pos: QPointF) -> int | None:
        if self._pixmap is None or self._image_draw_rect is None:
            return None

        img_pos = self._display_to_image(pos.x(), pos.y())
        if img_pos is None:
            return None
        x, y = img_pos

        for idx in range(len(self._bboxes) - 1, -1, -1):
            box = self._bboxes[idx]
            left = min(float(box["x1"]), float(box["x2"]))
            top = min(float(box["y1"]), float(box["y2"]))
            right = max(float(box["x1"]), float(box["x2"]))
            bottom = max(float(box["y1"]), float(box["y2"]))
            if left <= x <= right and top <= y <= bottom:
                return idx
        return None

    def _update_hover(self, pos: QPointF) -> None:
        old_group = self._hovered_group
        old_idx = self._hovered_index
        old_box = self._hovered_box_index

        self._hovered_group = None
        self._hovered_index = None
        self._hovered_box_index = self._find_bbox_hit(pos)

        best_group, best_idx = self._find_nearest_visible_point(pos, threshold=14.0)
        self._hovered_group = best_group
        self._hovered_index = best_idx

        if (old_group, old_idx, old_box) != (self._hovered_group, self._hovered_index, self._hovered_box_index):
            self.update()

    def _clear_hover(self) -> None:
        self._hovered_group = None
        self._hovered_index = None
        self._hovered_box_index = None

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

    def _clamp_box_to_image(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        if self._pixmap is None:
            return x1, y1, x2, y2

        max_x = float(max(self._pixmap.width() - 1, 0))
        max_y = float(max(self._pixmap.height() - 1, 0))

        bw = x2 - x1
        bh = y2 - y1

        x1 = min(max(0.0, x1), max_x)
        y1 = min(max(0.0, y1), max_y)
        x2 = x1 + bw
        y2 = y1 + bh

        if x2 > max_x:
            shift = x2 - max_x
            x1 -= shift
            x2 -= shift
        if y2 > max_y:
            shift = y2 - max_y
            y1 -= shift
            y2 -= shift

        x1 = min(max(0.0, x1), max_x)
        y1 = min(max(0.0, y1), max_y)
        x2 = min(max(0.0, x2), max_x)
        y2 = min(max(0.0, y2), max_y)
        return x1, y1, x2, y2

    def _emit_cursor(self, pos: QPointF) -> None:
        img = self._display_to_image(pos.x(), pos.y())
        if img is None:
            self.imageCursorMoved.emit(0.0, 0.0, False)
            return
        self.imageCursorMoved.emit(float(img[0]), float(img[1]), True)

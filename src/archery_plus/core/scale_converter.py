from __future__ import annotations

from typing import Any

from archery_plus.core.keypoint_schema import DEFAULT_ARCHERY_KEYPOINT_NAMES


class BowScaleConverter:
    def __init__(
        self,
        bow_length_cm: float = 177.8,
        scale_conf_threshold: float = 0.30,
        min_up_down_px: float = 8.0,
        freeze_on_invalid: bool = True,
    ) -> None:
        self.bow_length_cm = float(bow_length_cm)
        self.scale_conf_threshold = float(scale_conf_threshold)
        self.min_up_down_px = float(min_up_down_px)
        self.freeze_on_invalid = bool(freeze_on_invalid)
        self.cm_per_px: float | None = None

        names = [x.strip().upper() for x in DEFAULT_ARCHERY_KEYPOINT_NAMES]
        self._up_id = names.index("UP") if "UP" in names else 0
        self._down_id = names.index("DOWN") if "DOWN" in names else 1

    def set_bow_length_cm(self, value: float) -> None:
        self.bow_length_cm = max(1e-6, float(value))

    def set_threshold(self, conf_threshold: float) -> None:
        self.scale_conf_threshold = min(max(float(conf_threshold), 0.0), 1.0)

    def reset(self) -> None:
        self.cm_per_px = None

    def update(self, points: list[dict[str, Any]]) -> float | None:
        up = self._find_point(points, self._up_id, "UP")
        down = self._find_point(points, self._down_id, "DOWN")

        valid = False
        if up is not None and down is not None:
            up_score = float(up.get("score", 0.0))
            down_score = float(down.get("score", 0.0))
            if up_score >= self.scale_conf_threshold and down_score >= self.scale_conf_threshold:
                dx = float(up.get("x", 0.0)) - float(down.get("x", 0.0))
                dy = float(up.get("y", 0.0)) - float(down.get("y", 0.0))
                dist_px = (dx * dx + dy * dy) ** 0.5
                if dist_px >= self.min_up_down_px:
                    self.cm_per_px = self.bow_length_cm / dist_px
                    valid = True

        if not valid and not self.freeze_on_invalid:
            self.cm_per_px = None

        return self.cm_per_px

    def to_cm(self, x_px: float, y_px: float) -> tuple[float, float] | None:
        if self.cm_per_px is None:
            return None
        scale = float(self.cm_per_px)
        return round(float(x_px) * scale, 2), round(float(y_px) * scale, 2)

    def _find_point(
        self,
        points: list[dict[str, Any]],
        key_id: int,
        key_name: str,
    ) -> dict[str, Any] | None:
        by_name = key_name.strip().upper()
        for one in points:
            if not isinstance(one, dict):
                continue
            try:
                pid = int(one.get("id", -1))
            except Exception:
                pid = -1
            if pid == key_id:
                return one

        for one in points:
            if not isinstance(one, dict):
                continue
            name = str(one.get("name", "")).strip().upper()
            if name == by_name:
                return one
        return None

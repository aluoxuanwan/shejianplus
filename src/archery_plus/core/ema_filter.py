from __future__ import annotations

from copy import deepcopy
from typing import Any


class EmaPointFilter:
    def __init__(self, alpha: float = 0.7) -> None:
        self._alpha = 0.7
        self._state: dict[int, tuple[float, float]] = {}
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float) -> None:
        a = float(alpha)
        self._alpha = min(max(a, 0.0), 1.0)

    def reset(self) -> None:
        self._state.clear()

    def update(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in points:
            if not isinstance(item, dict):
                continue

            one = deepcopy(item)
            try:
                pid = int(one.get("id", -1))
            except Exception:
                pid = -1
            if pid < 0:
                continue

            try:
                x_now = float(one.get("x", 0.0))
                y_now = float(one.get("y", 0.0))
            except Exception:
                continue

            if pid in self._state:
                x_prev, y_prev = self._state[pid]
                x_new = self._alpha * x_now + (1.0 - self._alpha) * x_prev
                y_new = self._alpha * y_now + (1.0 - self._alpha) * y_prev
            else:
                x_new, y_new = x_now, y_now

            self._state[pid] = (x_new, y_new)
            one["x"] = round(float(x_new), 2)
            one["y"] = round(float(y_new), 2)
            out.append(one)

        return out

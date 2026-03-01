from __future__ import annotations

import math
import time
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np

try:
    from scipy.signal import butter, filtfilt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    butter = None
    filtfilt = None


class OneEuroPointFilter:
    def __init__(self, alpha: float = 0.7) -> None:
        self._alpha = 0.7
        self._min_cutoff = 2.8
        self._beta = 4.2
        self._d_cutoff = 1.0

        self._x_prev: dict[int, tuple[float, float]] = {}
        self._dx_prev: dict[int, tuple[float, float]] = {}
        self._t_prev: dict[int, float] = {}
        self.set_alpha(alpha)

    def set_one_euro_params(
        self,
        *,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        if min_cutoff is not None:
            self._min_cutoff = max(1e-4, float(min_cutoff))
        if beta is not None:
            self._beta = max(0.0, float(beta))
        if d_cutoff is not None:
            self._d_cutoff = max(1e-4, float(d_cutoff))

    def get_one_euro_params(self) -> dict[str, float]:
        return {
            "alpha": float(self._alpha),
            "min_cutoff": float(self._min_cutoff),
            "beta": float(self._beta),
            "d_cutoff": float(self._d_cutoff),
        }

    def set_alpha(self, alpha: float) -> None:
        a = min(max(float(alpha), 0.0), 1.0)
        self._alpha = a
        # Keep UI semantics close to the previous slider:
        # alpha up -> less lag, alpha down -> smoother.
        self._min_cutoff = 0.4 + 3.6 * a
        self._beta = 6.0 * a

    def reset(self) -> None:
        self._x_prev.clear()
        self._dx_prev.clear()
        self._t_prev.clear()

    def update(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.perf_counter()
        out: list[dict[str, Any]] = []

        for item in points:
            if not isinstance(item, dict):
                continue
            one = deepcopy(item)

            try:
                pid = int(one.get("id", -1))
                x_now = float(one.get("x", 0.0))
                y_now = float(one.get("y", 0.0))
            except Exception:
                continue
            if pid < 0:
                continue

            t_prev = self._t_prev.get(pid)
            dt = now - t_prev if t_prev is not None else 1.0 / 30.0
            dt = min(max(float(dt), 1e-3), 0.1)

            x_prev, y_prev = self._x_prev.get(pid, (x_now, y_now))
            dx_prev, dy_prev = self._dx_prev.get(pid, (0.0, 0.0))

            dx = (x_now - x_prev) / dt
            dy = (y_now - y_prev) / dt

            a_d = self._smoothing_factor(dt, self._d_cutoff)
            dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
            dy_hat = a_d * dy + (1.0 - a_d) * dy_prev

            cutoff_x = self._min_cutoff + self._beta * abs(dx_hat)
            cutoff_y = self._min_cutoff + self._beta * abs(dy_hat)
            a_x = self._smoothing_factor(dt, cutoff_x)
            a_y = self._smoothing_factor(dt, cutoff_y)

            x_hat = a_x * x_now + (1.0 - a_x) * x_prev
            y_hat = a_y * y_now + (1.0 - a_y) * y_prev

            self._x_prev[pid] = (x_hat, y_hat)
            self._dx_prev[pid] = (dx_hat, dy_hat)
            self._t_prev[pid] = now

            one["x"] = round(float(x_hat), 2)
            one["y"] = round(float(y_hat), 2)
            out.append(one)

        return out

    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(float(cutoff), 1e-6))
        return 1.0 / (1.0 + tau / dt)


class ButterworthBidirectionalPointFilter:
    """Near-real-time bidirectional 4th-order Butterworth low-pass filter.

    Notes:
    - Bidirectional (filtfilt) is zero-phase but non-causal, so in streaming mode
      we apply it over a rolling window and output the latest sample. This reduces
      phase lag but may introduce edge effects and needs a short warm-up window.
    - Requires scipy. If missing, initialization raises RuntimeError.
    """

    def __init__(
        self,
        cutoff_hz: float = 3.0,
        sample_hz: float = 30.0,
        window_seconds: float = 2.0,
    ) -> None:
        if butter is None or filtfilt is None:
            raise RuntimeError("Butterworth filter requires scipy. Please install scipy>=1.10.")

        self._cutoff_hz = 3.0
        self._sample_hz = 30.0
        self._window_seconds = 2.0
        self._history_xy: dict[int, deque[tuple[float, float]]] = {}
        self._set_params(
            cutoff_hz=cutoff_hz,
            sample_hz=sample_hz,
            window_seconds=window_seconds,
        )

    def _set_params(
        self,
        *,
        cutoff_hz: float | None = None,
        sample_hz: float | None = None,
        window_seconds: float | None = None,
    ) -> None:
        if cutoff_hz is not None:
            self._cutoff_hz = max(0.05, float(cutoff_hz))
        if sample_hz is not None:
            self._sample_hz = max(1.0, float(sample_hz))
        if window_seconds is not None:
            self._window_seconds = min(max(0.3, float(window_seconds)), 10.0)

        max_len = max(12, int(round(self._sample_hz * self._window_seconds)))
        # filtfilt edge length for 4th-order butter is 3*(max(len(a),len(b))-1)=12
        self._min_len = 16
        self._max_len = max(self._min_len, max_len)

        nyq = max(self._sample_hz * 0.5, 1e-6)
        normalized = min(max(self._cutoff_hz / nyq, 1e-4), 0.999)
        self._b, self._a = butter(4, normalized, btype="low", analog=False)

        for pid, hist in list(self._history_xy.items()):
            trimmed = deque(list(hist)[-self._max_len :], maxlen=self._max_len)
            self._history_xy[pid] = trimmed

    def set_alpha(self, alpha: float) -> None:
        # Reuse the old "smoothness" slider semantics:
        # alpha up => less smoothing => higher cutoff_hz.
        a = min(max(float(alpha), 0.0), 1.0)
        cutoff = 0.6 + a * 8.4
        self._set_params(cutoff_hz=cutoff)

    def set_cutoff_hz(self, cutoff_hz: float) -> None:
        self._set_params(cutoff_hz=cutoff_hz)

    def set_sample_hz(self, sample_hz: float) -> None:
        self._set_params(sample_hz=sample_hz)

    def set_window_seconds(self, window_seconds: float) -> None:
        self._set_params(window_seconds=window_seconds)

    def get_params(self) -> dict[str, float]:
        return {
            "cutoff_hz": float(self._cutoff_hz),
            "sample_hz": float(self._sample_hz),
            "window_seconds": float(self._window_seconds),
        }

    def reset(self) -> None:
        self._history_xy.clear()

    def update(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in points:
            if not isinstance(item, dict):
                continue
            one = deepcopy(item)
            try:
                pid = int(one.get("id", -1))
                x_now = float(one.get("x", 0.0))
                y_now = float(one.get("y", 0.0))
            except Exception:
                continue
            if pid < 0:
                continue

            hist = self._history_xy.get(pid)
            if hist is None:
                hist = deque(maxlen=self._max_len)
                self._history_xy[pid] = hist
            if hist.maxlen != self._max_len:
                hist = deque(list(hist)[-self._max_len :], maxlen=self._max_len)
                self._history_xy[pid] = hist

            hist.append((x_now, y_now))

            if len(hist) < self._min_len:
                x_hat, y_hat = x_now, y_now
            else:
                arr = np.asarray(hist, dtype=np.float64)
                try:
                    xf = filtfilt(self._b, self._a, arr[:, 0], method="pad")
                    yf = filtfilt(self._b, self._a, arr[:, 1], method="pad")
                    x_hat = float(xf[-1])
                    y_hat = float(yf[-1])
                except Exception:
                    x_hat, y_hat = x_now, y_now

            one["x"] = round(float(x_hat), 2)
            one["y"] = round(float(y_hat), 2)
            out.append(one)
        return out

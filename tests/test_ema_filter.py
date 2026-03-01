from __future__ import annotations

from shejianplus.core.ema_filter import OneEuroPointFilter


def test_one_euro_filter_preserves_point_shape() -> None:
    filt = OneEuroPointFilter(alpha=0.7)

    result = filt.update([{"id": 0, "x": 10.0, "y": 20.0, "score": 0.9}])

    assert len(result) == 1
    assert result[0]["id"] == 0
    assert "x" in result[0]
    assert "y" in result[0]
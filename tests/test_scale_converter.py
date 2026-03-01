from __future__ import annotations

from shejianplus.core.scale_converter import BowScaleConverter


def test_scale_converter_freezes_previous_scale_on_invalid_frame() -> None:
    conv = BowScaleConverter(bow_length_cm=177.8, scale_conf_threshold=0.3)

    first = conv.update(
        [
            {"id": 0, "name": "UP", "x": 0.0, "y": 0.0, "score": 0.9},
            {"id": 1, "name": "DOWN", "x": 0.0, "y": 100.0, "score": 0.9},
        ]
    )
    second = conv.update(
        [
            {"id": 0, "name": "UP", "x": 0.0, "y": 0.0, "score": 0.1},
            {"id": 1, "name": "DOWN", "x": 0.0, "y": 100.0, "score": 0.1},
        ]
    )

    assert first is not None
    assert second == first
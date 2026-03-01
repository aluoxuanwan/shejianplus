from __future__ import annotations

import json

import pytest

from shejianplus.core.keypoint_schema import KeypointSchemaError, load_keypoint_set_from_file


def test_load_keypoint_set_accepts_continuous_ids(tmp_path) -> None:
    path = tmp_path / "archery.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "set_name": "archery_5",
                "keypoints": [
                    {"id": 0, "name": "UP"},
                    {"id": 1, "name": "DOWN"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_keypoint_set_from_file(path, "archery")

    assert result.set_name == "archery_5"
    assert result.names == ["UP", "DOWN"]


def test_load_keypoint_set_rejects_duplicate_names(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "set_name": "bad",
                "keypoints": [
                    {"id": 0, "name": "UP"},
                    {"id": 1, "name": "UP"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeypointSchemaError):
        load_keypoint_set_from_file(path, "archery")
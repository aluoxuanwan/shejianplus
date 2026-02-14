from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class AnnotationStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.file_path = self.project_root / "annotations" / "manual_annotations.json"

    def load_all(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if not self.file_path.exists():
            return {}
        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): self._normalize_image_annotation(v) for k, v in data.items()}

    def save_all(self, data: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_image_annotation(self, image_name: str) -> dict[str, list[dict[str, Any]]]:
        annotation = self.load_all().get(image_name)
        if annotation is None:
            return {"human_keypoints": [], "archery_keypoints": []}
        return deepcopy(annotation)

    def set_image_annotation(self, image_name: str, annotation: dict[str, list[dict[str, Any]]]) -> None:
        all_data = self.load_all()
        all_data[image_name] = self._normalize_image_annotation(annotation)
        self.save_all(all_data)

    def get_image_points(self, image_name: str) -> list[dict[str, Any]]:
        return self.get_image_annotation(image_name)["archery_keypoints"]

    def set_image_points(self, image_name: str, points: list[dict[str, Any]]) -> None:
        all_data = self.load_all()
        normalized = self._normalize_image_annotation(all_data.get(image_name, {}))
        normalized["archery_keypoints"] = self._normalize_points(points)
        all_data[image_name] = normalized
        self.save_all(all_data)

    def set_human_points(self, image_name: str, points: list[dict[str, Any]]) -> None:
        all_data = self.load_all()
        normalized = self._normalize_image_annotation(all_data.get(image_name, {}))
        normalized["human_keypoints"] = self._normalize_points(points)
        all_data[image_name] = normalized
        self.save_all(all_data)

    def _normalize_image_annotation(self, raw: Any) -> dict[str, list[dict[str, Any]]]:
        if isinstance(raw, list):
            return {
                "human_keypoints": [],
                "archery_keypoints": self._normalize_points(raw),
            }
        if not isinstance(raw, dict):
            return {"human_keypoints": [], "archery_keypoints": []}

        human = self._normalize_points(raw.get("human_keypoints", []))
        archery = self._normalize_points(raw.get("archery_keypoints", []))
        return {"human_keypoints": human, "archery_keypoints": archery}

    def _normalize_points(self, raw_points: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_points, list):
            return []

        points: list[dict[str, Any]] = []
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            x = item.get("x")
            y = item.get("y")
            if x is None or y is None:
                continue
            point = {
                "name": str(item.get("name", "")),
                "x": float(x),
                "y": float(y),
                "score": float(item.get("score", 0.0)),
            }
            if "v" in item:
                point["v"] = int(item.get("v", 2))
            points.append(point)
        return points

from __future__ import annotations

import json
from pathlib import Path


class AnnotationStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.file_path = self.project_root / "annotations" / "manual_annotations.json"

    def load_all(self) -> dict[str, list[dict[str, float | int | str]]]:
        if not self.file_path.exists():
            return {}
        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data

    def save_all(self, data: dict[str, list[dict[str, float | int | str]]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_image_points(self, image_name: str) -> list[dict[str, float | int | str]]:
        return self.load_all().get(image_name, [])

    def set_image_points(self, image_name: str, points: list[dict[str, float | int | str]]) -> None:
        all_data = self.load_all()
        all_data[image_name] = points
        self.save_all(all_data)

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


TARGET_HUMAN = "human"
TARGET_ARCHERY = "archery"


class AnnotationStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.file_path = self.project_root / "annotations" / "manual_annotations.json"
        self.label_schema_path = self.project_root / "annotations" / "label_schema.json"

    # ---------- Annotation ----------
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
            return {"human_keypoints": [], "archery_keypoints": [], "bboxes": []}
        return deepcopy(annotation)

    def set_image_annotation(self, image_name: str, annotation: dict[str, list[dict[str, Any]]]) -> None:
        all_data = self.load_all()
        all_data[image_name] = self._normalize_image_annotation(annotation)
        self.save_all(all_data)

    # ---------- Label Schema ----------
    def load_label_schema(self) -> dict[str, list[str]]:
        if not self.label_schema_path.exists():
            return self._default_label_schema()

        try:
            with self.label_schema_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return self._default_label_schema()

        if not isinstance(raw, dict):
            return self._default_label_schema()

        return {
            "human_keypoint_labels": self._normalize_name_list(raw.get("human_keypoint_labels", [])),
            "archery_keypoint_labels": self._normalize_name_list(raw.get("archery_keypoint_labels", [])),
            "bbox_labels": self._normalize_name_list(raw.get("bbox_labels", [])),
        }

    def save_label_schema(self, schema: dict[str, list[str]]) -> None:
        normalized = {
            "human_keypoint_labels": self._normalize_name_list(schema.get("human_keypoint_labels", [])),
            "archery_keypoint_labels": self._normalize_name_list(schema.get("archery_keypoint_labels", [])),
            "bbox_labels": self._normalize_name_list(schema.get("bbox_labels", [])),
        }
        self.label_schema_path.parent.mkdir(parents=True, exist_ok=True)
        with self.label_schema_path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

    def ensure_label_schema(self) -> dict[str, list[str]]:
        schema = self.load_label_schema()
        if not self.label_schema_path.exists():
            self.save_label_schema(schema)
        return schema

    def set_keypoint_labels(self, target: str, names: list[str]) -> None:
        schema = self.ensure_label_schema()
        key = self._target_to_key(target)
        schema[key] = self._normalize_name_list(names)
        self.save_label_schema(schema)

    def add_keypoint_label(self, target: str, name: str) -> None:
        clean = str(name).strip()
        if not clean:
            return
        schema = self.ensure_label_schema()
        key = self._target_to_key(target)
        names = schema.get(key, [])
        if clean not in names:
            names.append(clean)
        schema[key] = self._normalize_name_list(names)
        self.save_label_schema(schema)

    def add_bbox_label(self, name: str) -> None:
        clean = str(name).strip()
        if not clean:
            return
        schema = self.ensure_label_schema()
        names = schema.get("bbox_labels", [])
        if clean not in names:
            names.append(clean)
        schema["bbox_labels"] = self._normalize_name_list(names)
        self.save_label_schema(schema)

    def get_keypoint_labels(self, target: str) -> list[str]:
        schema = self.ensure_label_schema()
        return list(schema.get(self._target_to_key(target), []))

    def get_bbox_labels(self) -> list[str]:
        schema = self.ensure_label_schema()
        return list(schema.get("bbox_labels", []))

    def count_mismatched_names(
        self,
        image_name: str,
        *,
        valid_human_names: list[str],
        valid_archery_names: list[str],
        valid_bbox_names: list[str] | None = None,
    ) -> dict[str, int]:
        ann = self.get_image_annotation(image_name)
        human_valid = set(valid_human_names)
        archery_valid = set(valid_archery_names)
        bbox_valid = set(valid_bbox_names or [])

        human_count = sum(
            1
            for p in ann.get("human_keypoints", [])
            if isinstance(p, dict)
            and str(p.get("name", "")).strip()
            and str(p.get("name", "")).strip() not in human_valid
        )
        archery_count = sum(
            1
            for p in ann.get("archery_keypoints", [])
            if isinstance(p, dict)
            and str(p.get("name", "")).strip()
            and str(p.get("name", "")).strip() not in archery_valid
        )
        bbox_count = sum(
            1
            for b in ann.get("bboxes", [])
            if isinstance(b, dict)
            and str(b.get("name", "")).strip()
            and str(b.get("name", "")).strip() not in bbox_valid
        )

        return {"human": int(human_count), "archery": int(archery_count), "bbox": int(bbox_count)}

    def _normalize_image_annotation(self, raw: Any) -> dict[str, list[dict[str, Any]]]:
        if isinstance(raw, list):
            return {
                "human_keypoints": [],
                "archery_keypoints": self._normalize_points(raw),
                "bboxes": [],
            }
        if not isinstance(raw, dict):
            return {"human_keypoints": [], "archery_keypoints": [], "bboxes": []}

        human = self._normalize_points(raw.get("human_keypoints", []))
        archery = self._normalize_points(raw.get("archery_keypoints", []))
        bboxes = self._normalize_boxes(raw.get("bboxes", []))
        return {"human_keypoints": human, "archery_keypoints": archery, "bboxes": bboxes}

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

    def _normalize_boxes(self, raw_boxes: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_boxes, list):
            return []

        boxes: list[dict[str, Any]] = []
        for item in raw_boxes:
            if not isinstance(item, dict):
                continue

            x1 = item.get("x1")
            y1 = item.get("y1")
            x2 = item.get("x2")
            y2 = item.get("y2")
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue

            nx1 = float(x1)
            ny1 = float(y1)
            nx2 = float(x2)
            ny2 = float(y2)

            left = min(nx1, nx2)
            top = min(ny1, ny2)
            right = max(nx1, nx2)
            bottom = max(ny1, ny2)

            if (right - left) < 1.0 or (bottom - top) < 1.0:
                continue

            boxes.append(
                {
                    "name": str(item.get("name", "")),
                    "x1": left,
                    "y1": top,
                    "x2": right,
                    "y2": bottom,
                    "score": float(item.get("score", 0.0)),
                }
            )
        return boxes

    def _default_label_schema(self) -> dict[str, list[str]]:
        return {
            "human_keypoint_labels": [],
            "archery_keypoint_labels": [],
            "bbox_labels": [],
        }

    def _normalize_name_list(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        names: list[str] = []
        for item in raw:
            name = str(item).strip()
            if name and name not in names:
                names.append(name)
        return names

    def _target_to_key(self, target: str) -> str:
        t = str(target).strip().lower()
        if t in {"human", "人体关键点", "human_keypoints"}:
            return "human_keypoint_labels"
        return "archery_keypoint_labels"

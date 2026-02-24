from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from archery_plus.data.annotation_store import AnnotationStore


@dataclass
class ExportOptions:
    export_raw_2d_csv: bool = True
    export_filtered_2d_csv: bool = True
    export_bbox_csv: bool = True
    export_annotations_json: bool = True
    export_label_schema_json: bool = True


@dataclass
class ExportResult:
    output_dir: Path
    image_count: int
    keypoint_rows: int
    bbox_rows: int
    files: list[Path]


def export_project_annotations(project_root: Path, options: ExportOptions) -> ExportResult:
    project_root = Path(project_root)
    store = AnnotationStore(project_root)
    annotations = store.load_all()
    schema = store.ensure_label_schema()

    output_dir = _create_export_dir(project_root)
    files: list[Path] = []

    image_root = project_root / "images" / "cam1"
    image_meta = _collect_image_meta(image_root, annotations.keys())

    keypoint_rows = _build_keypoint_rows(annotations, schema, image_meta)
    bbox_rows = _build_bbox_rows(annotations, image_meta)

    if options.export_raw_2d_csv:
        path = output_dir / "raw_2d.csv"
        _write_csv(path, keypoint_rows, fieldnames=_keypoint_fieldnames())
        files.append(path)

    if options.export_filtered_2d_csv:
        # Export v0.1: no temporal filtering pipeline here, so keep a deterministic copy
        # and mark the filter method to make semantics explicit.
        filtered_rows = [dict(row, filter_method="none_v0_1") for row in keypoint_rows]
        path = output_dir / "filtered_2d.csv"
        _write_csv(path, filtered_rows, fieldnames=_keypoint_fieldnames())
        files.append(path)

    if options.export_bbox_csv:
        path = output_dir / "bbox_2d.csv"
        _write_csv(path, bbox_rows, fieldnames=_bbox_fieldnames())
        files.append(path)

    if options.export_annotations_json:
        path = output_dir / "annotations_export.json"
        payload = {
            "schema_version": 1,
            "project_root": str(project_root),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "images": len(annotations),
                "keypoint_rows": len(keypoint_rows),
                "bbox_rows": len(bbox_rows),
            },
            "label_schema": schema,
            "annotations": annotations,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        files.append(path)

    if options.export_label_schema_json:
        path = output_dir / "label_schema_export.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        files.append(path)

    return ExportResult(
        output_dir=output_dir,
        image_count=len(annotations),
        keypoint_rows=len(keypoint_rows),
        bbox_rows=len(bbox_rows),
        files=files,
    )


def _create_export_dir(project_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = project_root / "output" / "exports" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _collect_image_meta(image_root: Path, image_names: Any) -> dict[str, dict[str, int | None]]:
    meta: dict[str, dict[str, int | None]] = {}
    for image_name in image_names:
        name = str(image_name)
        width: int | None = None
        height: int | None = None
        path = image_root / name
        if path.exists():
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    h, w = img.shape[:2]
                    width = int(w)
                    height = int(h)
            except Exception:
                pass
        meta[name] = {"width": width, "height": height}
    return meta


def _build_keypoint_rows(
    annotations: dict[str, dict[str, list[dict[str, Any]]]],
    schema: dict[str, list[str]],
    image_meta: dict[str, dict[str, int | None]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    human_labels = list(schema.get("human_keypoint_labels", []))
    archery_labels = list(schema.get("archery_keypoint_labels", []))
    human_index = {name: idx for idx, name in enumerate(human_labels)}
    archery_index = {name: idx for idx, name in enumerate(archery_labels)}

    for image_name in sorted(annotations.keys()):
        ann = annotations.get(image_name, {})
        meta = image_meta.get(image_name, {})

        for target_key, target_name, label_index in (
            ("human_keypoints", "human", human_index),
            ("archery_keypoints", "archery", archery_index),
        ):
            points = ann.get(target_key, []) if isinstance(ann, dict) else []
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                kp_name = str(point.get("name", "")).strip()
                row = {
                    "image_name": image_name,
                    "target": target_name,
                    "keypoint_name": kp_name,
                    "keypoint_id": int(label_index.get(kp_name, -1)),
                    "x": _to_float(point.get("x")),
                    "y": _to_float(point.get("y")),
                    "score": _to_float(point.get("score")),
                    "v": _to_int(point.get("v"), default=0),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "filter_method": "raw",
                }
                rows.append(row)
    return rows


def _build_bbox_rows(
    annotations: dict[str, dict[str, list[dict[str, Any]]]],
    image_meta: dict[str, dict[str, int | None]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_name in sorted(annotations.keys()):
        ann = annotations.get(image_name, {})
        meta = image_meta.get(image_name, {})
        boxes = ann.get("bboxes", []) if isinstance(ann, dict) else []
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict):
                continue
            x1 = _to_float(box.get("x1"))
            y1 = _to_float(box.get("y1"))
            x2 = _to_float(box.get("x2"))
            y2 = _to_float(box.get("y2"))
            if None in (x1, y1, x2, y2):
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "label": str(box.get("name", "")).strip(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": round(float(x2) - float(x1), 2),
                    "h": round(float(y2) - float(y1), 2),
                    "score": _to_float(box.get("score")),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _keypoint_fieldnames() -> list[str]:
    return [
        "image_name",
        "target",
        "keypoint_name",
        "keypoint_id",
        "x",
        "y",
        "score",
        "v",
        "width",
        "height",
        "filter_method",
    ]


def _bbox_fieldnames() -> list[str]:
    return [
        "image_name",
        "label",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "score",
        "width",
        "height",
    ]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

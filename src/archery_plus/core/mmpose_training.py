from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from archery_plus.core.keypoint_schema import (
    TARGET_ARCHERY,
    TARGET_HUMAN,
    ensure_project_keypoint_sets,
    get_active_keypoint_set,
    normalize_target_name,
)


@dataclass(frozen=True)
class SplitAnnotationPaths:
    train: Path
    val: Path
    test: Path


@dataclass
class MMPoseDatasetBuildResult:
    output_dir: Path
    annotation_dir: Path
    metadata_path: Path
    total_images: int
    train_images: int
    val_images: int
    test_images: int
    skipped_images: int
    human_paths: SplitAnnotationPaths
    archery_paths: SplitAnnotationPaths
    human_ann_counts: dict[str, int]
    archery_ann_counts: dict[str, int]


def build_mmpose_coco_dataset(
    project_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    human_category_name: str | None = None,
    archery_category_name: str | None = None,
    active_target: str = TARGET_HUMAN,
    active_category_name: str = "",
    output_dir: Path | None = None,
) -> MMPoseDatasetBuildResult:
    ann_path = project_root / "annotations" / "manual_annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise RuntimeError("manual_annotations.json format invalid: root should be dict")

    ensure_project_keypoint_sets(project_root)
    human_set = get_active_keypoint_set(project_root, TARGET_HUMAN)
    archery_set = get_active_keypoint_set(project_root, TARGET_ARCHERY)

    active = normalize_target_name(active_target)
    human_category = (human_category_name or "").strip() or human_set.set_name or "human"
    archery_category = (archery_category_name or "").strip() or archery_set.set_name or "archery"
    active_category = active_category_name.strip()
    if active_category:
        if active == TARGET_HUMAN:
            human_category = active_category
        else:
            archery_category = active_category

    human_names = human_set.names
    archery_names = archery_set.names

    image_root = project_root / "images" / "cam1"
    records: list[dict[str, Any]] = []
    skipped_images = 0

    for image_name, payload in raw.items():
        image_path = image_root / str(image_name)
        if not image_path.exists():
            skipped_images += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images += 1
            continue

        height, width = image.shape[:2]
        if isinstance(payload, list):
            human_points_raw = []
            archery_points_raw = payload
        elif isinstance(payload, dict):
            human_points_raw = payload.get("human_keypoints", [])
            archery_points_raw = payload.get("archery_keypoints", [])
        else:
            skipped_images += 1
            continue

        records.append(
            {
                "image_name": str(image_name),
                "image_path": str(Path("images") / "cam1" / str(image_name)).replace("\\", "/"),
                "width": int(width),
                "height": int(height),
                "human_keypoints": _order_points(human_points_raw, human_names),
                "archery_keypoints": _order_points(archery_points_raw, archery_names),
            }
        )

    if not records:
        raise RuntimeError("No valid records found to build mmpose dataset")

    rng = random.Random(random_seed)
    rng.shuffle(records)

    train_records, val_records, test_records = _split_records(records, train_ratio, val_ratio, test_ratio)

    dataset_output_dir = output_dir or (project_root / "output" / "mmpose_dataset_v1")
    annotation_dir = dataset_output_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    human_train_json, human_train_count = _build_coco_split(
        train_records,
        key_field="human_keypoints",
        keypoint_names=human_names,
        category_name=human_category,
    )
    human_val_json, human_val_count = _build_coco_split(
        val_records,
        key_field="human_keypoints",
        keypoint_names=human_names,
        category_name=human_category,
    )
    human_test_json, human_test_count = _build_coco_split(
        test_records,
        key_field="human_keypoints",
        keypoint_names=human_names,
        category_name=human_category,
    )

    archery_train_json, archery_train_count = _build_coco_split(
        train_records,
        key_field="archery_keypoints",
        keypoint_names=archery_names,
        category_name=archery_category,
    )
    archery_val_json, archery_val_count = _build_coco_split(
        val_records,
        key_field="archery_keypoints",
        keypoint_names=archery_names,
        category_name=archery_category,
    )
    archery_test_json, archery_test_count = _build_coco_split(
        test_records,
        key_field="archery_keypoints",
        keypoint_names=archery_names,
        category_name=archery_category,
    )

    human_paths = SplitAnnotationPaths(
        train=annotation_dir / "human_train.json",
        val=annotation_dir / "human_val.json",
        test=annotation_dir / "human_test.json",
    )
    archery_paths = SplitAnnotationPaths(
        train=annotation_dir / "archery_train.json",
        val=annotation_dir / "archery_val.json",
        test=annotation_dir / "archery_test.json",
    )

    _write_json(human_paths.train, human_train_json)
    _write_json(human_paths.val, human_val_json)
    _write_json(human_paths.test, human_test_json)
    _write_json(archery_paths.train, archery_train_json)
    _write_json(archery_paths.val, archery_val_json)
    _write_json(archery_paths.test, archery_test_json)

    metadata_path = dataset_output_dir / "dataset_meta.json"
    metadata = {
        "format": "mmpose_coco_keypoint",
        "project_root": str(project_root),
        "splits": {
            "train_images": len(train_records),
            "val_images": len(val_records),
            "test_images": len(test_records),
            "total_images": len(records),
            "skipped_images": skipped_images,
        },
        "categories": {
            "human": {
                "name": human_category,
                "set_name": human_set.set_name,
                "keypoints": human_names,
                "annotation_files": {
                    "train": str(human_paths.train),
                    "val": str(human_paths.val),
                    "test": str(human_paths.test),
                },
                "annotation_counts": {
                    "train": human_train_count,
                    "val": human_val_count,
                    "test": human_test_count,
                },
            },
            "archery": {
                "name": archery_category,
                "set_name": archery_set.set_name,
                "keypoints": archery_names,
                "annotation_files": {
                    "train": str(archery_paths.train),
                    "val": str(archery_paths.val),
                    "test": str(archery_paths.test),
                },
                "annotation_counts": {
                    "train": archery_train_count,
                    "val": archery_val_count,
                    "test": archery_test_count,
                },
            },
        },
        "active_target": active,
        "random_seed": random_seed,
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
    }
    _write_json(metadata_path, metadata)

    return MMPoseDatasetBuildResult(
        output_dir=dataset_output_dir,
        annotation_dir=annotation_dir,
        metadata_path=metadata_path,
        total_images=len(records),
        train_images=len(train_records),
        val_images=len(val_records),
        test_images=len(test_records),
        skipped_images=skipped_images,
        human_paths=human_paths,
        archery_paths=archery_paths,
        human_ann_counts={"train": human_train_count, "val": human_val_count, "test": human_test_count},
        archery_ann_counts={
            "train": archery_train_count,
            "val": archery_val_count,
            "test": archery_test_count,
        },
    )


def _build_coco_split(
    records: list[dict[str, Any]],
    key_field: str,
    keypoint_names: list[str],
    category_name: str,
) -> tuple[dict[str, Any], int]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    ann_id = 1

    for record in records:
        flattened, visible_count, bbox = _to_coco_keypoints(
            record.get(key_field, []),
            image_width=int(record["width"]),
            image_height=int(record["height"]),
            expected_count=len(keypoint_names),
        )

        if visible_count <= 0:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": str(record["image_path"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )

        box_x, box_y, box_w, box_h = bbox
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [round(box_x, 2), round(box_y, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(float(box_w * box_h), 2),
                "num_keypoints": int(visible_count),
                "keypoints": flattened,
            }
        )

        image_id += 1
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "keypoint_object",
                "keypoints": list(keypoint_names),
                "skeleton": _build_simple_skeleton(len(keypoint_names)),
            }
        ],
        "info": {
            "description": "archery_plus mmpose dataset",
            "version": "v1",
        },
    }
    return coco, len(annotations)


def _to_coco_keypoints(
    points: object,
    image_width: int,
    image_height: int,
    expected_count: int,
) -> tuple[list[float | int], int, tuple[float, float, float, float]]:
    flattened: list[float | int] = []
    visible_xy: list[tuple[float, float]] = []

    points_list = points if isinstance(points, list) else []
    normalized = points_list[:expected_count]
    if len(normalized) < expected_count:
        normalized = normalized + [None] * (expected_count - len(normalized))

    for point in normalized:
        if isinstance(point, dict):
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            raw_v = int(point.get("v", 0))
            if x <= 0.0 and y <= 0.0:
                v = 0
            else:
                v = 2 if raw_v > 0 else 0
            x = float(min(max(x, 0.0), max(image_width - 1, 0)))
            y = float(min(max(y, 0.0), max(image_height - 1, 0)))
        else:
            x = 0.0
            y = 0.0
            v = 0

        flattened.extend([round(x, 2), round(y, 2), int(v)])
        if v > 0:
            visible_xy.append((x, y))

    if not visible_xy:
        return flattened, 0, (0.0, 0.0, float(max(image_width, 1)), float(max(image_height, 1)))

    xs = [xy[0] for xy in visible_xy]
    ys = [xy[1] for xy in visible_xy]
    x_min = max(0.0, float(min(xs)))
    y_min = max(0.0, float(min(ys)))
    x_max = min(float(max(image_width - 1, 0)), float(max(xs)))
    y_max = min(float(max(image_height - 1, 0)), float(max(ys)))
    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    return flattened, len(visible_xy), (x_min, y_min, box_w, box_h)


def _split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(records)
    if total <= 1:
        return records[:], [], []

    train_r, val_r, test_r = _normalize_ratios(train_ratio, val_ratio, test_ratio)

    train_end = int(round(total * train_r))
    val_end = train_end + int(round(total * val_r))

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    if total >= 3:
        if not val_records and len(train_records) > 1:
            val_records.append(train_records.pop())
        if not test_records and len(train_records) > 1:
            test_records.append(train_records.pop())
        if not train_records:
            if len(val_records) > 1:
                train_records.append(val_records.pop())
            elif len(test_records) > 1:
                train_records.append(test_records.pop())

    if not test_records and len(val_records) > 1:
        test_records.append(val_records.pop())
    if not val_records and len(test_records) > 1:
        val_records.append(test_records.pop())

    return train_records, val_records, test_records


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    train = max(0.0, float(train_ratio))
    val = max(0.0, float(val_ratio))
    test = max(0.0, float(test_ratio))
    ratio_sum = train + val + test
    if ratio_sum <= 0:
        return 0.8, 0.1, 0.1
    return train / ratio_sum, val / ratio_sum, test / ratio_sum


def _build_simple_skeleton(num_keypoints: int) -> list[list[int]]:
    if num_keypoints <= 1:
        return []
    return [[idx, idx + 1] for idx in range(1, num_keypoints)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _order_points(raw_points: object, names_order: list[str]) -> list[dict[str, float | int | str]]:
    points = raw_points if isinstance(raw_points, list) else []
    by_name: dict[str, dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        name = str(point.get("name", "")).strip()
        if not name:
            continue
        by_name[name] = point

    ordered: list[dict[str, float | int | str]] = []
    for name in names_order:
        point = by_name.get(name)
        if point is None:
            ordered.append({"name": name, "x": 0.0, "y": 0.0, "score": 0.0, "v": 0})
            continue

        ordered.append(
            {
                "name": name,
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "score": float(point.get("score", 0.0)),
                "v": int(point.get("v", 2)),
            }
        )

    return ordered

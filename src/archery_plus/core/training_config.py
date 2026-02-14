from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from archery_plus.config import BASE_DIR, CONFIG_FILES, DEFAULT_PROJECT_DIR, WEIGHT_FILES
from archery_plus.core.mmpose_training import MMPoseDatasetBuildResult
from archery_plus.pipelines.auto_annotator_halpe26 import HALPE26_NAMES
from archery_plus.pipelines.auto_annotator_rtmo_archery import ARCHERY_KEYPOINTS

TRAIN_TASK_OPTIONS = ["关键点检测"]
TRAIN_TARGET_OPTIONS = ["人体关键点", "弓箭关键点"]
INIT_MODE_OPTIONS = ["加载预训练权重", "从头初始化"]
OPTIMIZER_OPTIONS = ["AdamW", "SGD"]
SCHEDULER_OPTIONS = ["CosineAnnealingLR", "MultiStepLR"]

MODEL_SPECS: dict[str, dict[str, Path]] = {
    "RTMO-s": {
        "config": BASE_DIR / CONFIG_FILES["mmpose_default_rtmo"],
        "weight": BASE_DIR / WEIGHT_FILES["archery_keypoints_pth"],
    },
    "RTMO-s(项目配置)": {
        "config": BASE_DIR / CONFIG_FILES["archery_rtmo_config"],
        "weight": BASE_DIR / WEIGHT_FILES["archery_keypoints_pth"],
    },
    "RTMPose-X(HALPE26)": {
        "config": BASE_DIR / CONFIG_FILES["halpe26_config"],
        "weight": BASE_DIR / WEIGHT_FILES["human_halpe26_pth"],
    },
}


@dataclass
class TrainingPreset:
    project_dir: Path
    dataset_dir: Path
    task_type: str = "关键点检测"
    model_name: str = "RTMO-s"
    target_name: str = "人体关键点"
    init_mode: str = "加载预训练权重"
    pretrained_weight_path: str = ""
    base_config_path: str = ""
    human_category_name: str = "human"
    archery_category_name: str = "archery"
    epochs: int = 300
    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 640
    learning_rate: float = 0.004
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    gpus: int = 1


@dataclass
class GeneratedTrainingConfig:
    config_path: Path
    command: str
    work_dir: Path
    target_name: str
    num_keypoints: int


def resolve_model_assets(model_name: str) -> tuple[Path, Path]:
    spec = MODEL_SPECS.get(model_name) or MODEL_SPECS["RTMO-s"]
    return spec["config"], spec["weight"]


def default_training_preset(project_root: Path | None = None) -> TrainingPreset:
    root = project_root or DEFAULT_PROJECT_DIR
    config_path, weight_path = resolve_model_assets("RTMO-s")
    return TrainingPreset(
        project_dir=root,
        dataset_dir=root / "output" / "mmpose_dataset_v1",
        model_name="RTMO-s",
        task_type="关键点检测",
        target_name="人体关键点",
        init_mode="加载预训练权重",
        pretrained_weight_path=str(weight_path),
        base_config_path=str(config_path),
        epochs=300,
        batch_size=16,
        num_workers=4,
        image_size=640,
        learning_rate=0.004,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        gpus=1,
        human_category_name="human",
        archery_category_name="archery",
    )


def save_training_preset(path: Path, preset: TrainingPreset) -> None:
    payload = {
        "project_dir": str(preset.project_dir),
        "dataset_dir": str(preset.dataset_dir),
        "task_type": preset.task_type,
        "model_name": preset.model_name,
        "target_name": preset.target_name,
        "init_mode": preset.init_mode,
        "pretrained_weight_path": preset.pretrained_weight_path,
        "base_config_path": preset.base_config_path,
        "human_category_name": preset.human_category_name,
        "archery_category_name": preset.archery_category_name,
        "epochs": int(preset.epochs),
        "batch_size": int(preset.batch_size),
        "num_workers": int(preset.num_workers),
        "image_size": int(preset.image_size),
        "learning_rate": float(preset.learning_rate),
        "optimizer": preset.optimizer,
        "scheduler": preset.scheduler,
        "train_ratio": float(preset.train_ratio),
        "val_ratio": float(preset.val_ratio),
        "test_ratio": float(preset.test_ratio),
        "random_seed": int(preset.random_seed),
        "gpus": int(preset.gpus),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_training_preset(path: Path, fallback_project_root: Path | None = None) -> TrainingPreset:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid preset format: {path}")

    default = default_training_preset(fallback_project_root)
    return TrainingPreset(
        project_dir=Path(str(raw.get("project_dir", default.project_dir))),
        dataset_dir=Path(str(raw.get("dataset_dir", default.dataset_dir))),
        task_type=str(raw.get("task_type", default.task_type)),
        model_name=str(raw.get("model_name", default.model_name)),
        target_name=str(raw.get("target_name", default.target_name)),
        init_mode=str(raw.get("init_mode", default.init_mode)),
        pretrained_weight_path=str(raw.get("pretrained_weight_path", default.pretrained_weight_path)),
        base_config_path=str(raw.get("base_config_path", default.base_config_path)),
        human_category_name=str(raw.get("human_category_name", default.human_category_name)),
        archery_category_name=str(raw.get("archery_category_name", default.archery_category_name)),
        epochs=int(raw.get("epochs", default.epochs)),
        batch_size=int(raw.get("batch_size", default.batch_size)),
        num_workers=int(raw.get("num_workers", default.num_workers)),
        image_size=int(raw.get("image_size", default.image_size)),
        learning_rate=float(raw.get("learning_rate", default.learning_rate)),
        optimizer=str(raw.get("optimizer", default.optimizer)),
        scheduler=str(raw.get("scheduler", default.scheduler)),
        train_ratio=float(raw.get("train_ratio", default.train_ratio)),
        val_ratio=float(raw.get("val_ratio", default.val_ratio)),
        test_ratio=float(raw.get("test_ratio", default.test_ratio)),
        random_seed=int(raw.get("random_seed", default.random_seed)),
        gpus=int(raw.get("gpus", default.gpus)),
    )


def generate_training_config(
    preset: TrainingPreset,
    dataset_result: MMPoseDatasetBuildResult,
) -> GeneratedTrainingConfig:
    project_dir = Path(preset.project_dir)
    config_dir = project_dir / "output" / "train_configs"
    run_root = project_dir / "output" / "train_runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    base_config = Path(preset.base_config_path.strip()) if preset.base_config_path.strip() else resolve_model_assets(preset.model_name)[0]
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    target_key = "archery" if preset.target_name == "弓箭关键点" else "human"
    if target_key == "archery":
        ann_paths = dataset_result.archery_paths
        keypoint_names = ARCHERY_KEYPOINTS
        category_name = preset.archery_category_name.strip() or "archery"
    else:
        ann_paths = dataset_result.human_paths
        keypoint_names = HALPE26_NAMES
        category_name = preset.human_category_name.strip() or "human"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _slugify(preset.model_name)
    config_path = config_dir / f"{timestamp}_{model_slug}_{target_key}.py"
    work_dir = run_root / f"{model_slug}_{target_key}"

    max_epochs = max(1, int(preset.epochs))
    val_interval = max(1, min(20, max_epochs // 10 if max_epochs >= 10 else 1))
    lr = max(1e-7, float(preset.learning_rate))
    gpus = max(0, int(preset.gpus))

    scheduler_block = _build_scheduler_block(preset.scheduler, max_epochs)
    keypoint_names_literal = json.dumps(keypoint_names, ensure_ascii=False)

    load_from_expr = "None"
    if preset.init_mode == "加载预训练权重" and preset.pretrained_weight_path.strip():
        load_from_expr = f"r'{_path_to_posix(Path(preset.pretrained_weight_path.strip()))}'"

    model_override_lines = [
        "model = dict(",
        "    head=dict(",
        "        num_keypoints=len(target_keypoint_names),",
        "        assigner=dict(oks_calculator=dict(metainfo=metafile)),",
        "        loss_oks=dict(metainfo=metafile),",
        "    ),",
        ")",
    ]
    if preset.init_mode == "从头初始化":
        model_override_lines = [
            "model = dict(",
            "    init_cfg=None,",
            "    backbone=dict(init_cfg=None),",
            "    head=dict(",
            "        num_keypoints=len(target_keypoint_names),",
            "        assigner=dict(oks_calculator=dict(metainfo=metafile)),",
            "        loss_oks=dict(metainfo=metafile),",
            "    ),",
            ")",
        ]
    model_override = "\n".join(model_override_lines)

    config_text = f"""
_base_ = r'{_path_to_posix(base_config)}'

data_root = r'{_path_to_posix(project_dir)}'
train_ann_file = r'{_path_to_posix(ann_paths.train)}'
val_ann_file = r'{_path_to_posix(ann_paths.val)}'
test_ann_file = r'{_path_to_posix(ann_paths.test)}'
target_keypoint_names = {keypoint_names_literal}

auto_scale_lr = dict(enable=False)

metafile = dict(
    dataset_name='archery_plus_{target_key}_{category_name}',
    keypoint_info={{
        i: dict(name=name, id=i, color=[51, 153, 255], type='', swap='')
        for i, name in enumerate(target_keypoint_names)
    }},
    skeleton_info={{}},
    joint_weights=[1.0 for _ in target_keypoint_names],
    sigmas=[0.05 for _ in target_keypoint_names],
)

input_size = ({int(preset.image_size)}, {int(preset.image_size)})
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_cfg = dict(max_epochs={max_epochs}, val_interval={val_interval})
optim_wrapper = dict(optimizer=dict(type='{preset.optimizer}', lr={lr}))
param_scheduler = {scheduler_block}

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval={max(1, max_epochs // 5)}, max_keep_ckpts=3))

train_dataloader = dict(
    batch_size={max(1, int(preset.batch_size))},
    num_workers={max(0, int(preset.num_workers))},
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=''),
        metainfo=metafile,
    ),
)

val_dataloader = dict(
    num_workers={max(0, int(preset.num_workers))},
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=''),
        metainfo=metafile,
    ),
)

test_dataloader = dict(
    num_workers={max(0, int(preset.num_workers))},
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=''),
        metainfo=metafile,
    ),
)

val_evaluator = dict(
    ann_file=val_ann_file,
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = dict(
    ann_file=test_ann_file,
    score_mode='bbox',
    nms_mode='none',
)

{model_override}

load_from = {load_from_expr}
resume = False
work_dir = r'{_path_to_posix(work_dir)}'
""".strip() + "\n"

    with config_path.open("w", encoding="utf-8") as f:
        f.write(config_text)

    command = build_train_command(config_path=config_path, work_dir=work_dir, gpus=gpus)
    return GeneratedTrainingConfig(
        config_path=config_path,
        command=command,
        work_dir=work_dir,
        target_name=preset.target_name,
        num_keypoints=len(keypoint_names),
    )


def build_train_command(config_path: Path, work_dir: Path, gpus: int = 1) -> str:
    gpu_num = max(0, int(gpus))
    return (
        f'python -m mim train mmpose "{_path_to_posix(config_path)}" '
        f'--work-dir "{_path_to_posix(work_dir)}" --gpus {gpu_num}'
    )


def _build_scheduler_block(scheduler_name: str, max_epochs: int) -> str:
    warmup_end = min(5, max_epochs)
    if scheduler_name == "MultiStepLR":
        m1 = max(1, int(max_epochs * 0.6))
        m2 = max(m1 + 1, int(max_epochs * 0.9))
        return (
            "["
            "dict(type='LinearLR', begin=0, end={we}, by_epoch=True, start_factor=0.001),"
            "dict(type='MultiStepLR', begin=0, end={me}, by_epoch=True, milestones=[{m1}, {m2}], gamma=0.1)"
            "]"
        ).format(we=warmup_end, me=max_epochs, m1=m1, m2=min(m2, max_epochs))

    cosine_tmax = max(1, max_epochs - warmup_end)
    return (
        "["
        "dict(type='LinearLR', begin=0, end={we}, by_epoch=True, start_factor=0.001),"
        "dict(type='CosineAnnealingLR', begin={we}, end={me}, T_max={tmax}, by_epoch=True, eta_min=1e-5)"
        "]"
    ).format(we=warmup_end, me=max_epochs, tmax=cosine_tmax)


def _slugify(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def _path_to_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def list_model_names() -> list[str]:
    return list(MODEL_SPECS.keys())


def list_training_options() -> dict[str, list[str]]:
    return {
        "task_types": list(TRAIN_TASK_OPTIONS),
        "targets": list(TRAIN_TARGET_OPTIONS),
        "init_modes": list(INIT_MODE_OPTIONS),
        "optimizers": list(OPTIMIZER_OPTIONS),
        "schedulers": list(SCHEDULER_OPTIONS),
        "models": list_model_names(),
    }


def parse_dataset_meta(dataset_meta_path: Path) -> dict[str, Any]:
    with dataset_meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid dataset meta file: {dataset_meta_path}")
    return payload

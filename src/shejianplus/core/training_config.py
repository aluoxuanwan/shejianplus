from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from shejianplus.config import BASE_DIR, CONFIG_FILES, DEFAULT_PROJECT_DIR, WEIGHT_FILES
from shejianplus.core.keypoint_schema import (
    TARGET_ARCHERY,
    TARGET_HUMAN,
    ensure_project_keypoint_sets,
    get_active_keypoint_set,
    normalize_target_name,
)
from shejianplus.core.mmpose_training import MMPoseDatasetBuildResult

TRAIN_TASK_OPTIONS = ["关键点检测"]
TRAIN_TARGET_OPTIONS = ["人体关键点", "弓箭关键点"]
INIT_MODE_OPTIONS = ["加载预训练权重", "从头初始化"]
OPTIMIZER_OPTIONS = ["AdamW", "SGD"]
SCHEDULER_OPTIONS = ["CosineAnnealingLR", "MultiStepLR"]

REQUIRED_MMPOSE_VERSION = "1.3.2"


def detect_mmpose_version() -> str | None:
    try:
        import mmpose  # type: ignore

        return str(getattr(mmpose, "__version__", "")).strip() or None
    except Exception:
        return None


def is_required_mmpose_version() -> tuple[bool, str | None]:
    current = detect_mmpose_version()
    return current == REQUIRED_MMPOSE_VERSION, current


def _mmpose_config_root() -> Path | None:
    try:
        import mmpose  # type: ignore
    except Exception:
        return None

    root = Path(mmpose.__file__).resolve().parent / ".mim" / "configs"
    return root if root.exists() else None


def _find_mmpose_config(*relative_candidates: str) -> Path | None:
    root = _mmpose_config_root()
    if root is None:
        return None

    for rel in relative_candidates:
        candidate = (root / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def _build_model_specs() -> dict[str, dict[str, Path | None]]:
    packaged_rtmo_s = _find_mmpose_config(
        "body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py",
    )
    packaged_rtmpose_x_halpe26 = _find_mmpose_config(
        "body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
    )

    return {
        "RTMO-s": {
            "config": packaged_rtmo_s or BASE_DIR / CONFIG_FILES["mmpose_default_rtmo"],
            "weight": BASE_DIR / WEIGHT_FILES["mmpose_rtmo_s_coco_pth"],
        },
        "RTMPose-X(HALPE26)": {
            "config": packaged_rtmpose_x_halpe26 or BASE_DIR / CONFIG_FILES["halpe26_config"],
            "weight": BASE_DIR / WEIGHT_FILES["human_halpe26_pth"],
        },
    }


MODEL_SPECS: dict[str, dict[str, Path | None]] = _build_model_specs()


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
    category_name: str = ""
    skeleton_info: str = ""
    joint_weights: str = ""
    sigmas: str = ""
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    input_width: int = 640
    input_height: int = 640
    image_size: int = 640  # legacy compatibility
    learning_rate: float = 0.0001
    aug_random_affine: bool = True
    aug_mosaic: bool = True
    aug_mixup: bool = True
    aug_hsv: bool = True
    aug_random_flip: bool = False
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


def resolve_model_assets(model_name: str) -> tuple[Path, Path | None]:
    spec = MODEL_SPECS.get(model_name) or MODEL_SPECS["RTMO-s"]
    config = spec.get("config")
    if not isinstance(config, Path) or not config.exists():
        raise RuntimeError(
            f"Model config path missing for model: {model_name}. "
            "Please ensure mmpose==1.3.2 is installed."
        )
    weight = spec.get("weight")
    return config, weight if isinstance(weight, Path) else None


def _default_category_name(project_root: Path, target_name: str) -> str:
    try:
        ensure_project_keypoint_sets(project_root)
        key_set = get_active_keypoint_set(project_root, target_name)
        return key_set.set_name
    except Exception:
        try:
            target_key = normalize_target_name(target_name)
        except Exception:
            target_key = TARGET_HUMAN
        return "archery" if target_key == TARGET_ARCHERY else "human"


def default_training_preset(project_root: Path | None = None) -> TrainingPreset:
    root = project_root or DEFAULT_PROJECT_DIR
    target_name = TRAIN_TARGET_OPTIONS[0]

    try:
        config_path, weight_path = resolve_model_assets("RTMO-s")
        base_config_path = str(config_path)
        pretrained_weight_path = str(weight_path) if weight_path else ""
    except Exception:
        base_config_path = ""
        pretrained_weight_path = ""

    return TrainingPreset(
        project_dir=root,
        dataset_dir=root / "output" / "mmpose_dataset_v1",
        model_name="RTMO-s",
        task_type=TRAIN_TASK_OPTIONS[0],
        target_name=target_name,
        init_mode=INIT_MODE_OPTIONS[0],
        pretrained_weight_path=pretrained_weight_path,
        base_config_path=base_config_path,
        category_name=_default_category_name(root, target_name),
        epochs=100,
        batch_size=16,
        num_workers=4,
        input_width=640,
        input_height=640,
        image_size=640,
        learning_rate=0.0001,
        aug_random_affine=True,
        aug_mosaic=True,
        aug_mixup=True,
        aug_hsv=True,
        aug_random_flip=False,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        gpus=1,
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
        "category_name": preset.category_name,
        "skeleton_info": preset.skeleton_info,
        "joint_weights": preset.joint_weights,
        "sigmas": preset.sigmas,
        "epochs": int(preset.epochs),
        "batch_size": int(preset.batch_size),
        "num_workers": int(preset.num_workers),
        "input_width": int(preset.input_width),
        "input_height": int(preset.input_height),
        "image_size": int(preset.input_width),
        "learning_rate": float(preset.learning_rate),
        "aug_random_affine": bool(preset.aug_random_affine),
        "aug_mosaic": bool(preset.aug_mosaic),
        "aug_mixup": bool(preset.aug_mixup),
        "aug_hsv": bool(preset.aug_hsv),
        "aug_random_flip": bool(preset.aug_random_flip),
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
    target_name = str(raw.get("target_name", default.target_name))

    category_name = str(raw.get("category_name", "")).strip()
    if not category_name:
        legacy_human = str(raw.get("human_category_name", "")).strip()
        legacy_archery = str(raw.get("archery_category_name", "")).strip()
        try:
            target_key = normalize_target_name(target_name)
        except Exception:
            target_key = TARGET_HUMAN
        category_name = legacy_archery if target_key == TARGET_ARCHERY else legacy_human
        if not category_name:
            category_name = _default_category_name(default.project_dir, target_name)

    model_name = str(raw.get("model_name", default.model_name)).strip()
    if model_name not in MODEL_SPECS:
        model_name = default.model_name

    legacy_size = int(raw.get("image_size", default.image_size))
    input_width = int(raw.get("input_width", legacy_size))
    input_height = int(raw.get("input_height", legacy_size))

    return TrainingPreset(
        project_dir=Path(str(raw.get("project_dir", default.project_dir))),
        dataset_dir=Path(str(raw.get("dataset_dir", default.dataset_dir))),
        task_type=str(raw.get("task_type", default.task_type)),
        model_name=model_name,
        target_name=target_name,
        init_mode=str(raw.get("init_mode", default.init_mode)),
        pretrained_weight_path=str(raw.get("pretrained_weight_path", default.pretrained_weight_path)),
        base_config_path=str(raw.get("base_config_path", default.base_config_path)),
        category_name=category_name,
        skeleton_info=str(raw.get("skeleton_info", default.skeleton_info)),
        joint_weights=str(raw.get("joint_weights", default.joint_weights)),
        sigmas=str(raw.get("sigmas", default.sigmas)),
        epochs=int(raw.get("epochs", default.epochs)),
        batch_size=int(raw.get("batch_size", default.batch_size)),
        num_workers=int(raw.get("num_workers", default.num_workers)),
        input_width=max(64, input_width),
        input_height=max(64, input_height),
        image_size=legacy_size,
        learning_rate=float(raw.get("learning_rate", default.learning_rate)),
        aug_random_affine=bool(raw.get("aug_random_affine", default.aug_random_affine)),
        aug_mosaic=bool(raw.get("aug_mosaic", default.aug_mosaic)),
        aug_mixup=bool(raw.get("aug_mixup", default.aug_mixup)),
        aug_hsv=bool(raw.get("aug_hsv", default.aug_hsv)),
        aug_random_flip=bool(raw.get("aug_random_flip", default.aug_random_flip)),
        optimizer=str(raw.get("optimizer", default.optimizer)),
        scheduler=str(raw.get("scheduler", default.scheduler)),
        train_ratio=float(raw.get("train_ratio", default.train_ratio)),
        val_ratio=float(raw.get("val_ratio", default.val_ratio)),
        test_ratio=float(raw.get("test_ratio", default.test_ratio)),
        random_seed=int(raw.get("random_seed", default.random_seed)),
        gpus=int(raw.get("gpus", default.gpus)),
    )


def _resolve_input_size(preset: TrainingPreset) -> tuple[int, int]:
    input_w = max(64, int(getattr(preset, "input_width", getattr(preset, "image_size", 640))))
    input_h = max(64, int(getattr(preset, "input_height", getattr(preset, "image_size", 640))))
    return (input_w, input_h)


def generate_training_config(
    preset: TrainingPreset,
    dataset_result: MMPoseDatasetBuildResult,
) -> GeneratedTrainingConfig:
    project_dir = Path(preset.project_dir)
    config_dir = project_dir / "output" / "train_configs"
    run_root = project_dir / "output" / "train_runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    ensure_project_keypoint_sets(project_dir)
    target_key = normalize_target_name(preset.target_name)
    keypoint_set = get_active_keypoint_set(project_dir, target_key)
    keypoint_names = keypoint_set.names

    base_config = _resolve_base_config_path(preset)

    if target_key == TARGET_ARCHERY:
        ann_paths = dataset_result.archery_paths
    else:
        ann_paths = dataset_result.human_paths

    category_name = preset.category_name.strip() or keypoint_set.set_name or target_key

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
    persistent_workers_literal = "True" if int(preset.num_workers) > 0 else "False"
    input_w, input_h = _resolve_input_size(preset)

    skeleton_info_literal = _build_skeleton_info_literal(preset.skeleton_info, keypoint_names)
    joint_weights_literal = _build_float_list_literal(preset.joint_weights, len(keypoint_names), default_value=1.0)
    sigmas_literal = _build_float_list_literal(preset.sigmas, len(keypoint_names), default_value=0.05)

    val_evaluator_literal = _build_evaluator_literal(base_config, evaluator_key="val_evaluator", ann_var="val_ann_file")
    test_evaluator_literal = _build_evaluator_literal(base_config, evaluator_key="test_evaluator", ann_var="test_ann_file")
    train_dataset_literal = _build_dataset_override_literal(
        base_config_path=base_config,
        dataloader_key="train_dataloader",
        ann_var="train_ann_file",
        input_size=(input_w, input_h),
        is_train=True,
        augment_flags={
            "random_affine": bool(preset.aug_random_affine),
            "mosaic": bool(preset.aug_mosaic),
            "mixup": bool(preset.aug_mixup),
            "hsv": bool(preset.aug_hsv),
            "random_flip": bool(preset.aug_random_flip),
        },
    )
    val_dataset_literal = _build_dataset_override_literal(
        base_config_path=base_config,
        dataloader_key="val_dataloader",
        ann_var="val_ann_file",
        input_size=(input_w, input_h),
    )
    test_dataset_literal = _build_dataset_override_literal(
        base_config_path=base_config,
        dataloader_key="test_dataloader",
        ann_var="test_ann_file",
        input_size=(input_w, input_h),
    )

    metainfo_dir = config_dir / "metainfo"
    metainfo_dir.mkdir(parents=True, exist_ok=True)
    metainfo_path = metainfo_dir / f"{timestamp}_{model_slug}_{target_key}_metainfo.py"
    metainfo_text = _build_pose_metainfo_file_text(
        dataset_name=f"archery_plus_{target_key}_{category_name}",
        keypoint_names=keypoint_names,
        skeleton_info_literal=skeleton_info_literal,
        joint_weights_literal=joint_weights_literal,
        sigmas_literal=sigmas_literal,
    )
    with metainfo_path.open("w", encoding="utf-8") as mf:
        mf.write(metainfo_text)

    load_from_expr = "None"
    if preset.init_mode == "加载预训练权重" and preset.pretrained_weight_path.strip():
        weight_raw = preset.pretrained_weight_path.strip()
        if _is_url(weight_raw):
            load_from_expr = f"r'{weight_raw}'"
        else:
            weight_path = Path(weight_raw)
            if weight_path.exists():
                load_from_expr = f"r'{_path_to_posix(weight_path)}'"
    head_type = _detect_head_type(base_config)

    if head_type == "RTMOHead":
        oks_type, loss_oks_type = _detect_rtmo_meta_types(base_config)
        head_override_lines = [
            "head=dict(",
            "    num_keypoints=len(target_keypoint_names),",
            f"    assigner=dict(oks_calculator=dict(_delete_=True, type='{oks_type}', metainfo=metafile)),",
            f"    loss_oks=dict(_delete_=True, type='{loss_oks_type}', metainfo=metafile),",
            "),",
        ]
    elif head_type == "RTMCCHead":
        head_override_lines = [
            "head=dict(",
            "    out_channels=len(target_keypoint_names),",
            "),",
        ]
    else:
        head_override_lines = [
            "head=dict(",
            "    out_channels=len(target_keypoint_names),",
            "    num_keypoints=len(target_keypoint_names),",
            "),",
        ]

    model_override_lines = ["model = dict("]
    if preset.init_mode == "从头初始化":
        model_override_lines.extend(
            [
                "    init_cfg=None,",
                "    backbone=dict(init_cfg=None),",
            ]
        )

    for line in head_override_lines:
        model_override_lines.append(f"    {line}" if not line.startswith("head=") else f"    {line}")
    model_override_lines.append(")")

    model_override = "\n".join(model_override_lines)

    config_text = f"""
_base_ = r'{_path_to_posix(base_config)}'

data_root = r'{_path_to_posix(project_dir)}'
train_ann_file = r'{_path_to_posix(ann_paths.train)}'
val_ann_file = r'{_path_to_posix(ann_paths.val)}'
test_ann_file = r'{_path_to_posix(ann_paths.test)}'
metafile = r'{_path_to_posix(metainfo_path)}'
target_keypoint_names = {keypoint_names_literal}

auto_scale_lr = dict(enable=False)

target_metainfo = dict(
    dataset_name='archery_plus_{target_key}_{category_name}',
    keypoint_info={{
        i: dict(name=name, id=i, color=[51, 153, 255], type='', swap='')
        for i, name in enumerate(target_keypoint_names)
    }},
    skeleton_info={skeleton_info_literal},
    joint_weights={joint_weights_literal},
    sigmas={sigmas_literal},
)

input_size = ({input_w}, {input_h})
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_cfg = dict(max_epochs={max_epochs}, val_interval={val_interval})
optim_wrapper = dict(optimizer=dict(type='{preset.optimizer}', lr={lr}))
param_scheduler = {scheduler_block}

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval={max(1, max_epochs // 5)}, max_keep_ckpts=3))

train_dataloader = dict(
    batch_size={max(1, int(preset.batch_size))},
    num_workers={max(0, int(preset.num_workers))},
    persistent_workers={persistent_workers_literal},
    dataset={train_dataset_literal},
)

val_dataloader = dict(
    num_workers={max(0, int(preset.num_workers))},
    persistent_workers={persistent_workers_literal},
    dataset={val_dataset_literal},
)

test_dataloader = dict(
    num_workers={max(0, int(preset.num_workers))},
    persistent_workers={persistent_workers_literal},
    dataset={test_dataset_literal},
)

val_evaluator = {val_evaluator_literal}
test_evaluator = {test_evaluator_literal}

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
    train_script = _resolve_mmpose_train_script()
    if train_script is not None and train_script.exists():
        return (
            f'python "{_path_to_posix(train_script)}" '
            f'"{_path_to_posix(config_path)}" --launcher none '
            f'--work-dir "{_path_to_posix(work_dir)}"'
        )

    # Fallback to OpenMIM if direct script cannot be resolved.
    gpu_num = max(0, int(gpus))
    return (
        f'python -m mim train mmpose "{_path_to_posix(config_path)}" '
        f'--work-dir "{_path_to_posix(work_dir)}" --gpus {gpu_num}'
    )


def _resolve_mmpose_train_script() -> Path | None:
    # Prefer resolving from the current interpreter path to avoid importing
    # mmpose/torch here (which may fail on low pagefile environments).
    py = Path(sys.executable).resolve()
    venv_root = py.parent.parent
    site_pkg = venv_root / 'Lib' / 'site-packages'
    candidates = [
        site_pkg / 'mmpose' / '.mim' / 'tools' / 'train.py',
        site_pkg / 'mmpose' / 'tools' / 'train.py',
    ]
    for script in candidates:
        if script.exists():
            return script

    # Fallback to import-based discovery when path inference misses.
    try:
        import mmpose  # type: ignore

        package_root = Path(mmpose.__file__).resolve().parent
        for script in [package_root / '.mim' / 'tools' / 'train.py', package_root / 'tools' / 'train.py']:
            if script.exists():
                return script
    except Exception:
        pass

    return None


def _build_pose_metainfo_file_text(
    dataset_name: str,
    keypoint_names: list[str],
    skeleton_info_literal: str,
    joint_weights_literal: str,
    sigmas_literal: str,
) -> str:
    keypoint_lines = []
    for i, name in enumerate(keypoint_names):
        safe_name = str(name).replace("'", "\'")
        keypoint_lines.append(
            f"        {i}: dict(name='{safe_name}', id={i}, color=[51, 153, 255], type='', swap=''),"
        )
    keypoint_block = "\n".join(keypoint_lines)

    safe_dataset_name = dataset_name.replace("'", "\'")
    return (
        "dataset_info = dict(\n"
        f"    dataset_name='{safe_dataset_name}',\n"
        "    keypoint_info={\n"
        f"{keypoint_block}\n"
        "    },\n"
        f"    skeleton_info={skeleton_info_literal},\n"
        f"    joint_weights={joint_weights_literal},\n"
        f"    sigmas={sigmas_literal},\n"
        ")\n"
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


def _resolve_base_config_path(preset: TrainingPreset) -> Path:
    model_default_config, _ = resolve_model_assets(preset.model_name)

    candidates: list[Path] = []
    raw_user = preset.base_config_path.strip()
    if raw_user:
        candidates.append(Path(raw_user))
    candidates.append(model_default_config)

    checked: list[str] = []
    for candidate in candidates:
        c = candidate.resolve() if candidate.exists() else candidate
        checked.append(str(c))
        if not candidate.exists():
            continue
        if _is_config_loadable(candidate):
            return candidate

    raise FileNotFoundError(
        "No usable base config found. Checked: " + ", ".join(checked)
    )


def _is_config_loadable(config_path: Path) -> bool:
    try:
        from mmengine.config import Config

        Config.fromfile(str(config_path))
        return True
    except Exception:
        return False




def _detect_rtmo_meta_types(config_path: Path) -> tuple[str, str]:
    oks_type = "PoseOKS"
    loss_oks_type = "OKSLoss"
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            head_cfg = model_cfg.get("head", {})
            if isinstance(head_cfg, dict):
                assigner = head_cfg.get("assigner", {})
                if isinstance(assigner, dict):
                    oks_cfg = assigner.get("oks_calculator", {})
                    if isinstance(oks_cfg, dict):
                        t1 = str(oks_cfg.get("type", "")).strip()
                        if t1:
                            oks_type = t1
                loss_oks = head_cfg.get("loss_oks", {})
                if isinstance(loss_oks, dict):
                    t2 = str(loss_oks.get("type", "")).strip()
                    if t2:
                        loss_oks_type = t2
    except Exception:
        pass
    return oks_type, loss_oks_type

def _build_skeleton_info_literal(raw_text: str, keypoint_names: list[str]) -> str:
    default = {
        i: {
            "link": [keypoint_names[i], keypoint_names[i + 1]],
            "id": i,
            "color": [51, 153, 255],
        }
        for i in range(max(0, len(keypoint_names) - 1))
    }
    if not raw_text.strip():
        return str(default)

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return str(default)

    if not isinstance(parsed, dict):
        return str(default)

    normalized: dict[int, dict[str, Any]] = {}
    next_id = 0
    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue

        link = value.get("link")
        if not isinstance(link, (list, tuple)) or len(link) != 2:
            continue

        p0 = str(link[0]).strip()
        p1 = str(link[1]).strip()
        if not p0 or not p1:
            continue

        try:
            key_id = int(key)
        except Exception:
            key_id = next_id

        raw_color = value.get("color", [51, 153, 255])
        if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
            color = [51, 153, 255]
        else:
            color = [int(raw_color[0]), int(raw_color[1]), int(raw_color[2])]

        normalized[key_id] = {
            "link": [p0, p1],
            "id": int(value.get("id", key_id)),
            "color": color,
        }
        next_id += 1

    if not normalized:
        return str(default)
    return str(normalized)


def _build_float_list_literal(raw_text: str, count: int, default_value: float) -> str:
    default_list = [float(default_value) for _ in range(max(0, count))]
    text = raw_text.strip()
    if not text:
        return str(default_list)

    parsed_list: list[float] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed_list = [float(x) for x in parsed]
    except Exception:
        items = [x.strip() for x in text.split(",") if x.strip()]
        try:
            parsed_list = [float(x) for x in items]
        except Exception:
            parsed_list = []

    if not parsed_list:
        return str(default_list)

    if len(parsed_list) < count:
        parsed_list.extend([float(default_value)] * (count - len(parsed_list)))
    elif len(parsed_list) > count:
        parsed_list = parsed_list[:count]

    return str([float(x) for x in parsed_list])


def _build_evaluator_literal(base_config_path: Path, evaluator_key: str, ann_var: str) -> str:
    container_type, _ = _detect_evaluator_type(base_config_path, evaluator_key)
    item = f"dict(type='CocoMetric', ann_file={ann_var}, score_mode='bbox', nms_mode='none')"
    if container_type == "list":
        return f"[{item}]"
    return item


def _build_dataset_override_literal(
    base_config_path: Path,
    dataloader_key: str,
    ann_var: str,
    input_size: tuple[int, int],
    is_train: bool = False,
    augment_flags: dict[str, bool] | None = None,
) -> str:
    dataset_type, data_mode, pipeline_obj, test_mode = _detect_dataset_template(base_config_path, dataloader_key)
    pipeline_obj = _adapt_pipeline_input_size(pipeline_obj, input_size)
    if is_train:
        pipeline_obj = _filter_train_pipeline_augmentations(pipeline_obj, augment_flags or {})

    lines = [
        "dict(",
        "    _delete_=True,",
        f"    type='{dataset_type}',",
    ]
    if data_mode:
        lines.append(f"    data_mode='{data_mode}',")
    lines.extend(
        [
            "    data_root=data_root,",
            f"    ann_file={ann_var},",
            "    data_prefix=dict(img=''),",
        ]
    )
    if pipeline_obj is not None:
        lines.append(f"    pipeline={repr(pipeline_obj)},")
    if test_mode is not None:
        lines.append(f"    test_mode={'True' if bool(test_mode) else 'False'},")
    lines.append("    metainfo=target_metainfo,")
    lines.append(")")
    return "\n".join(lines)


def _detect_dataset_template(
    config_path: Path,
    dataloader_key: str,
) -> tuple[str, str | None, Any | None, bool | None]:
    default_type = "CocoDataset"
    default_mode = "topdown"
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
        dataloader = cfg.get(dataloader_key, {})
        if not isinstance(dataloader, dict):
            return default_type, default_mode, None, None

        dataset_cfg = dataloader.get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            return default_type, default_mode, None, None

        test_mode: bool | None = None
        if "test_mode" in dataset_cfg:
            test_mode = bool(dataset_cfg.get("test_mode"))

        dataset_type = str(dataset_cfg.get("type", "")).strip()
        data_mode = str(dataset_cfg.get("data_mode", "")).strip() or default_mode
        pipeline_obj: Any | None = dataset_cfg.get("pipeline")

        if dataset_type == "CombinedDataset":
            # CombinedDataset in official RTMPose configs often wraps CocoWholeBodyDataset,
            # which expects extra fields (foot/face/hand keypoints). Our exported data uses
            # plain COCO keypoints, so force generic CocoDataset for compatibility.
            first_dataset: dict[str, Any] = {}
            datasets = dataset_cfg.get("datasets")
            if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
                first_dataset = datasets[0]

            dataset_type = default_type
            if not data_mode:
                data_mode = str(first_dataset.get("data_mode", "")).strip() or default_mode
            if pipeline_obj is None:
                pipeline_obj = first_dataset.get("pipeline")
            if test_mode is None and "test_mode" in first_dataset:
                test_mode = bool(first_dataset.get("test_mode"))
        else:
            dataset_type = dataset_type or default_type

        return dataset_type, (data_mode or None), pipeline_obj, test_mode
    except Exception:
        return default_type, default_mode, None, None


def _adapt_pipeline_input_size(pipeline_obj: Any, input_size: tuple[int, int]) -> Any:
    if pipeline_obj is None:
        return None
    size = (int(input_size[0]), int(input_size[1]))

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "input_size" and isinstance(value, (list, tuple)) and len(value) == 2:
                    out[key] = size
                else:
                    out[key] = _walk(value)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_walk(item) for item in node)
        return node

    return _walk(pipeline_obj)


def _filter_train_pipeline_augmentations(pipeline_obj: Any, augment_flags: dict[str, bool]) -> Any:
    if not isinstance(pipeline_obj, list):
        return pipeline_obj

    enable_map = {
        "BottomupRandomAffine": bool(augment_flags.get("random_affine", True)),
        "Mosaic": bool(augment_flags.get("mosaic", True)),
        "YOLOXMixUp": bool(augment_flags.get("mixup", True)),
        "YOLOXHSVRandomAug": bool(augment_flags.get("hsv", True)),
        "RandomFlip": bool(augment_flags.get("random_flip", False)),
    }

    filtered: list[Any] = []
    for step in pipeline_obj:
        if isinstance(step, dict):
            step_type = str(step.get("type", "")).strip()
            if step_type in enable_map and not enable_map[step_type]:
                continue
        filtered.append(step)
    return filtered


def _detect_evaluator_type(config_path: Path, evaluator_key: str) -> tuple[str, str | None]:
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
        evaluator = cfg.get(evaluator_key)
    except Exception:
        return "dict", None

    if isinstance(evaluator, list):
        eval_type = None
        if evaluator and isinstance(evaluator[0], dict):
            t = str(evaluator[0].get("type", "")).strip()
            eval_type = t or None
        return "list", eval_type

    if isinstance(evaluator, dict):
        t = str(evaluator.get("type", "")).strip()
        return "dict", (t or None)

    return "dict", None

def _detect_head_type(config_path: Path) -> str:
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            head_cfg = model_cfg.get("head", {})
            if isinstance(head_cfg, dict):
                head_type = str(head_cfg.get("type", "")).strip()
                if head_type:
                    return head_type
    except Exception:
        pass
    return ""


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


def _is_url(value: str) -> bool:
    low = value.lower()
    return low.startswith("http://") or low.startswith("https://")


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




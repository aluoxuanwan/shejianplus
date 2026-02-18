from __future__ import annotations

from pathlib import Path

APP_NAME = "射箭Plus"
APP_VERSION = "0.1.0-mvp"

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_DIR = BASE_DIR / "workspace" / "demo_project"

MODEL_FILES = {
    "archery_keypoints_onnx": "rtmo-s_640-8x32-600e.onnx",
    "human_halpe26_onnx": "rtmpose-x_8xb256-700e_body8-halpe26-384x288.onnx",
}

WEIGHT_FILES = {
    "archery_keypoints_pth": "rtmo-s_640-8x32-600e.pth",
    "mmpose_rtmo_s_coco_pth": "rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth",
    "human_halpe26_pth": "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth",
}

CONFIG_FILES = {
    "archery_rtmo_config": "rtmo-s_640-8x32-600e.py",
    "halpe26_config": "rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
    "mmpose_default_rtmo": "rtmo-s_8xb32-600e_coco-640x640.py",
}

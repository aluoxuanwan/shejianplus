# 射箭Plus（初期实现骨架）

当前版本是按 PRD 落地的 `v0.1.0-mvp` 开发骨架，目标是先打通项目工程结构与 5 模块界面流程，并接入模型资产检查。

## 已实现内容

- 桌面框架：`PySide6` 主窗口 + 左侧 5 模块导航
- 模型资产扫描：
  - `rtmo-s_640-8x32-600e.onnx`
  - `rtmpose-x_8xb256-700e_body8-halpe26-384x288.onnx`
  - 相关 `.pth` / `.py` 配置文件可见性检查
- 运行时探测：
  - ONNX Runtime Provider 列表
  - OpenCV CUDA 设备数量
- 导入模块 MVP：
  - 项目目录结构一键初始化（`images/annotations/models/calibration/output`）
  - 视频路径与抽帧参数入口（抽帧流程为占位，待接入 OpenCV）

## 目录

```text
.
├── run.py
├── requirements.txt
└── src/
    └── archery_plus/
        ├── main.py
        ├── config.py
        ├── core/
        ├── services/
        └── ui/
```

## 运行

1. 安装依赖

```powershell
pip install -r requirements.txt
```

2. 启动应用

```powershell
python run.py
```

## 当前状态说明

- 标注、训练、应用、导出模块已建立参数面板与日志区域（占位）
- 下一步优先接入：
  1. OpenCV 抽帧
  2. Halpe26 自动预标注推理
  3. 训练任务子进程调度
  4. 摄像头实时推理与 CSV 导出

## Training Requirements (v0.3)

- `mmpose==1.3.2` is required for GUI training. If version mismatch is detected, config generation and training start are blocked.
- Keypoint-set JSON import is supported in both Annotate and Train pages.
- Keypoint-set files are copied into `<project>/config/keypoint_sets/`.
- Index file: `<project>/config/keypoint_sets/project_keypoint_sets.json`.
- JSON schema (v1):

```json
{
  "schema_version": 1,
  "set_name": "archery_5_v1",
  "keypoints": [
    {"id": 0, "name": "UP"},
    {"id": 1, "name": "DOWN"},
    {"id": 2, "name": "FL"},
    {"id": 3, "name": "ST"},
    {"id": 4, "name": "FS"}
  ]
}
```

- Manual annotation format is unchanged:

```json
{"human_keypoints": [...], "archery_keypoints": [...]}
```


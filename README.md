# 射箭Plus（ArcheryPlus）

射箭Plus 是一个面向“射箭动作数据生产与模型训练”的桌面工具，当前阶段聚焦 `2D MVP`：
- 数据导入与项目初始化
- 关键点/矩形标注
- ONNX 自动预标注（人体 + 弓箭）
- 基于 MMPose 的训练配置生成与训练启动

## 1. 当前能力

1. 模块1 导入素材
- 初始化项目目录
- 加载图片列表（`<project>/images/cam1`）

2. 模块2 标注图片
- 关键点标注（人体/弓箭）
- 目标框标注（bbox）
- 标签 JSON 导入与新增标签
- 自动保存开关
- 自动预标注（Halpe26 + RTMO）
- 批量自动标注（全部图片）

3. 模块3 训练模型
- 训练参数 GUI 配置
- 构建 MMPose 数据集（train/val/test）
- 生成训练配置
- 填充训练命令并启动训练
- 实时日志、loss/precision 曲线、训练状态（含 ETA）

4. 模块4 应用模型（Beta）
- 视频文件实时推理（RTMO 弓箭关键点 / RTMPose-HALPE26 人体关键点）
- 支持双模型同时推理（弓箭 + 人体）
- 单位换算（px -> cm，弓箭 UP/DOWN 标尺）
- 实时平滑滤波：
  - `One Euro`（一欧元滤波器）
  - 双向四阶巴特沃斯（滚动窗口近实时）
- 参数分析面板：速度、轨迹、两点/三点/四点角度
- 实时帧数显示：`推理FPS` 与 `视频FPS`
- 应用内支持导出“推理时序 CSV”（raw/filter）

5. 模块5 导出数据（v0.2）
- 标注数据导出：
  - `raw_2d.csv`
  - `filtered_2d.csv`（v0.1 占位导出，标记 `filter_method`）
  - `bbox_2d.csv`
  - `annotations_export.json`
  - `label_schema_export.json`
- 推理时序导出（导出模块内统一入口）：
  - `raw_2d_timeseries.csv`
  - `filtered_2d_timeseries.csv`
  - `session_meta.json`
  - 导出范围筛选：仅弓箭 / 仅人体 / 双模型全部

## 2. 环境准备（uv）

以下命令在 PowerShell 中执行，项目目录以 `D:\work\shejianplus` 为例。

### 2.1 创建并激活虚拟环境

```powershell
cd D:\work\shejianplus
uv venv .venv
.\.venv\Scripts\activate
```

激活后终端一般会显示：`(.venv) PS D:\work\shejianplus>`。

### 2.2 安装基础依赖（GUI + ONNX）

```powershell
pip install -r requirements.txt
```

### 2.3 安装训练依赖（MMPose 1.3.2）

```powershell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
pip install mmpose==1.3.2
pip install xtcocotools
```

### 2.4 验证关键版本

```powershell
python -c "import mmpose; print(mmpose.__version__)"
```

应输出：`1.3.2`。

## 3. 启动应用

```powershell
python run.py
```

## 4. 模型与配置文件准备

请确认项目根目录存在以下文件（至少对应你要跑的流程）：

- RTMO（弓箭）
  - `rtmo-s_640-8x32-600e.onnx`
  - `rtmo-s_8xb32-600e_coco-640x640.py`
  - `rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth`

- RTMPose（人体 Halpe26）
  - `rtmpose-x_8xb256-700e_body8-halpe26-384x288.onnx`
  - `rtmpose-x_8xb256-700e_body8-halpe26-384x288.py`
  - `rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth`

## 5. 标注数据与标签文件

### 5.1 每张图标注结构

```json
{
  "human_keypoints": [...],
  "archery_keypoints": [...],
  "bboxes": [...]
}
```

手工点默认 `score=0`。

### 5.2 关键点集合 JSON（推荐）

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

规则：`id` 必须连续 `0..N-1`。

## 6. 训练通用流程（GUI）

1. 在“导入素材”完成项目目录初始化，并保证图片在 `<project>/images/cam1`。
2. 在“标注图片”完成标注（可手工、可自动预标注、可混合）。
3. 在“训练模型”：
- 配置模型、目标、初始化方式、训练参数
- 点击 `生成MMPose数据`
- 点击 `生成训练配置`
- 点击 `填充训练命令`
- 点击 `开始训练`

## 7. 示例A：RTMPose-X（HALPE26 人体关键点）

### 7.1 标注阶段

1. 打开“标注图片”。
2. 编辑目标切换到“人体关键点”。
3. 关键点集合使用 Halpe26（26点）。
4. 可点击“人体预标注”或“全部图片自动标注”，再人工修正。

### 7.2 训练阶段推荐配置

在“训练模型”中设置：
- 模型：`RTMPose-X(HALPE26)`
- 目标：`人体关键点`
- 初始化：`加载预训练权重`
- 类别名称：例如 `human`
- `batch_size=16`
- `num_workers=4`
- 数据划分：`train=0.8 val=0.1 test=0.1`
- `input_size`：默认会带出 `288x384`，可按需要改

然后依次点击：
- `生成MMPose数据`
- `生成训练配置`
- `填充训练命令`
- `开始训练`

示例命令（自动填充后类似）：

```powershell
python -m mim train mmpose "D:/work/shejianplus/workspace/demo_project/output/train_configs/xxxx_rtmpose_x_halpe26_human.py" --work-dir "D:/work/shejianplus/workspace/demo_project/output/train_runs/rtmpose_x_halpe26_human" --gpus 1
```

## 8. 示例B：RTMO-s（弓箭关键点）

### 8.1 标注阶段

1. 打开“标注图片”。
2. 编辑目标切换到“弓箭关键点”。
3. 关键点集合使用 5 点（`UP/DOWN/FL/ST/FS`）。
4. 可点击“弓箭预标注”或“全部图片自动标注”，再人工修正。

### 8.2 训练阶段推荐配置

在“训练模型”中设置：
- 模型：`RTMO-s`
- 目标：`弓箭关键点`
- 初始化：`加载预训练权重`
- 预训练权重应为：`rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth`
- 类别名称：例如 `bow`
- `batch_size=16`
- `num_workers=4`
- 数据划分：`train=0.8 val=0.1 test=0.1`
- `input_size`：默认 `640x640`，可改

然后依次点击：
- `生成MMPose数据`
- `生成训练配置`
- `填充训练命令`
- `开始训练`

示例命令（自动填充后类似）：

```powershell
python -m mim train mmpose "D:/work/shejianplus/workspace/demo_project/output/train_configs/xxxx_rtmo_s_archery.py" --work-dir "D:/work/shejianplus/workspace/demo_project/output/train_runs/rtmo_s_archery" --gpus 1
```

## 9. 训练状态说明（ETA）

训练状态中的 `ETA` 是 `Estimated Time of Arrival`，表示“预计剩余训练时间”。
示例：`ETA=01:23:45` 表示大约还需 1 小时 23 分 45 秒。

## 10. 常见问题

1. 训练被版本门禁拦截
- 现象：提示 `mmpose` 版本不满足。
- 处理：确保 `python -c "import mmpose; print(mmpose.__version__)"` 输出 `1.3.2`。

2. `python` 命令找不到
- 处理：先激活 `.venv` 环境，再执行 `python run.py`。

3. Windows 编码相关报错
- 当前 GUI 已做子进程兼容处理；若仍出现异常，先重启 GUI 再试。

4. 关键点标签看起来错位
- 确认当前标签集合 id 连续且与训练目标一致。
- 建议先“全部图片自动标注”后再人工修正。

## 11. 应用模块使用（推理）

1. 进入“应用模型”页面。
2. 选择推理目标：
- `弓箭关键点（RTMO）`
- `人体关键点（RTMPose-HALPE26）`
- `双模型（弓箭 + 人体）`
3. 选择视频文件并开始推理。
4. 推理结束后如需导出时序数据，请到“导出数据”模块统一导出。

## 12. 导出模块使用（标注导出 + 推理时序导出）

1. 进入“导出数据”页面。
2. 选择项目目录（包含 `annotations/` 与 `images/`）。
3. 标注导出：勾选导出项后点击 `开始导出`。
4. 推理时序导出：
   - 先在“应用模型”完成一次推理
   - 回到“导出数据”页面，选择导出范围（仅弓箭/仅人体/双模型全部）
   - 点击 `导出推理时序CSV`
5. 输出目录位置：`<project>/output/exports/<时间戳>/`

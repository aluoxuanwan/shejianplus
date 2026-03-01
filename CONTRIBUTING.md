# 射箭Plus 协作规范

本文档用于约定 `shejianplus` 项目的双人协作方式，目标是让我们在保持开发速度的同时，尽量减少冲突、返工和误提交。

## 1. 分支职责

- `main`：只保留可运行、相对稳定的版本。
- `dev`：日常集成分支，功能开发先合入这里。
- `feature/xxx`：单个功能或单个问题修复分支，例如 `feature/export-ui`、`feature/train-log-panel`。

日常原则：
- 不直接在 `main` 上开发。
- 不长期在 `dev` 上直接堆提交。
- 一个功能尽量只在一个 `feature/*` 分支中完成。

## 2. 日常协作流程

每个功能都建议按下面的顺序执行。

### 2.1 同步 `dev`

```bash
git checkout dev
git pull
```

### 2.2 新建功能分支

```bash
git checkout -b feature/xxx
```

分支命名建议：
- `feature/xxx`：新功能
- `fix/xxx`：缺陷修复
- `docs/xxx`：文档改动

### 2.3 小步提交

每次提交只解决一个明确目的，例如：
- 只改一个页面布局
- 只修一个训练配置问题
- 只补一份文档

```bash
git add .
git commit -m "feat: xxx"
```

### 2.4 合并前同步最新 `dev`

在准备合并之前，先把 `dev` 的最新改动合到自己的功能分支，尽量提前解决冲突。

```bash
git checkout dev
git pull
git checkout feature/xxx
git merge dev
```

### 2.5 做最小自测

至少做一轮轻量检查，确认这次改动没有明显问题。

示例：

```bash
python -m py_compile src/archery_plus/ui/pages/inference_page.py
python run.py
```

说明：
- 如果改的是某个具体文件，至少对该文件做一次 `py_compile`。
- 如果改的是界面或流程，至少启动一次程序并手工点一下相关页面。
- 如果改的是训练、推理、导出逻辑，至少跑一次对应入口，确认不会立即报错。

### 2.6 推送分支并发起 PR

```bash
git push -u origin feature/xxx
```

然后在 GitHub 上发起 Pull Request，目标分支为 `dev`。

### 2.7 `dev` 稳定后再合到 `main`

当 `dev` 里的功能已经通过基本验证，再合并到 `main`。

```bash
git checkout main
git pull
git merge dev
git push
```

## 3. 提交信息规范

统一使用下面这些前缀：

- `feat:` 新功能
- `fix:` 修复问题
- `docs:` 文档更新
- `refactor:` 重构
- `chore:` 杂项调整

示例：

- `feat: add inference session export`
- `fix: correct keypoint label mapping`
- `docs: update training workflow in readme`
- `refactor: simplify inference page state handling`
- `chore: ignore model artifacts in git`

提交信息建议：
- 用英文前缀，正文可以是英文短句。
- 尽量具体，不要只写 `update`、`change`、`fix bug`。

## 4. Pull Request 规范

每个 PR 尽量只做一件事，不要把多个无关修改混在一起。

PR 描述建议至少说明：
- 本次改了什么
- 为什么改
- 你自己做了哪些验证
- 是否影响训练、推理、导出、标注等现有流程

建议：
- 合并前至少让另一位成员看一遍
- UI 改动尽量附截图
- 训练或推理逻辑改动尽量附关键日志

不建议直接合并的情况：
- 本地还没跑过
- 明显包含无关文件
- 改动目的说不清楚

## 5. 本地最小自测

当前项目还没有完整测试体系，因此先采用“最小但可靠”的自测规则。

至少执行其中与本次改动相关的检查：

- 语法检查

```bash
python -m py_compile run.py
python -m py_compile src/archery_plus/ui/pages/inference_page.py
python -m py_compile src/archery_plus/ui/pages/train_page.py
```

- 启动检查

```bash
python run.py
```

- 功能检查
- 改标注页：打开“标注图片”页面，至少完成一次新增或删除操作。
- 改训练页：至少完成一次“生成配置”或“填充训练命令”。
- 改应用页：至少加载一个视频或打开相关页面。
- 改导出页：至少点击一次导出入口，确认不会立即报错。

后续如果仓库加入自动化测试或 CI，再把要求逐步升级。

## 6. 项目特殊约定

### 6.1 大文件不进 Git

以下文件默认不提交到 Git：

- `*.onnx`
- `*.pth`
- `*.pt`
- `*.ckpt`

原因：
- GitHub 对大文件有限制
- 模型文件变化频率低，不适合直接进普通 Git 历史

建议做法：
- 模型文件保留在本地或共享存储
- 仓库里只保留配置文件、说明文档和下载方式

### 6.2 本地产物不提交

以下目录或产物默认不提交：

- `workspace/`
- 训练输出
- 推理导出结果
- 临时测试文件

如果确实需要共享结果，优先导出成独立压缩包或放到共享目录，不直接进仓库。

### 6.3 改功能时同步文档

如果改动影响以下内容，记得同步更新文档：

- 用户操作流程：更新 `README.md`
- 项目阶段和里程碑：更新 `实施计划.md`
- 协作方式变化：更新 `CONTRIBUTING.md`

## 7. 推荐分工

两人协作时，推荐按下面方式分工：

- A 负责：核心算法、训练、导出
- B 负责：UI、文档、验收、测试

这不是硬性限制。遇到复杂任务时，可以交叉协作，但最好在开始前先说清楚谁主改、谁复核。

## 8. 常用 Git 命令速查

### 新功能开发

```bash
git checkout dev
git pull
git checkout -b feature/xxx
```

### 提交代码

```bash
git add .
git commit -m "feat: xxx"
```

### 合并前同步

```bash
git checkout dev
git pull
git checkout feature/xxx
git merge dev
```

### 推送分支

```bash
git push -u origin feature/xxx
```

### 合并到主分支

```bash
git checkout main
git pull
git merge dev
git push
```

## 9. 当前阶段的执行建议

对现在这个项目，最重要的不是一次性把流程做得很重，而是先保证下面三点：

1. 每次改动都走功能分支。
2. 每次合并前都做最小自测。
3. 模型文件、训练产物、临时数据不要进入 Git。

只要这三条稳定执行，双人协作就会顺很多，后面再叠加 CI 和测试也会自然很多。

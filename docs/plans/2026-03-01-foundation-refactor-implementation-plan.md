# Foundation Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 统一虚拟环境约定、重命名 Python 包、清理样例数据并建立首批自动化测试。

**Architecture:** 在独立 worktree 分支中完成本轮工程化重构。首先修复命名与路径问题，再清理 Git 中的样例数据，最后建立 `tests/` 骨架和第一批纯逻辑测试，确保后续 CI 能直接接入。

**Tech Stack:** Python, PySide6, pytest, Git, PowerShell

---

### Task 1: 包名与入口重命名

**Files:**
- Modify: `run.py`
- Move: `src/shejianplus` -> `src/shejianplus`
- Modify: `src/shejianplus/**/*.py`
- Modify: `README.md`
- Modify: `实施计划.md`

**Step 1: 重命名包目录**

将 `src/shejianplus` 重命名为 `src/shejianplus`。

**Step 2: 批量替换导入**

将源码中的 `from shejianplus...` 和 `import shejianplus...` 替换为 `shejianplus`。

**Step 3: 修正启动入口**

将 `run.py` 与 `__main__.py` 的入口导入改为 `shejianplus.main`。

**Step 4: 运行语法检查**

Run: `python -m py_compile run.py`
Expected: PASS

**Step 5: Commit**

```bash
git add run.py src README.md 实施计划.md
git commit -m "refactor: rename python package to shejianplus"
```

### Task 2: 统一虚拟环境约定

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `实施计划.md`

**Step 1: 将文档中的虚拟环境目录统一为 `.venv/`**

**Step 2: 更新忽略规则**

确保 `.venv/` 被忽略，旧 `shejianplus/` 保留兼容说明。

**Step 3: 验证文档一致性**

Run: `rg -n "uv venv|\.venv|shejianplus\\Scripts|\.venv\\Scripts" README.md CONTRIBUTING.md 实施计划.md`
Expected: 与新约定一致

**Step 4: Commit**

```bash
git add .gitignore README.md CONTRIBUTING.md 实施计划.md
git commit -m "docs: standardize venv path to .venv"
```

### Task 3: 清理样例数据与视频

**Files:**
- Modify: `.gitignore`
- Delete from git tracking: `workspace/demo_project/**`
- Delete from git tracking: `56val12times-2.avi`

**Step 1: 保留本地文件，仅从 Git 跟踪中移除**

Run: `git rm -r --cached workspace/demo_project`
Run: `git rm --cached 56val12times-2.avi`

**Step 2: 验证跟踪清理结果**

Run: `git ls-files workspace`
Expected: no output

Run: `git ls-files "*.avi"`
Expected: no output

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: remove sample data from git tracking"
```

### Task 4: 建立测试骨架

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_keypoint_schema.py`
- Create: `tests/test_ema_filter.py`
- Create: `tests/test_scale_converter.py`
- Create: `tests/test_inference_session_exporter.py`
- Modify: `requirements.txt`

**Step 1: 添加测试依赖**

在 `requirements.txt` 中加入 `pytest`。

**Step 2: 编写第一批测试**

覆盖 JSON 校验、滤波更新、比例冻结、导出字段完整性。

**Step 3: 运行测试**

Run: `pytest -q`
Expected: all pass

**Step 4: Commit**

```bash
git add requirements.txt tests
git commit -m "test: add core logic regression tests"
```

### Task 5: 最终验证

**Files:**
- Check: `run.py`
- Check: `src/shejianplus/**`
- Check: `tests/**`

**Step 1: 运行语法检查**

Run: `python -m py_compile run.py`
Expected: PASS

**Step 2: 运行测试**

Run: `pytest -q`
Expected: PASS

**Step 3: 检查 Git 状态**

Run: `git status -sb`
Expected: clean

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize engineering foundation refactor"
```
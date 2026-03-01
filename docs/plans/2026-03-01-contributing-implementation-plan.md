# CONTRIBUTING.md Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为仓库补充一份中文协作文档，明确双人 GitHub 协作流程、提交规范、最小自测和项目特殊约束。

**Architecture:** 采用单文件文档方案，主文档为仓库根目录下的 `CONTRIBUTING.md`，同时保留一份设计记录与一份实施计划，便于后续继续扩展 CI、测试和评审规则。文档内容与当前仓库现状对齐，不引入尚未落地的强制流程。

**Tech Stack:** Markdown, GitHub, Git

---

### Task 1: 建立文档目录与设计记录

**Files:**
- Create: `docs/plans/2026-03-01-contributing-design.md`
- Create: `docs/plans/2026-03-01-contributing-implementation-plan.md`

**Step 1: 创建设计文档**

写入协作文档目标、方案比较、采用方案与文档结构。

**Step 2: 创建实施计划**

写入本计划，明确文件位置与执行范围。

**Step 3: 自查文件结构**

Run: `Get-ChildItem docs/plans`
Expected: 能看到新增的两个 Markdown 文件

**Step 4: Commit**

```bash
git add docs/plans
git commit -m "docs: add contributing design docs"
```

### Task 2: 编写 CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`
- Check: `README.md`
- Check: `.gitignore`

**Step 1: 编写文档主体**

在 `CONTRIBUTING.md` 中写入：
- 分支职责：`main` / `dev` / `feature/*`
- 日常协作流程
- 提交信息规范
- Pull Request 规范
- 本地最小自测
- 项目特殊约定
- 推荐分工
- Git 命令速查

**Step 2: 与当前仓库状态对齐**

确保文档与以下事实一致：
- 模型权重与 `onnx/pth` 文件不进入 Git
- `workspace/` 和训练产物不提交
- 当前最小自测以语法检查和启动验证为主

**Step 3: 人工检查可读性**

Run: `Get-Content CONTRIBUTING.md`
Expected: 标题清晰、步骤完整、命令可直接复制

**Step 4: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add contributing guide"
```

### Task 3: 轻量验证

**Files:**
- Check: `CONTRIBUTING.md`

**Step 1: 查看 git 状态**

Run: `git status -sb`
Expected: 只出现本次新增文档文件

**Step 2: 确认文档编码正常**

Run: `Get-Content CONTRIBUTING.md`
Expected: 中文显示正常，无乱码

**Step 3: Commit**

```bash
git add CONTRIBUTING.md docs/plans
git commit -m "docs: finalize contributing workflow docs"
```

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shejianplus.config import DEFAULT_PROJECT_DIR
from shejianplus.core.exporter import ExportOptions, export_project_annotations
from shejianplus.core.inference_session_cache import get_inference_session_snapshot
from shejianplus.core.inference_session_exporter import export_inference_session_timeseries


TARGET_ARCHERY = "archery"
TARGET_HUMAN = "human"
TARGET_BOTH = "both"


class ExportPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_dir = Path(DEFAULT_PROJECT_DIR)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        desc = QLabel(
            "导出模块 v0.2：统一导出标注数据与应用推理时序数据。"
            "推理时序导出基于“应用模块最近一次推理”的缓存结果。"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        project_box = QGroupBox("项目与输出")
        project_form = QFormLayout(project_box)

        project_row = QHBoxLayout()
        self.project_path_edit = QLineEdit(str(self._project_dir))
        self.project_path_edit.setPlaceholderText("项目目录（包含 annotations/ 与 images/）")
        project_row.addWidget(self.project_path_edit, stretch=1)
        self.browse_project_btn = QPushButton("浏览")
        self.browse_project_btn.clicked.connect(self._browse_project)
        project_row.addWidget(self.browse_project_btn)
        project_form.addRow("项目目录", project_row)

        output_hint = QLabel("输出目录：<project>/output/exports/<时间戳>/")
        output_hint.setWordWrap(True)
        project_form.addRow("输出位置", output_hint)
        layout.addWidget(project_box)

        ann_box = QGroupBox("标注数据导出")
        ann_layout = QVBoxLayout(ann_box)
        self.raw_2d = QCheckBox("raw_2d.csv（关键点长表）")
        self.raw_2d.setChecked(True)
        self.filtered_2d = QCheckBox("filtered_2d.csv（当前为占位导出）")
        self.filtered_2d.setChecked(True)
        self.bbox_2d = QCheckBox("bbox_2d.csv（矩形框）")
        self.bbox_2d.setChecked(True)
        self.annotations_json = QCheckBox("annotations_export.json（统一导出包）")
        self.annotations_json.setChecked(True)
        self.label_schema_json = QCheckBox("label_schema_export.json（标签结构）")
        self.label_schema_json.setChecked(True)
        ann_layout.addWidget(self.raw_2d)
        ann_layout.addWidget(self.filtered_2d)
        ann_layout.addWidget(self.bbox_2d)
        ann_layout.addWidget(self.annotations_json)
        ann_layout.addWidget(self.label_schema_json)
        layout.addWidget(ann_box)

        timeseries_box = QGroupBox("推理时序导出（应用模块）")
        timeseries_form = QFormLayout(timeseries_box)

        self.session_status = QLabel("缓存状态：未检测到推理会话")
        self.session_status.setWordWrap(True)
        timeseries_form.addRow("缓存状态", self.session_status)

        self.session_filter = QComboBox()
        self.session_filter.addItem("仅弓箭", TARGET_ARCHERY)
        self.session_filter.addItem("仅人体", TARGET_HUMAN)
        self.session_filter.addItem("双模型全部", TARGET_BOTH)
        timeseries_form.addRow("导出范围", self.session_filter)

        ts_action_row = QHBoxLayout()
        self.export_session_btn = QPushButton("导出推理时序CSV")
        self.export_session_btn.clicked.connect(self._export_inference_session)
        self.refresh_session_btn = QPushButton("刷新缓存")
        self.refresh_session_btn.clicked.connect(self._refresh_session_status)
        ts_action_row.addWidget(self.export_session_btn)
        ts_action_row.addWidget(self.refresh_session_btn)
        ts_action_row.addStretch(1)
        ts_action_wrap = QWidget()
        ts_action_wrap.setLayout(ts_action_row)
        timeseries_form.addRow("操作", ts_action_wrap)
        layout.addWidget(timeseries_box)

        action_row = QHBoxLayout()
        self.start_export = QPushButton("开始导出标注")
        self.start_export.clicked.connect(self._start_export)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(self._clear_log)
        action_row.addWidget(self.start_export)
        action_row.addWidget(self.clear_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("导出日志")
        layout.addWidget(self.log, stretch=1)

        self._refresh_session_status()

    def _browse_project(self) -> None:
        current = self.project_path_edit.text().strip() or str(self._project_dir)
        selected = QFileDialog.getExistingDirectory(self, "选择项目目录", current)
        if selected:
            self.project_path_edit.setText(selected)
            self._project_dir = Path(selected)

    def _start_export(self) -> None:
        project_dir = Path(self.project_path_edit.text().strip() or str(self._project_dir))
        ann_path = project_dir / "annotations" / "manual_annotations.json"
        if not ann_path.exists():
            QMessageBox.warning(self, "提示", f"未找到标注文件：{ann_path}")
            self._append_log(f"[ERROR] 未找到标注文件: {ann_path}")
            return

        options = ExportOptions(
            export_raw_2d_csv=bool(self.raw_2d.isChecked()),
            export_filtered_2d_csv=bool(self.filtered_2d.isChecked()),
            export_bbox_csv=bool(self.bbox_2d.isChecked()),
            export_annotations_json=bool(self.annotations_json.isChecked()),
            export_label_schema_json=bool(self.label_schema_json.isChecked()),
        )
        if not any(
            [
                options.export_raw_2d_csv,
                options.export_filtered_2d_csv,
                options.export_bbox_csv,
                options.export_annotations_json,
                options.export_label_schema_json,
            ]
        ):
            QMessageBox.information(self, "提示", "请至少勾选一个导出项。")
            return

        self._append_log(f"[INFO] 开始导出标注，项目目录: {project_dir}")
        try:
            result = export_project_annotations(project_dir, options)
        except Exception as exc:
            self._append_log(f"[ERROR] 导出失败: {exc}")
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        self._append_log(
            "[INFO] 标注导出完成: "
            f"{result.output_dir} | images={result.image_count}, "
            f"keypoint_rows={result.keypoint_rows}, bbox_rows={result.bbox_rows}"
        )
        for path in result.files:
            self._append_log(f"[FILE] {path}")
        QMessageBox.information(self, "导出完成", f"导出成功，输出目录：\n{result.output_dir}")

    def _export_inference_session(self) -> None:
        snapshot = get_inference_session_snapshot()
        if snapshot is None:
            QMessageBox.information(self, "提示", "未检测到推理缓存，请先在应用模块完成一次推理。")
            return

        scope = self.session_filter.currentData()
        raw_rows = self._filter_timeseries_rows(snapshot.raw_rows, scope)
        filtered_rows = self._filter_timeseries_rows(snapshot.filtered_rows, scope)
        if not raw_rows and not filtered_rows:
            QMessageBox.information(self, "提示", "当前筛选范围没有可导出的时序数据。")
            return

        project_dir = Path(self.project_path_edit.text().strip() or str(self._project_dir))
        export_root = project_dir / "output" / "exports"
        export_root.mkdir(parents=True, exist_ok=True)

        try:
            result = export_inference_session_timeseries(
                export_root=export_root,
                session_meta=dict(snapshot.meta, export_scope=str(scope)),
                raw_rows=raw_rows,
                filtered_rows=filtered_rows,
            )
        except Exception as exc:
            self._append_log(f"[ERROR] 推理时序导出失败: {exc}")
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        self._append_log(
            "[INFO] 推理时序导出完成: "
            f"{result.output_dir} | raw_rows={result.raw_rows}, filtered_rows={result.filtered_rows}"
        )
        self._append_log(f"[FILE] {result.raw_csv}")
        self._append_log(f"[FILE] {result.filtered_csv}")
        self._append_log(f"[FILE] {result.meta_json}")
        QMessageBox.information(self, "导出完成", f"推理时序导出成功：\n{result.output_dir}")

    def _filter_timeseries_rows(self, rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
        scope_text = str(scope)
        if scope_text == TARGET_BOTH:
            return [dict(r) for r in rows if isinstance(r, dict)]
        return [dict(r) for r in rows if isinstance(r, dict) and str(r.get("target", "")) == scope_text]

    def _refresh_session_status(self) -> None:
        snapshot = get_inference_session_snapshot()
        if snapshot is None:
            self.session_status.setText("缓存状态：未检测到推理会话")
            return
        raw_count = len(snapshot.raw_rows)
        filtered_count = len(snapshot.filtered_rows)
        updated = snapshot.updated_at
        self.session_status.setText(
            f"缓存状态：已缓存推理会话 | raw_rows={raw_count} | filtered_rows={filtered_count} | 更新于 {updated}"
        )

    def _append_log(self, text: str) -> None:
        self.log.append(str(text))

    def _clear_log(self) -> None:
        self.log.clear()

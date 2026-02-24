from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
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

from archery_plus.config import DEFAULT_PROJECT_DIR
from archery_plus.core.exporter import ExportOptions, export_project_annotations


class ExportPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._project_dir = Path(DEFAULT_PROJECT_DIR)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        desc = QLabel(
            "导出模块 v0.1：导出项目标注到统一 JSON / 2D 关键点 CSV / bbox CSV。"
            "当前 filtered_2d.csv 为占位导出（与 raw_2d 同数据，标记 filter_method）。"
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

        output_hint = QLabel("导出目录将自动创建在：<项目>/output/exports/<时间戳>/")
        output_hint.setWordWrap(True)
        project_form.addRow("输出位置", output_hint)
        layout.addWidget(project_box)

        options_box = QGroupBox("导出选项")
        options_layout = QVBoxLayout(options_box)
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
        options_layout.addWidget(self.raw_2d)
        options_layout.addWidget(self.filtered_2d)
        options_layout.addWidget(self.bbox_2d)
        options_layout.addWidget(self.annotations_json)
        options_layout.addWidget(self.label_schema_json)
        layout.addWidget(options_box)

        action_row = QHBoxLayout()
        self.start_export = QPushButton("开始导出")
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

        self._append_log(f"[INFO] 开始导出，项目目录: {project_dir}")
        try:
            result = export_project_annotations(project_dir, options)
        except Exception as exc:
            self._append_log(f"[ERROR] 导出失败: {exc}")
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        self._append_log(
            "[INFO] 导出完成: "
            f"{result.output_dir} | images={result.image_count}, "
            f"keypoint_rows={result.keypoint_rows}, bbox_rows={result.bbox_rows}"
        )
        for path in result.files:
            self._append_log(f"[FILE] {path}")
        QMessageBox.information(self, "导出完成", f"导出成功，输出目录：\n{result.output_dir}")

    def _append_log(self, text: str) -> None:
        self.log.append(str(text))

    def _clear_log(self) -> None:
        self.log.clear()

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from archery_plus.config import DEFAULT_PROJECT_DIR
from archery_plus.core.mmpose_training import MMPoseDatasetBuildResult, build_mmpose_coco_dataset
from archery_plus.core.training_config import (
    GeneratedTrainingConfig,
    INIT_MODE_OPTIONS,
    OPTIMIZER_OPTIONS,
    SCHEDULER_OPTIONS,
    TRAIN_TARGET_OPTIONS,
    TRAIN_TASK_OPTIONS,
    TrainingPreset,
    default_training_preset,
    generate_training_config,
    list_model_names,
    load_training_preset,
    resolve_model_assets,
    save_training_preset,
)


class TrainPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._process: QProcess | None = None
        self._dataset_result: MMPoseDatasetBuildResult | None = None
        self._generated_config: GeneratedTrainingConfig | None = None
        self._build_ui()

        preset = default_training_preset(DEFAULT_PROJECT_DIR)
        self._apply_preset_to_ui(preset)
        self._on_init_mode_changed(self.init_mode_combo.currentText())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "训练模块 v0.2：支持 MMPose 数据构建、配置生成、参数预设与命令启动。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        config_box = QGroupBox("训练配置")
        form = QFormLayout(config_box)

        project_row = QHBoxLayout()
        self.project_path_edit = QLineEdit(str(DEFAULT_PROJECT_DIR))
        self.project_path_edit.setPlaceholderText("项目目录")
        project_row.addWidget(self.project_path_edit, stretch=1)

        choose_project_btn = QPushButton("浏览")
        choose_project_btn.setFixedWidth(80)
        choose_project_btn.clicked.connect(self._select_project_dir)
        project_row.addWidget(choose_project_btn)

        task_row = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(TRAIN_TASK_OPTIONS)
        task_row.addWidget(self.task_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list_model_names())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        task_row.addWidget(self.model_combo)

        self.target_combo = QComboBox()
        self.target_combo.addItems(TRAIN_TARGET_OPTIONS)
        task_row.addWidget(self.target_combo)

        init_row = QHBoxLayout()
        self.init_mode_combo = QComboBox()
        self.init_mode_combo.addItems(INIT_MODE_OPTIONS)
        self.init_mode_combo.currentTextChanged.connect(self._on_init_mode_changed)
        init_row.addWidget(self.init_mode_combo)

        self.pretrained_weight_edit = QLineEdit()
        self.pretrained_weight_edit.setPlaceholderText("预训练权重路径（.pth/.pt）")
        init_row.addWidget(self.pretrained_weight_edit, stretch=1)

        self.choose_weight_btn = QPushButton("权重")
        self.choose_weight_btn.setFixedWidth(80)
        self.choose_weight_btn.clicked.connect(self._select_weight_file)
        init_row.addWidget(self.choose_weight_btn)

        split_row = QHBoxLayout()
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setPrefix("train=")
        split_row.addWidget(self.train_ratio_spin)

        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 1.0)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setValue(0.1)
        self.val_ratio_spin.setPrefix("val=")
        split_row.addWidget(self.val_ratio_spin)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.1)
        self.test_ratio_spin.setPrefix("test=")
        split_row.addWidget(self.test_ratio_spin)

        self.build_dataset_btn = QPushButton("生成MMPose数据")
        self.build_dataset_btn.clicked.connect(self._build_dataset_clicked)
        split_row.addWidget(self.build_dataset_btn)

        dataset_row = QHBoxLayout()
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setPlaceholderText("数据集输出目录")
        dataset_row.addWidget(self.dataset_dir_edit, stretch=1)

        choose_dataset_btn = QPushButton("目录")
        choose_dataset_btn.setFixedWidth(80)
        choose_dataset_btn.clicked.connect(self._select_dataset_dir)
        dataset_row.addWidget(choose_dataset_btn)

        category_row = QHBoxLayout()
        self.human_category_edit = QLineEdit("human")
        self.human_category_edit.setPlaceholderText("人体类别名称")
        category_row.addWidget(self.human_category_edit)
        self.archery_category_edit = QLineEdit("archery")
        self.archery_category_edit.setPlaceholderText("弓箭类别名称")
        category_row.addWidget(self.archery_category_edit)

        hp_row_1 = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(300)
        self.epochs_spin.setPrefix("epoch=")
        hp_row_1.addWidget(self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.batch_spin.setPrefix("batch=")
        hp_row_1.addWidget(self.batch_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(4)
        self.workers_spin.setPrefix("workers=")
        hp_row_1.addWidget(self.workers_spin)

        self.gpus_spin = QSpinBox()
        self.gpus_spin.setRange(0, 8)
        self.gpus_spin.setValue(1)
        self.gpus_spin.setPrefix("gpus=")
        hp_row_1.addWidget(self.gpus_spin)

        hp_row_2 = QHBoxLayout()
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(64, 2048)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(640)
        self.image_size_spin.setPrefix("img=")
        hp_row_2.addWidget(self.image_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.004)
        self.lr_spin.setPrefix("lr=")
        hp_row_2.addWidget(self.lr_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(OPTIMIZER_OPTIONS)
        hp_row_2.addWidget(self.optimizer_combo)

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(SCHEDULER_OPTIONS)
        hp_row_2.addWidget(self.scheduler_combo)

        config_template_row = QHBoxLayout()
        self.base_config_edit = QLineEdit()
        self.base_config_edit.setPlaceholderText("基础配置文件路径（.py）")
        config_template_row.addWidget(self.base_config_edit, stretch=1)

        choose_cfg_btn = QPushButton("配置")
        choose_cfg_btn.setFixedWidth(80)
        choose_cfg_btn.clicked.connect(self._select_base_config_file)
        config_template_row.addWidget(choose_cfg_btn)

        generated_cfg_row = QHBoxLayout()
        self.generated_config_edit = QLineEdit()
        self.generated_config_edit.setReadOnly(True)
        self.generated_config_edit.setPlaceholderText("生成后的训练配置文件")
        generated_cfg_row.addWidget(self.generated_config_edit, stretch=1)

        self.generate_cfg_btn = QPushButton("生成训练配置")
        self.generate_cfg_btn.clicked.connect(self._generate_training_config_clicked)
        generated_cfg_row.addWidget(self.generate_cfg_btn)

        preset_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("保存预设")
        self.load_preset_btn = QPushButton("加载预设")
        self.save_preset_btn.clicked.connect(self._save_preset)
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_row.addWidget(self.save_preset_btn)
        preset_row.addWidget(self.load_preset_btn)
        preset_row.addStretch(1)

        command_row = QHBoxLayout()
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("训练命令（可自动填充）")
        command_row.addWidget(self.command_edit, stretch=1)

        self.fill_command_btn = QPushButton("填充训练命令")
        self.fill_command_btn.clicked.connect(self._fill_training_command)
        command_row.addWidget(self.fill_command_btn)

        form.addRow("项目目录", project_row)
        form.addRow("任务/模型/目标", task_row)
        form.addRow("初始化", init_row)
        form.addRow("数据划分", split_row)
        form.addRow("数据目录", dataset_row)
        form.addRow("类别名称", category_row)
        form.addRow("训练参数(1)", hp_row_1)
        form.addRow("训练参数(2)", hp_row_2)
        form.addRow("基础配置", config_template_row)
        form.addRow("生成配置", generated_cfg_row)
        form.addRow("参数预设", preset_row)
        form.addRow("训练命令", command_row)
        layout.addWidget(config_box)

        control_row = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.stop_btn = QPushButton("停止训练")
        self.clear_btn = QPushButton("清空日志")

        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.clear_btn.clicked.connect(self.log.clear)
        self.stop_btn.setEnabled(False)

        control_row.addWidget(self.start_btn)
        control_row.addWidget(self.stop_btn)
        control_row.addWidget(self.clear_btn)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志输出区域")
        layout.addWidget(self.log)

    def _select_project_dir(self) -> None:
        current = self.project_path_edit.text().strip() or str(DEFAULT_PROJECT_DIR)
        selected = QFileDialog.getExistingDirectory(self, "选择项目目录", current)
        if selected:
            self.project_path_edit.setText(selected)
            self.dataset_dir_edit.setText(str(Path(selected) / "output" / "mmpose_dataset_v1"))
            self._dataset_result = None
            self._generated_config = None
            self.generated_config_edit.clear()

    def _select_weight_file(self) -> None:
        current = self.pretrained_weight_edit.text().strip() or str(Path(self.project_path_edit.text().strip()))
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择预训练权重",
            current,
            "权重文件 (*.pth *.pt *.ckpt);;所有文件 (*)",
        )
        if selected:
            self.pretrained_weight_edit.setText(selected)

    def _select_base_config_file(self) -> None:
        current = self.base_config_edit.text().strip() or str(Path(self.project_path_edit.text().strip()))
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择基础配置文件",
            current,
            "Python配置 (*.py);;所有文件 (*)",
        )
        if selected:
            self.base_config_edit.setText(selected)

    def _select_dataset_dir(self) -> None:
        current = self.dataset_dir_edit.text().strip() or str(Path(self.project_path_edit.text().strip()) / "output")
        selected = QFileDialog.getExistingDirectory(self, "选择数据目录", current)
        if selected:
            self.dataset_dir_edit.setText(selected)

    def _build_dataset_clicked(self) -> None:
        self._build_dataset(show_message=True)

    def _build_dataset(self, show_message: bool) -> MMPoseDatasetBuildResult | None:
        preset = self._collect_preset_from_ui()
        project_root = preset.project_dir

        if not project_root.exists():
            QMessageBox.warning(self, "提示", f"项目目录不存在: {project_root}")
            return None

        ratio_sum = preset.train_ratio + preset.val_ratio + preset.test_ratio
        if ratio_sum <= 0:
            QMessageBox.warning(self, "提示", "train/val/test 比例之和必须大于 0。")
            return None

        try:
            result = build_mmpose_coco_dataset(
                project_root=project_root,
                train_ratio=preset.train_ratio,
                val_ratio=preset.val_ratio,
                test_ratio=preset.test_ratio,
                random_seed=preset.random_seed,
                human_category_name=preset.human_category_name,
                archery_category_name=preset.archery_category_name,
                output_dir=preset.dataset_dir,
            )
        except Exception as exc:
            self._append_log(f"[ERROR] MMPose数据构建失败: {exc}")
            QMessageBox.critical(self, "构建失败", str(exc))
            return None

        self._dataset_result = result
        self._generated_config = None
        self.generated_config_edit.clear()
        self.dataset_dir_edit.setText(str(result.output_dir))

        self._append_log(
            (
                f"[INFO] MMPose数据集构建完成: {result.output_dir} | "
                f"images(total/train/val/test)={result.total_images}/{result.train_images}/{result.val_images}/{result.test_images} | "
                f"human_ann(train/val/test)={result.human_ann_counts['train']}/{result.human_ann_counts['val']}/{result.human_ann_counts['test']} | "
                f"archery_ann(train/val/test)={result.archery_ann_counts['train']}/{result.archery_ann_counts['val']}/{result.archery_ann_counts['test']}"
            )
        )

        if show_message:
            QMessageBox.information(
                self,
                "构建完成",
                (
                    f"输出目录: {result.output_dir}\n"
                    f"总图片: {result.total_images}\n"
                    f"train/val/test: {result.train_images}/{result.val_images}/{result.test_images}\n"
                    f"人体标注(train/val/test): {result.human_ann_counts['train']}/{result.human_ann_counts['val']}/{result.human_ann_counts['test']}\n"
                    f"弓箭标注(train/val/test): {result.archery_ann_counts['train']}/{result.archery_ann_counts['val']}/{result.archery_ann_counts['test']}"
                ),
            )

        return result

    def _generate_training_config_clicked(self) -> None:
        self._generate_config(show_message=True)

    def _generate_config(self, show_message: bool) -> GeneratedTrainingConfig | None:
        preset = self._collect_preset_from_ui()
        if not preset.project_dir.exists():
            QMessageBox.warning(self, "提示", f"项目目录不存在: {preset.project_dir}")
            return None

        dataset_result = self._dataset_result
        if dataset_result is None:
            dataset_result = self._build_dataset(show_message=False)
            if dataset_result is None:
                return None

        try:
            generated = generate_training_config(preset=preset, dataset_result=dataset_result)
        except Exception as exc:
            self._append_log(f"[ERROR] 训练配置生成失败: {exc}")
            QMessageBox.critical(self, "生成失败", str(exc))
            return None

        self._generated_config = generated
        self.generated_config_edit.setText(str(generated.config_path))
        self._append_log(
            f"[INFO] 训练配置已生成: {generated.config_path} | target={generated.target_name} | num_keypoints={generated.num_keypoints}"
        )

        if show_message:
            QMessageBox.information(
                self,
                "配置已生成",
                f"配置文件: {generated.config_path}\nwork_dir: {generated.work_dir}",
            )

        return generated

    def _fill_training_command(self) -> None:
        generated = self._generated_config
        if generated is None:
            generated = self._generate_config(show_message=False)
            if generated is None:
                return

        self.command_edit.setText(generated.command)
        self._append_log(f"[INFO] 已填充训练命令: {generated.command}")

    def _save_preset(self) -> None:
        preset = self._collect_preset_from_ui()
        default_path = preset.project_dir / "output" / "train_presets" / "train_preset.json"

        selected, _ = QFileDialog.getSaveFileName(
            self,
            "保存训练预设",
            str(default_path),
            "JSON文件 (*.json)",
        )
        if not selected:
            return

        try:
            save_training_preset(Path(selected), preset)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        self._append_log(f"[INFO] 训练预设已保存: {selected}")

    def _load_preset(self) -> None:
        project_root = Path(self.project_path_edit.text().strip() or str(DEFAULT_PROJECT_DIR))
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "加载训练预设",
            str(project_root),
            "JSON文件 (*.json)",
        )
        if not selected:
            return

        try:
            preset = load_training_preset(Path(selected), fallback_project_root=project_root)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self._apply_preset_to_ui(preset)
        self._dataset_result = None
        self._generated_config = None
        self.generated_config_edit.clear()
        self._append_log(f"[INFO] 训练预设已加载: {selected}")

    def _collect_preset_from_ui(self) -> TrainingPreset:
        project_dir_text = self.project_path_edit.text().strip() or str(DEFAULT_PROJECT_DIR)
        project_dir = Path(project_dir_text)

        dataset_dir_text = self.dataset_dir_edit.text().strip()
        dataset_dir = Path(dataset_dir_text) if dataset_dir_text else project_dir / "output" / "mmpose_dataset_v1"

        return TrainingPreset(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            task_type=self.task_combo.currentText().strip() or TRAIN_TASK_OPTIONS[0],
            model_name=self.model_combo.currentText().strip() or list_model_names()[0],
            target_name=self.target_combo.currentText().strip() or TRAIN_TARGET_OPTIONS[0],
            init_mode=self.init_mode_combo.currentText().strip() or INIT_MODE_OPTIONS[0],
            pretrained_weight_path=self.pretrained_weight_edit.text().strip(),
            base_config_path=self.base_config_edit.text().strip(),
            human_category_name=self.human_category_edit.text().strip() or "human",
            archery_category_name=self.archery_category_edit.text().strip() or "archery",
            epochs=int(self.epochs_spin.value()),
            batch_size=int(self.batch_spin.value()),
            num_workers=int(self.workers_spin.value()),
            image_size=int(self.image_size_spin.value()),
            learning_rate=float(self.lr_spin.value()),
            optimizer=self.optimizer_combo.currentText().strip() or OPTIMIZER_OPTIONS[0],
            scheduler=self.scheduler_combo.currentText().strip() or SCHEDULER_OPTIONS[0],
            train_ratio=float(self.train_ratio_spin.value()),
            val_ratio=float(self.val_ratio_spin.value()),
            test_ratio=float(self.test_ratio_spin.value()),
            random_seed=42,
            gpus=int(self.gpus_spin.value()),
        )

    def _apply_preset_to_ui(self, preset: TrainingPreset) -> None:
        self.project_path_edit.setText(str(preset.project_dir))
        self.dataset_dir_edit.setText(str(preset.dataset_dir))

        self._set_combo(self.task_combo, preset.task_type)
        self._set_combo(self.model_combo, preset.model_name)
        self._set_combo(self.target_combo, preset.target_name)
        self._set_combo(self.init_mode_combo, preset.init_mode)
        self._set_combo(self.optimizer_combo, preset.optimizer)
        self._set_combo(self.scheduler_combo, preset.scheduler)

        self.pretrained_weight_edit.setText(str(preset.pretrained_weight_path))
        self.base_config_edit.setText(str(preset.base_config_path))

        self.human_category_edit.setText(preset.human_category_name)
        self.archery_category_edit.setText(preset.archery_category_name)

        self.epochs_spin.setValue(max(1, int(preset.epochs)))
        self.batch_spin.setValue(max(1, int(preset.batch_size)))
        self.workers_spin.setValue(max(0, int(preset.num_workers)))
        self.image_size_spin.setValue(max(64, int(preset.image_size)))
        self.lr_spin.setValue(max(0.000001, float(preset.learning_rate)))
        self.gpus_spin.setValue(max(0, int(preset.gpus)))

        self.train_ratio_spin.setValue(max(0.0, float(preset.train_ratio)))
        self.val_ratio_spin.setValue(max(0.0, float(preset.val_ratio)))
        self.test_ratio_spin.setValue(max(0.0, float(preset.test_ratio)))

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_model_changed(self, model_name: str) -> None:
        config_path, weight_path = resolve_model_assets(model_name)
        self.base_config_edit.setText(str(config_path))
        if self.init_mode_combo.currentText() == "加载预训练权重":
            self.pretrained_weight_edit.setText(str(weight_path))

    def _on_init_mode_changed(self, mode_text: str) -> None:
        use_pretrained = mode_text == "加载预训练权重"
        self.pretrained_weight_edit.setEnabled(use_pretrained)
        self.choose_weight_btn.setEnabled(use_pretrained)
        if use_pretrained and not self.pretrained_weight_edit.text().strip():
            _, weight_path = resolve_model_assets(self.model_combo.currentText())
            self.pretrained_weight_edit.setText(str(weight_path))

    def _start_training(self) -> None:
        if self._process is not None:
            QMessageBox.information(self, "提示", "训练任务已在运行。")
            return

        project_dir = self.project_path_edit.text().strip()
        if not project_dir:
            QMessageBox.warning(self, "提示", "请先设置项目目录。")
            return

        command = self.command_edit.text().strip()
        if not command:
            QMessageBox.warning(self, "提示", "请先填充或输入训练命令。")
            return

        work_dir = Path(project_dir)
        if not work_dir.exists():
            QMessageBox.warning(self, "提示", f"项目目录不存在: {work_dir}")
            return

        process = QProcess(self)
        process.setWorkingDirectory(str(work_dir))

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")

        python_dir = str(Path(sys.executable).resolve().parent)
        current_path = env.value("PATH")
        if python_dir:
            path_entries = current_path.split(os.pathsep) if current_path else []
            if python_dir not in path_entries:
                merged = python_dir if not current_path else python_dir + os.pathsep + current_path
                env.insert("PATH", merged)

        if os.name == "nt":
            user_profile = env.value("USERPROFILE") or os.environ.get("USERPROFILE", "")
            home_drive = env.value("HOMEDRIVE") or os.environ.get("HOMEDRIVE", "")
            home_path = env.value("HOMEPATH") or os.environ.get("HOMEPATH", "")
            system_drive = env.value("SystemDrive") or os.environ.get("SystemDrive", "")

            if not user_profile and home_drive and home_path:
                user_profile = f"{home_drive}{home_path}"
            if not user_profile:
                user_profile = str(work_dir.resolve())

            if not home_drive:
                drive, _ = os.path.splitdrive(user_profile)
                home_drive = drive
            if not home_path:
                _, path_part = os.path.splitdrive(user_profile)
                home_path = path_part
            if not system_drive:
                drive, _ = os.path.splitdrive(user_profile)
                system_drive = drive

            if user_profile:
                env.insert("USERPROFILE", user_profile)
                env.insert("HOME", user_profile)
            if home_drive:
                env.insert("HOMEDRIVE", home_drive)
            if home_path:
                env.insert("HOMEPATH", home_path)
            if system_drive:
                env.insert("SystemDrive", system_drive)

        process.setProcessEnvironment(env)

        resolved_command = command
        if os.name == "nt":
            stripped = command.lstrip()
            python_exe = str(Path(sys.executable).resolve())
            if stripped.lower().startswith("python "):
                resolved_command = f'"{python_exe}" {stripped[7:]}'
                program = python_exe
                args = shlex.split(stripped[7:], posix=True)
            elif stripped.lower() == "python":
                resolved_command = f'"{python_exe}"'
                program = python_exe
                args = []
            else:
                program = "cmd"
                args = ["/d", "/c", resolved_command]
        else:
            program = "/bin/bash"
            args = ["-lc", command]

        process.readyReadStandardOutput.connect(self._on_stdout)
        process.readyReadStandardError.connect(self._on_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self._append_log(f"[INFO] 工作目录: {work_dir}")
        self._append_log(f"[INFO] 启动命令: {resolved_command if os.name == 'nt' else command}")

        process.start(program, args)
        if not process.waitForStarted(5000):
            self._append_log("[ERROR] 启动失败：无法拉起训练进程。")
            QMessageBox.critical(self, "启动失败", "无法拉起训练进程，请检查命令是否正确。")
            process.deleteLater()
            return

        self._process = process
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._append_log("[INFO] 训练任务已启动。")

    def _stop_training(self) -> None:
        if self._process is None:
            return

        self._append_log("[INFO] 正在停止训练任务...")
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._process.kill()
            self._process.waitForFinished(2000)

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._append_log(text.rstrip("\n"))

    def _on_stderr(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self._append_log(text.rstrip("\n"))

    def _on_finished(self, exit_code: int, _exit_status) -> None:  # type: ignore[override]
        self._append_log(f"[INFO] 训练任务结束，退出码: {exit_code}")
        self._cleanup_process_state()

    def _on_error(self, error) -> None:  # type: ignore[override]
        self._append_log(f"[ERROR] 训练进程异常: {error}")

    def _cleanup_process_state(self) -> None:
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        for line in text.splitlines():
            self.log.append(line)




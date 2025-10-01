import threading
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import subprocess
from subprocess import Popen, PIPE

# 将项目根目录加入 sys.path，确保可导入本地模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from unified_emotion_recognition import UnifiedEmotionRecognizer  # noqa: E402


class EmotionGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("面部情绪识别 - 图形界面")
        self.geometry("980x640")
        self.minsize(900, 580)
        ctk.set_appearance_mode("System")  # 可选: "Light" / "Dark" / "System"
        ctk.set_default_color_theme("blue")

        try:
            # macOS 程序坞与窗口图标（可选）
            icon_path = os.path.join(PROJECT_ROOT, "images", "90.jpg")
            if os.path.exists(icon_path):
                try:
                    self.iconphoto(True, tk.PhotoImage(file=icon_path))
                except Exception:
                    pass
        except Exception:
            pass

        # 识别器实例（按需创建，避免启动时加载模型耗时）
        self.recognizer = None
        self.is_running = False
        self.current_process: Popen | None = None
        self.python_exec = self._resolve_python_exec()

        # 状态变量
        self.model_path_var = tk.StringVar(value=os.path.join(PROJECT_ROOT, "models", "final_emotion_model_optimized.tflite"))
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.use_tflite_var = tk.BooleanVar(value=True)
        self.use_m1_var = tk.BooleanVar(value=False)
        self.show_fps_var = tk.BooleanVar(value=True)
        self.res_w_var = tk.IntVar(value=640)
        self.res_h_var = tk.IntVar(value=480)
        self.skip_frames_var = tk.IntVar(value=0)
        # 语言选择（显示名与代码映射）
        self.language_display_var = tk.StringVar(value="中文")
        self._language_name_to_code = {"中文": "zh", "English": "en", "Русский": "ru"}

        # 训练相关状态
        self.train_dataset_dir_var = tk.StringVar(value="")
        self.train_fer_csv_var = tk.StringVar(value="")
        self.train_epochs_var = tk.IntVar(value=20)
        self.train_batch_var = tk.IntVar(value=32)
        self.train_lr_var = tk.DoubleVar(value=0.0001)
        self.train_augment_var = tk.StringVar(value="medium")
        self.train_class_weight_var = tk.BooleanVar(value=True)
        self.train_exp_name_var = tk.StringVar(value="")
        self.train_save_dir_var = tk.StringVar(value=os.path.join(PROJECT_ROOT, "models"))

        self._build_layout()
        # 启动后将窗口置顶并聚焦
        self.after(100, self._bring_to_front)
        # 显示当前将用于子进程的 Python 解释器
        self.after(150, lambda: self._log(f"子进程将使用 Python: {self.python_exec}"))

    def _build_layout(self):
        # 左侧面板：操作&参数
        sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        sidebar.pack(side="left", fill="y")

        title = ctk.CTkLabel(sidebar, text="面部情绪识别", font=("PingFang SC", 20, "bold"))
        title.pack(padx=20, pady=(20, 10), anchor="w")

        # 模型选择
        model_label = ctk.CTkLabel(sidebar, text="模型路径")
        model_label.pack(padx=20, pady=(10, 5), anchor="w")
        model_row = ctk.CTkFrame(sidebar)
        model_row.pack(fill="x", padx=20)
        model_entry = ctk.CTkEntry(model_row, textvariable=self.model_path_var)
        model_entry.pack(side="left", fill="x", expand=True)
        model_button = ctk.CTkButton(model_row, text="选择", width=70, command=self._choose_model_file)
        model_button.pack(side="left", padx=(8, 0))

        # 置信度
        conf_label = ctk.CTkLabel(sidebar, text="置信度阈值 (0.0 - 1.0)")
        conf_label.pack(padx=20, pady=(14, 0), anchor="w")
        conf_slider = ctk.CTkSlider(sidebar, from_=0.0, to=1.0, number_of_steps=100, variable=self.confidence_var)
        conf_slider.pack(fill="x", padx=20, pady=(4, 4))
        conf_value = ctk.CTkLabel(sidebar, textvariable=ctk.StringVar(value=f"当前: {self.confidence_var.get():.2f}"))
        conf_value.pack(padx=20, anchor="w")

        def _update_conf_label(*_):
            conf_value.configure(text=f"当前: {self.confidence_var.get():.2f}")

        self.confidence_var.trace_add("write", _update_conf_label)

        # 分辨率与帧跳过
        res_label = ctk.CTkLabel(sidebar, text="摄像头分辨率 (宽 x 高)")
        res_label.pack(padx=20, pady=(14, 4), anchor="w")
        res_row = ctk.CTkFrame(sidebar)
        res_row.pack(fill="x", padx=20)
        res_w = ctk.CTkEntry(res_row, width=86, textvariable=self.res_w_var)
        res_w.pack(side="left")
        x_label = ctk.CTkLabel(res_row, text=" × ")
        x_label.pack(side="left", padx=6)
        res_h = ctk.CTkEntry(res_row, width=86, textvariable=self.res_h_var)
        res_h.pack(side="left")

        skip_label = ctk.CTkLabel(sidebar, text="帧跳过数量 (M1优化下有效)")
        skip_label.pack(padx=20, pady=(14, 4), anchor="w")
        skip_spin = ctk.CTkEntry(sidebar, textvariable=self.skip_frames_var)
        skip_spin.pack(fill="x", padx=20)

        # 语言选择
        lang_label = ctk.CTkLabel(sidebar, text="显示语言")
        lang_label.pack(padx=20, pady=(14, 4), anchor="w")
        lang_select = ctk.CTkOptionMenu(sidebar, values=["中文", "English", "Русский"], variable=self.language_display_var)
        lang_select.pack(fill="x", padx=20)

        # 开关
        switches = ctk.CTkFrame(sidebar)
        switches.pack(fill="x", padx=20, pady=(14, 8))
        tflite_switch = ctk.CTkSwitch(switches, text="使用 TFLite", variable=self.use_tflite_var)
        tflite_switch.pack(anchor="w", pady=(0, 6))
        m1_switch = ctk.CTkSwitch(switches, text="启用 M1/M2 优化", variable=self.use_m1_var)
        m1_switch.pack(anchor="w", pady=(0, 6))
        fps_switch = ctk.CTkSwitch(switches, text="显示 FPS", variable=self.show_fps_var)
        fps_switch.pack(anchor="w")

        # 保存结果（当前支持图片识别保存）
        save_frame = ctk.CTkFrame(sidebar)
        save_frame.pack(fill="x", padx=20, pady=(10, 8))
        self.save_result_var = tk.BooleanVar(value=False)
        save_switch = ctk.CTkSwitch(save_frame, text="保存识别结果（仅图片）", variable=self.save_result_var)
        save_switch.pack(anchor="w")
        self.save_dir_var = tk.StringVar(value="")
        dir_row = ctk.CTkFrame(save_frame)
        dir_row.pack(fill="x", pady=(6, 0))
        self.save_dir_entry = ctk.CTkEntry(dir_row, textvariable=self.save_dir_var, placeholder_text="请选择保存目录")
        self.save_dir_entry.pack(side="left", fill="x", expand=True)
        dir_btn = ctk.CTkButton(dir_row, text="选择目录", width=90, command=self._choose_save_dir)
        dir_btn.pack(side="left", padx=(8, 0))

        # 操作按钮
        actions = ctk.CTkFrame(sidebar)
        actions.pack(fill="x", padx=20, pady=(10, 20))
        self.btn_camera = ctk.CTkButton(actions, text="打开摄像头识别", command=self._handle_camera)
        self.btn_camera.pack(fill="x", pady=6)
        self.btn_image = ctk.CTkButton(actions, text="上传图片识别", command=self._handle_image)
        self.btn_image.pack(fill="x", pady=6)
        self.btn_video = ctk.CTkButton(actions, text="上传视频识别", command=self._handle_video)
        self.btn_video.pack(fill="x", pady=6)

        # 训练模块
        train_title = ctk.CTkLabel(sidebar, text="训练模型", font=("PingFang SC", 16, "bold"))
        train_title.pack(padx=20, pady=(10, 6), anchor="w")

        # 数据集选择
        ds_frame = ctk.CTkFrame(sidebar)
        ds_frame.pack(fill="x", padx=20, pady=(4, 4))
        ds_row1 = ctk.CTkFrame(ds_frame)
        ds_row1.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(ds_row1, text="目录数据集").pack(side="left")
        ctk.CTkButton(ds_row1, text="选择", width=70, command=self._choose_train_dataset_dir).pack(side="right")
        ctk.CTkEntry(ds_frame, textvariable=self.train_dataset_dir_var, placeholder_text="包含子目录的图像数据集路径").pack(fill="x")

        ds_row2 = ctk.CTkFrame(sidebar)
        ds_row2.pack(fill="x", padx=20, pady=(6, 2))
        ctk.CTkLabel(ds_row2, text="FER2013 CSV").pack(side="left")
        ctk.CTkButton(ds_row2, text="选择", width=70, command=self._choose_train_fer_csv).pack(side="right")
        ctk.CTkEntry(sidebar, textvariable=self.train_fer_csv_var, placeholder_text="datasSource/fer2013.csv").pack(fill="x", padx=20)

        # 训练参数
        hp_frame = ctk.CTkFrame(sidebar)
        hp_frame.pack(fill="x", padx=20, pady=(8, 4))
        row_hp1 = ctk.CTkFrame(hp_frame); row_hp1.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(row_hp1, text="轮数").pack(side="left")
        ctk.CTkEntry(row_hp1, width=80, textvariable=self.train_epochs_var).pack(side="left", padx=(6, 12))
        ctk.CTkLabel(row_hp1, text="批次").pack(side="left")
        ctk.CTkEntry(row_hp1, width=80, textvariable=self.train_batch_var).pack(side="left", padx=(6, 0))

        row_hp2 = ctk.CTkFrame(hp_frame); row_hp2.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(row_hp2, text="学习率").pack(side="left")
        ctk.CTkEntry(row_hp2, width=120, textvariable=self.train_lr_var).pack(side="left", padx=(6, 12))
        ctk.CTkLabel(row_hp2, text="增强").pack(side="left")
        ctk.CTkOptionMenu(row_hp2, values=["none","low","medium","high","very_high"], variable=self.train_augment_var).pack(side="left", padx=(6,0))

        cw_row = ctk.CTkFrame(hp_frame); cw_row.pack(fill="x", pady=(2, 2))
        ctk.CTkSwitch(cw_row, text="类别权重平衡", variable=self.train_class_weight_var).pack(anchor="w")

        # 保存设置
        sv_frame = ctk.CTkFrame(sidebar)
        sv_frame.pack(fill="x", padx=20, pady=(8, 6))
        name_row = ctk.CTkFrame(sv_frame); name_row.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(name_row, text="实验名").pack(side="left")
        ctk.CTkEntry(name_row, textvariable=self.train_exp_name_var).pack(side="left", fill="x", expand=True, padx=(6,0))
        dir_row = ctk.CTkFrame(sv_frame); dir_row.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(dir_row, text="保存到").pack(side="left")
        ctk.CTkButton(dir_row, text="选择", width=70, command=self._choose_train_save_dir).pack(side="right")
        ctk.CTkEntry(sv_frame, textvariable=self.train_save_dir_var).pack(fill="x")

        # 启动训练按钮
        self.btn_train = ctk.CTkButton(sidebar, text="开始训练模型", command=self._handle_train)
        self.btn_train.pack(fill="x", padx=20, pady=(6, 14))

        # 右侧：日志与说明
        main_area = ctk.CTkFrame(self)
        main_area.pack(side="left", fill="both", expand=True)

        guide_title = ctk.CTkLabel(main_area, text="使用说明", font=("PingFang SC", 18, "bold"))
        guide_title.pack(padx=16, pady=(16, 8), anchor="w")

        guide_text = (
            "- 点击左侧按钮启动对应功能。\n"
            "- 摄像头/视频将弹出 OpenCV 窗口，在窗口按 ESC 键退出。\n"
            "- 图片识别会直接显示结果弹窗，或在命令行中查看详情。\n"
            "- 首次运行会加载模型，耗时取决于设备。"
        )
        self.guide_label = ctk.CTkLabel(main_area, text=guide_text, justify="left")
        self.guide_label.pack(padx=16, pady=(0, 8), anchor="w")

        log_title = ctk.CTkLabel(main_area, text="运行日志", font=("PingFang SC", 16, "bold"))
        log_title.pack(padx=16, pady=(12, 6), anchor="w")
        self.log_text = tk.Text(main_area, height=24)
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def _log(self, text: str):
        # 将 UI 更新调度到主线程，避免线程中直接操作 Tk 导致 TclError
        def _append():
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")

        try:
            self.after(0, _append)
        except Exception:
            # 兜底打印到控制台
            print(text)

    def _show_error(self, title: str, message: str):
        # 在线程中调用时通过 after 委托到主线程
        try:
            self.after(0, lambda: messagebox.showerror(title, message))
        except Exception:
            print(f"{title}: {message}")

    def _resolve_python_exec(self) -> str:
        # 优先使用当前虚拟环境
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            cand = os.path.join(venv, "bin", "python")
            if os.path.exists(cand):
                return cand
        # 其次尝试项目内常见虚拟环境
        for name in [".venv1", ".venv", "venv"]:
            cand = os.path.join(PROJECT_ROOT, name, "bin", "python")
            if os.path.exists(cand):
                return cand
        # 回退为当前进程解释器
        return sys.executable

    def _bring_to_front(self):
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
            # 临时置顶，再还原，避免永久顶置干扰其他窗口
            try:
                self.attributes('-topmost', True)
                self.after(500, lambda: self.attributes('-topmost', False))
            except Exception:
                pass
        except Exception:
            pass

    def _set_running(self, running: bool, note: str = ""):
        self.is_running = running
        def _apply():
            state = "disabled" if running else "normal"
            self.btn_camera.configure(state=state)
            self.btn_image.configure(state=state)
            self.btn_video.configure(state=state)
            if note:
                self._log(note)
        try:
            self.after(0, _apply)
        except Exception:
            pass

    def _run_subprocess(self, args: list[str], start_note: str, end_note: str, *, cwd: str | None = None, on_line=None):
        if self.is_running:
            self._log("已有任务在运行，请稍候…")
            return
        self._set_running(True, start_note)

        def reader_thread(proc: Popen):
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    text = line.rstrip()
                    self._log(text)
                    if on_line:
                        try:
                            on_line(text)
                        except Exception:
                            pass
            finally:
                code = proc.wait()
                self._log(f"进程退出，返回码: {code}")
                self._log(end_note)
                self._set_running(False)

        try:
            note = f" (工作目录: {cwd})" if cwd else ""
            self._log("启动命令: " + " ".join(args) + note)
            proc = subprocess.Popen(args, stdout=PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
            self.current_process = proc
            threading.Thread(target=reader_thread, args=(proc,), daemon=True).start()
        except Exception as e:
            self._log(f"启动子进程失败: {e}")
            self._show_error("错误", f"启动子进程失败:\n{e}")
            self._set_running(False)

    def _get_language_code(self) -> str:
        return self._language_name_to_code.get(self.language_display_var.get(), "zh")

    def _choose_model_file(self):
        filetypes = [("TFLite / Keras 模型", "*.tflite *.h5"), ("所有文件", "*.*")]
        path = filedialog.askopenfilename(title="选择模型文件", initialdir=os.path.join(PROJECT_ROOT, "models"), filetypes=filetypes)
        if path:
            self.model_path_var.set(path)

    def _choose_save_dir(self):
        path = filedialog.askdirectory(title="选择保存目录", initialdir=PROJECT_ROOT)
        if path:
            self.save_dir_var.set(path)

    def _choose_train_dataset_dir(self):
        path = filedialog.askdirectory(title="选择目录数据集", initialdir=PROJECT_ROOT)
        if path:
            self.train_dataset_dir_var.set(path)

    def _choose_train_fer_csv(self):
        path = filedialog.askopenfilename(title="选择 FER2013 CSV", initialdir=os.path.join(PROJECT_ROOT, "datasSource"), filetypes=[("CSV","*.csv"), ("所有文件","*.*")])
        if path:
            self.train_fer_csv_var.set(path)

    def _choose_train_save_dir(self):
        path = filedialog.askdirectory(title="选择模型保存目录", initialdir=os.path.join(PROJECT_ROOT, "models"))
        if path:
            self.train_save_dir_var.set(path)

    def _ensure_recognizer(self):
        if self.recognizer is not None:
            return
        model_path = self.model_path_var.get().strip()
        use_tflite = self.use_tflite_var.get() or (model_path.endswith(".tflite"))
        try:
            self._log(f"初始化模型: {model_path} (TFLite: {use_tflite}, M1优化: {self.use_m1_var.get()})")
            self.recognizer = UnifiedEmotionRecognizer(
                model_path=model_path,
                use_m1_optimizations=self.use_m1_var.get(),
                use_tflite=use_tflite,
                verbose=True,
                language="zh"
            )
            self._log("模型已就绪。")
        except Exception as e:
            self._log(f"初始化模型失败: {e}")
            messagebox.showerror("错误", f"初始化模型失败:\n{e}")

    def _handle_camera(self):
        script = os.path.join(PROJECT_ROOT, "unified_emotion_recognition.py")
        resolution = (int(self.res_w_var.get()), int(self.res_h_var.get()))
        args = [
            self.python_exec,
            script,
            "--camera",
            "--resolution", str(resolution[0]), str(resolution[1]),
            "--skip_frames", str(int(self.skip_frames_var.get())),
            "--confidence", str(float(self.confidence_var.get())),
            "--language", self._get_language_code(),
        ]
        if self.show_fps_var.get():
            args.append("--show_fps")
        if self.use_m1_var.get():
            args.append("--use_m1_optimizations")
        # 模型/推理后端
        model_path = self.model_path_var.get().strip()
        if model_path:
            args.extend(["--model", model_path])
        if self.use_tflite_var.get() or model_path.endswith(".tflite"):
            args.append("--use_tflite")

        self._run_subprocess(args, "正在启动摄像头识别… 在弹出的 OpenCV 窗口按 ESC 退出。", "摄像头识别结束。")

    def _handle_image(self):
        if self.is_running:
            self._log("已有任务在运行，请稍候…")
            return
        filetypes = [("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        image_path = filedialog.askopenfilename(title="选择待识别图片", initialdir=os.path.join(PROJECT_ROOT, "images"), filetypes=filetypes)
        if not image_path:
            return
        script = os.path.join(PROJECT_ROOT, "unified_emotion_recognition.py")
        args = [
            self.python_exec,
            script,
            "--image", image_path,
            "--confidence", str(float(self.confidence_var.get())),
            "--language", self._get_language_code(),
        ]
        if self.use_m1_var.get():
            args.append("--use_m1_optimizations")
        model_path = self.model_path_var.get().strip()
        if model_path:
            args.extend(["--model", model_path])
        if self.use_tflite_var.get() or model_path.endswith(".tflite"):
            args.append("--use_tflite")

        # 处理保存：切换子进程工作目录 + --save_result
        run_cwd = None
        if self.save_result_var.get():
            sel_dir = self.save_dir_var.get().strip()
            if sel_dir:
                run_cwd = sel_dir
            args.append("--save_result")

        self._run_subprocess(args, "正在进行图片识别…", "图片识别完成。", cwd=run_cwd)

    def _handle_video(self):
        if self.is_running:
            self._log("已有任务在运行，请稍候…")
            return
        filetypes = [("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        video_path = filedialog.askopenfilename(title="选择待识别视频", initialdir=os.path.join(PROJECT_ROOT, "video"), filetypes=filetypes)
        if not video_path:
            return
        script = os.path.join(PROJECT_ROOT, "unified_emotion_recognition.py")
        args = [
            self.python_exec,
            script,
            "--video", video_path,
            "--confidence", str(float(self.confidence_var.get())),
            "--language", self._get_language_code(),
        ]
        if self.show_fps_var.get():
            args.append("--show_fps")
        if self.use_m1_var.get():
            args.append("--use_m1_optimizations")
        model_path = self.model_path_var.get().strip()
        if model_path:
            args.extend(["--model", model_path])
        if self.use_tflite_var.get() or model_path.endswith(".tflite"):
            args.append("--use_tflite")

        self._run_subprocess(args, "正在进行视频识别… 在弹出的 OpenCV 窗口按 ESC 退出。", "视频识别完成。")

    def _handle_train(self):
        if self.is_running:
            self._log("已有任务在运行，请稍候…")
            return
        script = os.path.join(PROJECT_ROOT, "combined_train.py")
        args = [
            self.python_exec,
            script,
            "--epochs", str(int(self.train_epochs_var.get())),
            "--batch_size", str(int(self.train_batch_var.get())),
            "--learning_rate", str(float(self.train_lr_var.get())),
            "--augment_level", self.train_augment_var.get(),
            "--save_dir", self.train_save_dir_var.get().strip() or os.path.join(PROJECT_ROOT, "models"),
            "--log_dir", os.path.join(PROJECT_ROOT, "logs"),
        ]
        if self.train_class_weight_var.get():
            args.append("--class_weight")
        exp = self.train_exp_name_var.get().strip()
        if exp:
            args.extend(["--experiment_name", exp])

        # 数据集参数（二选一，优先 CSV）
        fer_csv = self.train_fer_csv_var.get().strip()
        ds_dir = self.train_dataset_dir_var.get().strip()
        if fer_csv:
            args.extend(["--fer2013", fer_csv])
        elif ds_dir:
            args.extend(["--dataset", ds_dir])
        else:
            self._show_error("提示", "请先选择目录数据集或 FER2013 CSV 文件")
            return

        # 行处理回调：捕获“最终模型已保存至: ...”并自动切换模型路径
        def on_line(text: str):
            key = "最终模型已保存至: "
            if text.startswith(key):
                new_model = text[len(key):].strip()
                if os.path.isfile(new_model):
                    self.model_path_var.set(new_model)
                    self._log(f"已自动切换模型路径: {new_model}")

        self._run_subprocess(args, "开始训练模型… 训练过程较长，请耐心等待。", "训练结束。", on_line=on_line)


def main():
    app = EmotionGUI()
    app.mainloop()


if __name__ == "__main__":
    main()



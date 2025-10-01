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

        self._build_layout()
        # 启动后将窗口置顶并聚焦
        self.after(100, self._bring_to_front)

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

    def _run_subprocess(self, args: list[str], start_note: str, end_note: str, *, cwd: str | None = None):
        if self.is_running:
            self._log("已有任务在运行，请稍候…")
            return
        self._set_running(True, start_note)

        def reader_thread(proc: Popen):
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._log(line.rstrip())
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
            sys.executable,
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
            sys.executable,
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
            sys.executable,
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


def main():
    app = EmotionGUI()
    app.mainloop()


if __name__ == "__main__":
    main()



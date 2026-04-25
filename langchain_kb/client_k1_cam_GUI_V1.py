import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import threading
import requests
import time
import os
import cv2
import numpy as np

# 导入推理脚本 (请确保 yolo_infer.py 在同级目录)
from yolo_infer import YOLOv12_ONNX_Inference

# ================= 配置 =================
DEFAULT_SERVER = "http://172.28.82.33:8001"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
ONNX_MODEL_PATH = "/home/yolov12/320n/YOLOv12n-ShuffleNetv2-C3k2-320.onnx"

# K1 摄像头 GStreamer 配置
GST_STR = (
    'spacemitsrc location=/home/cam/csi3_camera_ov5647_auto.json close-dmabuf=1 ! '
    'video/x-raw,format=NV12,width=1920,height=1080 ! appsink'
)


class MacauEcoSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Macau Eco AI System (Spacemit K1)")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.bg_color = "#1e1e1e"
        self.root.configure(bg=self.bg_color)
        self.root.resizable(False, False)

        # 全局状态
        self.running = True
        self.detected_label = None
        self.rag_query = ""
        self.current_frame = None
        self.last_result_img = None  # 存储带框的BGR图片
        self.tk_res_img = None  # 关键：存储Tkinter图片引用，防止回收

        # 初始化 UI
        self._init_ui()

        # 启动实时时钟
        self.update_clock()

        # 初始化 YOLO 模型 (后台线程)
        self.detector = None
        self.init_detector_thread()

        # 摄像头初始化
        self.log_to_report("System: Initializing Camera...")
        self.cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.log_to_report("Error: Camera not found.")

        self.update_camera_feed()

    def _init_ui(self):
        # --- Top Frame (IP, Clock, Status) ---
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 第一行：IP 和 连接指示
        ip_line = tk.Frame(top_frame, bg=self.bg_color)
        ip_line.pack(fill=tk.X)
        tk.Label(ip_line, text="Server IP:", bg=self.bg_color, fg="white").pack(side=tk.LEFT)
        self.server_entry = tk.Entry(ip_line, bg="#333", fg="white", width=20)
        self.server_entry.pack(side=tk.LEFT, padx=5)
        self.server_entry.insert(0, DEFAULT_SERVER)

        # 连接状态指示 (English)
        self.lbl_conn_status = tk.Label(ip_line, text="● Offline", bg=self.bg_color, fg="gray",
                                        font=("Arial", 10, "bold"))
        self.lbl_conn_status.pack(side=tk.LEFT, padx=10)

        # 模型加载指示
        self.lbl_model_status = tk.Label(ip_line, text="● Model Loading...", bg=self.bg_color, fg="#ffcc00",
                                         font=("Arial", 10))
        self.lbl_model_status.pack(side=tk.RIGHT)

        # 第二行：实时时间 (显示在IP下方)
        self.lbl_clock = tk.Label(top_frame, text="Time: Loading...", bg=self.bg_color, fg="#888", font=("Arial", 9))
        self.lbl_clock.pack(anchor=tk.W, pady=2)

        # --- Middle Frame (Monitor & Buttons) ---
        middle_frame = tk.Frame(self.root, bg=self.bg_color)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. 实时监控窗口
        self.live_frame = tk.LabelFrame(middle_frame, text=" Live Monitor ", bg=self.bg_color, fg="#00ff00",
                                        font=("Arial", 10, "bold"))
        self.live_frame.place(x=0, y=0, width=320, height=250)
        self.lbl_video = tk.Label(self.live_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # 2. 识别结果窗口 (显示带框的图)
        self.result_frame = tk.LabelFrame(middle_frame, text=" Recognition Result ", bg=self.bg_color, fg="#ffcc00",
                                          font=("Arial", 10, "bold"))
        self.result_frame.place(x=330, y=0, width=320, height=250)
        self.lbl_result = tk.Label(self.result_frame, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)
        self.lbl_result.config(image=self._create_placeholder_image("Waiting for inference..."))

        # 3. 按钮栏 (增加高度和间距)
        btn_frame = tk.Frame(middle_frame, bg=self.bg_color)
        btn_frame.place(x=660, y=5, width=130, height=250)

        # ipady: 按钮内部高度填充, pady: 按钮外部垂直间距
        btn_opts = {"font": ("Arial", 10, "bold"), "fg": "white", "bd": 0, "activebackground": "#444"}

        self.btn_rec = tk.Button(btn_frame, text="Recognize", bg="#28a745", command=self.do_recognition, **btn_opts)
        self.btn_rec.pack(fill=tk.X, pady=10, ipady=12)

        self.btn_info = tk.Button(btn_frame, text="Sci-Pop", bg="#fd7e14", command=self.call_rag_server, **btn_opts)
        self.btn_info.pack(fill=tk.X, pady=10, ipady=12)

        self.btn_save = tk.Button(btn_frame, text="Save", bg="#17a2b8", command=self.save_image, **btn_opts)
        self.btn_save.pack(fill=tk.X, pady=10, ipady=12)

        self.btn_exit = tk.Button(btn_frame, text="Exit", bg="#6c757d", command=self.on_exit, **btn_opts)
        self.btn_exit.pack(side=tk.BOTTOM, fill=tk.X, pady=5, ipady=8)

        # --- Bottom Frame (Report Area) ---
        bottom_frame = tk.LabelFrame(self.root, text=" AI Knowledge Report ", bg=self.bg_color, fg="#ff6666",
                                     font=("Arial", 10, "bold"))
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        bottom_frame.pack_propagate(False)
        bottom_frame.configure(height=140)

        self.txt_report = scrolledtext.ScrolledText(bottom_frame, bg="#2b2b2b", fg="#e0e0e0",
                                                    font=("Microsoft YaHei", 12), bd=0)
        self.txt_report.pack(fill=tk.BOTH, expand=True)

    def init_detector_thread(self):
        """异步加载模型"""

        def load():
            try:
                self.detector = YOLOv12_ONNX_Inference(ONNX_MODEL_PATH, input_size=(320, 320))
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.log_to_report(f"Model Load Error: {e}"))

        threading.Thread(target=load, daemon=True).start()

    def _on_model_loaded(self):
        self.lbl_model_status.config(text="● Model Ready", fg="#00ff00")
        self.log_to_report("System: YOLO Model Loaded Successfully.")

    def update_clock(self):
        """更新系统时间"""
        now = time.strftime("Time: %Y-%m-%d %H:%M:%S")
        self.lbl_clock.config(text=now)
        self.root.after(1000, self.update_clock)

    def do_recognition(self):
        """识别逻辑 - 修复了只显示首字母的 Bug"""
        if self.detector is None:
            messagebox.showwarning("Wait", "Model is still loading...")
            return
        if self.current_frame is None:
            self.log_to_report("Error: No Camera Feed.")
            return

        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, self.current_frame)
        self.log_to_report("Action: Running Recognition...")

        def task():
            try:
                # 推理：假设返回 (绘制好的图像, 标签列表或字符串)
                outputs = self.detector.detect(temp_path)
                res_img = outputs[0]
                labels = outputs[1]

                # 1. 更新图像显示 (保持引用防止回收)
                if res_img is not None:
                    self.last_result_img = res_img
                    res_img_resized = cv2.resize(res_img, (310, 230))
                    res_img_rgb = cv2.cvtColor(res_img_resized, cv2.COLOR_BGR2RGB)
                    pil_res = Image.fromarray(res_img_rgb)

                    def update_ui_img(p_img):
                        self.tk_res_img = ImageTk.PhotoImage(p_img)
                        self.lbl_result.config(image=self.tk_res_img)

                    self.root.after(0, lambda: update_ui_img(pil_res))

                # 2. 【核心修复】处理标签，确保获取全称
                full_label_name = ""
                if labels:
                    if isinstance(labels, (list, tuple)) and len(labels) > 0:
                        # 如果是列表，取第一个元素的全称
                        full_label_name = str(labels[0])
                    elif isinstance(labels, str):
                        # 如果本身就是字符串，直接使用全称
                        full_label_name = labels

                if full_label_name:
                    self.detected_label = full_label_name
                    self.rag_query = f"Introduce the Morphology, Habits, Distribution, and Ecological Role of {self.detected_label}."
                    # 输出全称到窗口
                    self.root.after(0, lambda: self.log_to_report(f"Success: Recognized [{self.detected_label}]."))
                else:
                    self.detected_label = None
                    self.rag_query = ""
                    self.root.after(0, lambda: self.log_to_report("Result: No object detected."))

            except Exception as e:
                self.root.after(0, lambda: self.log_to_report(f"Detection Error: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def call_rag_server(self):
        """请求 RAG 服务器并计算延迟"""
        if not self.rag_query:
            messagebox.showinfo("Tip", "Please click 'Recognize' first.")
            return

        server_url = self.server_entry.get().strip()
        self.log_to_report(f"> Requesting RAG Knowledge...", clear=True)

        start_time = time.time()  # 记录提交时间

        def task():
            try:
                resp = requests.post(
                    f"{server_url}/chat",
                    json={"query": self.rag_query},
                    timeout=30,
                    stream=True
                )

                if resp.status_code == 200:
                    self.root.after(0, lambda: self.lbl_conn_status.config(text="● Connected", fg="#00ff00"))

                    first_token = False
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            if not first_token:
                                latency = time.time() - start_time
                                self.root.after(0, lambda l=latency: self.append_to_report(
                                    f"[Prompt Latency: {l:.2f}s]\n\n"))
                                first_token = True

                            self.root.after(0, lambda c=chunk: self.append_to_report(c))

                    self.root.after(0, lambda: self.append_to_report("\n\n-- Report Completed --"))
                else:
                    self.root.after(0, lambda: self.lbl_conn_status.config(text="● Server Error", fg="red"))
            except Exception as e:
                self.root.after(0, lambda: self.lbl_conn_status.config(text="● Connection Failed", fg="red"))
                self.root.after(0, lambda: self.log_to_report(f"Network Error: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def update_camera_feed(self):
        if not self.running: return
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                self.current_frame = frame
                frame_resized = cv2.resize(frame, (310, 230))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                self.tk_live_img = ImageTk.PhotoImage(pil_img)
                self.lbl_video.config(image=self.tk_live_img)
        self.root.after(30, self.update_camera_feed)

    def _create_placeholder_image(self, text):
        img = Image.new('RGB', (310, 230), color="#202020")
        draw = ImageDraw.Draw(img)
        draw.text((60, 100), text, fill="white")
        return ImageTk.PhotoImage(img)

    def append_to_report(self, text):
        self.txt_report.configure(state='normal')
        self.txt_report.insert(tk.END, text)
        self.txt_report.see(tk.END)
        self.txt_report.configure(state='disabled')

    def log_to_report(self, text, clear=False):
        self.txt_report.configure(state='normal')
        if clear: self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"{text}\n")
        self.txt_report.see(tk.END)
        self.txt_report.configure(state='disabled')

    def save_image(self):
        if self.last_result_img is not None:
            if not os.path.exists("captures"): os.makedirs("captures")
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"captures/res_{ts}.jpg"
            cv2.imwrite(path, self.last_result_img)
            self.log_to_report(f"System: Image saved to {path}")

    def on_exit(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MacauEcoSystemApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()
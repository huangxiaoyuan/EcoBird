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
try:
    from yolo_infer import YOLOv12_ONNX_Inference
except ImportError:
    pass

# ================= 适配 800*480 布局参数 =================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480

# 蓝色预览区尺寸 (等大)
CAM_W, CAM_H = 300, 225

# 右侧红色按键区宽度
SIDE_BAR_W = 180

DEFAULT_SERVER = "http://172.28.82.33:8001"
ONNX_MODEL_PATH = "/home/yolov12/320n/YOLOv12n-ShuffleNetv2-C3k2-320.onnx"

# K1 摄像头 GStreamer 配置
GST_STR = (
    'spacemitsrc location=/home/cam/csi3_camera_ov5647_auto.json close-dmabuf=1 ! '
    'video/x-raw,format=NV12,width=1920,height=1080 ! appsink'
)


class MacauEcoSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Macau Eco AI")

        # 1. 设置全屏启动
        self.root.attributes('-fullscreen', True)
        self.is_fullscreen = True
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.bg_color = "#1a1a1a"
        self.root.configure(bg=self.bg_color)

        self.running = True
        self.current_frame = None
        self.last_result_img = None
        self.detector = None
        self.detected_label = None
        self.rag_query = ""

        # 初始化UI
        self._init_ui()

        # 启动后台任务
        self.init_detector_thread()
        self.cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
        self.update_camera_feed()

    def _init_ui(self):
        # --- A. 顶栏信息 (极简) ---
        top_info = tk.Frame(self.root, bg="#222", height=35)
        top_info.place(x=0, y=0, width=WINDOW_WIDTH, height=35)

        tk.Label(top_info, text="IP:", bg="#222", fg="#888", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.server_entry = tk.Entry(top_info, bg="#333", fg="white", bd=0, font=("Arial", 10))
        self.server_entry.place(x=35, y=7, width=140, height=20)
        self.server_entry.insert(0, DEFAULT_SERVER)

        self.lbl_conn = tk.Label(top_info, text="● Offline", bg="#222", fg="gray", font=("Arial", 9, "bold"))
        self.lbl_conn.place(x=185, y=7)

        # --- B. 蓝色区域：预览与识别 (等大并排) ---
        # 实时窗口
        self.box_live = tk.LabelFrame(self.root, text=" Live ", bg=self.bg_color, fg="#00ff00",
                                      font=("Arial", 10, "bold"))
        self.box_live.place(x=10, y=40, width=CAM_W + 10, height=CAM_H + 25)
        self.lbl_video = tk.Label(self.box_live, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # 结果窗口
        self.box_res = tk.LabelFrame(self.root, text=" Recognition Result ", bg=self.bg_color, fg="#ffcc00",
                                     font=("Arial", 10, "bold"))
        self.box_res.place(x=CAM_W + 25, y=40, width=CAM_W + 10, height=CAM_H + 25)
        self.lbl_result = tk.Label(self.box_res, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)

        # --- C. 黄色区域：AI 报告区 (下方大面积) ---
        self.box_report = tk.LabelFrame(self.root, text=" AI Knowledge Report ", bg=self.bg_color, fg="#ff6666",
                                        font=("Arial", 10, "bold"))
        self.box_report.place(x=10, y=CAM_H + 75, width=CAM_W * 2 + 25, height=170)

        # 极大滚动条 (width=45 适配触摸)
        self.scroll = tk.Scrollbar(self.box_report, orient=tk.VERTICAL, width=45)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_report = tk.Text(self.box_report, bg="#111", fg="#eee",
                                  font=("Microsoft YaHei", 12), bd=0, padx=10, pady=10,
                                  yscrollcommand=self.scroll.set)
        self.txt_report.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.config(command=self.txt_report.yview)

        # --- D. 红色区域：按键区 (右侧垂直) ---
        btn_container = tk.Frame(self.root, bg="#252525")
        btn_container.place(x=WINDOW_WIDTH - SIDE_BAR_W, y=35, width=SIDE_BAR_W, height=WINDOW_HEIGHT - 35)

        btn_font = ("Arial", 12, "bold")
        # 识别按钮
        tk.Button(btn_container, text="RECOGNIZE", bg="#28a745", fg="white", font=btn_font,
                  command=self.do_recognition).pack(fill=tk.X, padx=10, pady=8, ipady=18)

        # 知识按钮
        tk.Button(btn_container, text="SCI-INFO", bg="#fd7e14", fg="white", font=btn_font,
                  command=self.call_rag_server).pack(fill=tk.X, padx=10, pady=8, ipady=18)

        # 保存按钮
        tk.Button(btn_container, text="SAVE IMG", bg="#17a2b8", fg="white", font=btn_font,
                  command=self.save_image).pack(fill=tk.X, padx=10, pady=8, ipady=12)

        # 全屏切换按钮 (新增)
        tk.Button(btn_container, text="WINDOW/FULL", bg="#6f42c1", fg="white", font=("Arial", 10, "bold"),
                  command=self.toggle_fullscreen).pack(fill=tk.X, padx=10, pady=8, ipady=8)

        # 退出按钮
        tk.Button(btn_container, text="EXIT APP", bg="#dc3545", fg="white", font=("Arial", 10, "bold"),
                  command=self.on_exit).pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10, ipady=5)

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

    def init_detector_thread(self):
        def load():
            try:
                if 'YOLOv12_ONNX_Inference' in globals():
                    self.detector = YOLOv12_ONNX_Inference(ONNX_MODEL_PATH, input_size=(320, 320))
                    self.log_to_report("System: AI Engine Initialized.")
            except Exception as e:
                self.log_to_report(f"Model Error: {e}")

        threading.Thread(target=load, daemon=True).start()

    def do_recognition(self):
        if not self.detector or self.current_frame is None: return
        self.log_to_report("Action: Analyzing Scene...", clear=True)
        cv2.imwrite("temp.jpg", self.current_frame)

        def task():
            try:
                res_img, labels = self.detector.detect("temp.jpg")
                if res_img is not None:
                    self.last_result_img = res_img
                    img_rgb = cv2.cvtColor(cv2.resize(res_img, (CAM_W, CAM_H)), cv2.COLOR_BGR2RGB)
                    tk_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                    self.root.after(0, lambda: self._update_res_ui(tk_img))
                if labels:
                    self.detected_label = str(labels[0])
                    self.rag_query = f"Provide a detailed scientific introduction of {self.detected_label}."
                    self.root.after(0, lambda: self.log_to_report(f"Result: {self.detected_label}"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_report(f"AI Error: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def _update_res_ui(self, tk_img):
        self.tk_res_img = tk_img
        self.lbl_result.config(image=self.tk_res_img)

    def call_rag_server(self):
        if not self.rag_query: return
        self.log_to_report("\n> Fetching AI Knowledge Base...", clear=False)

        def task():
            try:
                url = self.server_entry.get().strip()
                resp = requests.post(f"{url}/chat", json={"query": self.rag_query}, timeout=20, stream=True)
                if resp.status_code == 200:
                    self.root.after(0, lambda: self.lbl_conn.config(text="● Online", fg="#00ff00"))
                    for chunk in resp.iter_content(decode_unicode=True):
                        if chunk: self.root.after(0, lambda c=chunk: self.append_to_report(c))
                else:
                    self.root.after(0, lambda: self.lbl_conn.config(text="● Server Error", fg="red"))
            except:
                self.root.after(0, lambda: self.lbl_conn.config(text="● Conn Failed", fg="red"))

        threading.Thread(target=task, daemon=True).start()

    def update_camera_feed(self):
        if not self.running: return
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                except:
                    pass
                self.current_frame = frame
                # 调整到 UI 定义的尺寸
                small = cv2.resize(frame, (CAM_W, CAM_H))
                self.tk_live = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                self.lbl_video.config(image=self.tk_live)
        self.root.after(30, self.update_camera_feed)

    def append_to_report(self, text):
        self.txt_report.insert(tk.END, text)
        self.txt_report.see(tk.END)

    def log_to_report(self, text, clear=False):
        if clear: self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"{text}\n")
        self.txt_report.see(tk.END)

    def save_image(self):
        if self.last_result_img is not None:
            path = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(path, self.last_result_img)
            self.log_to_report(f"System: Image saved as {path}")

    def on_exit(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MacauEcoSystemApp(root)
    # 按键盘 ESC 键也可以退出全屏
    root.bind("<Escape>", lambda e: app.toggle_fullscreen())
    root.mainloop()
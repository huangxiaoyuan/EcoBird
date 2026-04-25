#❗初始化摄像头，需要知道摄像头是安装在哪个通通上，配置好46行代码
#❗修改服务器地址，找server运行的地址
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import threading
import requests
import time
import os
import cv2
import numpy as np

# 导入推理脚本
from yolo_infer import YOLOv12_ONNX_Inference

# ================= 配置 =================
DEFAULT_SERVER = "http://127.0.0.1:8001"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480

# 模型路径
ONNX_MODEL_PATH = "E:/K1/yolov12/model_320_n/YOLOv12n-ShuffleNetv2-GhostC2f-320.onnx"


class MacauEcoSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Macau Eco AI System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.bg_color = "#1e1e1e"
        self.root.configure(bg=self.bg_color)
        self.root.resizable(False, False)

        # 全局状态
        self.running = True
        self.detected_label = None
        self.rag_query = ""
        self.current_frame = None  # 原始摄像头画面
        self.last_result_img = None  # 存储识别后的画框图片

        # 初始化 YOLO 模型
        self.detector = None
        self.init_detector_thread()

        # 初始化 UI
        self._init_ui()

        # ❗初始化摄像头，需要知道摄像头是安装在哪个dcsi通通上
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.log_to_report("Error: Could not open camera.")

        # 启动实时刷新
        self.update_camera_feed()

    def init_detector_thread(self):
        def load():
            try:
                print("Loading YOLO model...")
                self.detector = YOLOv12_ONNX_Inference(ONNX_MODEL_PATH, input_size=(320, 320))
                print("Model loaded.")
                self.root.after(0, lambda: self.log_to_report("System Ready. Model Loaded."))
            except Exception as e:
                print(f"Failed to load model: {e}")
                # 【修复1】先转存为字符串变量
                err_msg = f"Error loading model: {e}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        threading.Thread(target=load, daemon=True).start()

    def _init_ui(self):
        # === Top: Server Address ===
        top_frame = tk.Frame(self.root, bg=self.bg_color, height=30)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(top_frame, text="Server:", bg=self.bg_color, fg="white").pack(side=tk.LEFT)
        self.server_entry = tk.Entry(top_frame, bg="#333", fg="white", width=30)
        self.server_entry.pack(side=tk.LEFT, padx=5)
        self.server_entry.insert(0, DEFAULT_SERVER)

        # === Middle Area ===
        middle_frame = tk.Frame(self.root, bg=self.bg_color)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. Live Monitor
        self.live_frame = tk.LabelFrame(middle_frame, text=" Live Monitor ", bg=self.bg_color, fg="#00ff00",
                                        font=("Arial", 10, "bold"))
        self.live_frame.place(x=0, y=0, width=320, height=260)
        self.lbl_video = tk.Label(self.live_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 2. Recognition Result
        self.result_frame = tk.LabelFrame(middle_frame, text=" Recognition Result ", bg=self.bg_color, fg="#ffcc00",
                                          font=("Arial", 10, "bold"))
        self.result_frame.place(x=330, y=0, width=320, height=260)
        self.lbl_result = tk.Label(self.result_frame, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.lbl_result.config(image=self._create_placeholder_image("Waiting...", "gray"))

        # 3. Buttons (Right Side)
        btn_frame = tk.Frame(middle_frame, bg=self.bg_color)
        btn_frame.place(x=660, y=10, width=130, height=260)

        btn_opts = {"font": ("Arial", 10, "bold"), "fg": "white", "bd": 0}

        # Button 1: Recognize
        self.btn_rec = tk.Button(btn_frame, text="Recognize", bg="#28a745", command=self.do_recognition, **btn_opts)
        self.btn_rec.pack(fill=tk.X, pady=4, ipady=4)

        # Button 2: Sci-Pop
        self.btn_info = tk.Button(btn_frame, text="Sci-Pop (Info)", bg="#fd7e14", command=self.call_rag_server,
                                  **btn_opts)
        self.btn_info.pack(fill=tk.X, pady=4, ipady=4)

        # Button 3: Save Image
        self.btn_save = tk.Button(btn_frame, text="Save Image", bg="#17a2b8", command=self.save_image, **btn_opts)
        self.btn_save.pack(fill=tk.X, pady=4, ipady=4)

        # Button 4: Exit
        self.btn_exit = tk.Button(btn_frame, text="Exit", bg="#6c757d", command=self.on_exit, **btn_opts)
        self.btn_exit.pack(side=tk.BOTTOM, fill=tk.X, pady=4, ipady=4)

        # === Bottom: Report ===
        bottom_frame = tk.LabelFrame(self.root, text=" AI Knowledge Report ", bg=self.bg_color, fg="#ff6666",
                                     font=("Arial", 10, "bold"))
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        bottom_frame.pack_propagate(False)
        bottom_frame.configure(height=125)

        self.txt_report = scrolledtext.ScrolledText(bottom_frame, bg="#2b2b2b", fg="#e0e0e0",
                                                    font=("Microsoft YaHei", 9), bd=0)
        self.txt_report.pack(fill=tk.BOTH, expand=True)

    def _create_placeholder_image(self, text, color="blue"):
        img = Image.new('RGB', (310, 230), color=color if color != "gray" else "#202020")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill="white")
        return ImageTk.PhotoImage(img)

    def update_camera_feed(self):
        if not self.running: return

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Resize for Live Monitor
                frame_resized = cv2.resize(frame, (310, 230))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                self.tk_live_img = ImageTk.PhotoImage(pil_img)
                self.lbl_video.config(image=self.tk_live_img)

        self.root.after(30, self.update_camera_feed)

    def do_recognition(self):
        if self.detector is None:
            messagebox.showwarning("Wait", "Model is loading...")
            return
        if self.current_frame is None:
            self.log_to_report("Error: No camera feed.")
            return

        # 保存临时图用于推理
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, self.current_frame)
        self.log_to_report("Analyzing frame...")

        def task():
            try:
                # 推理
                res_img, label = self.detector.detect(temp_path)

                if res_img is not None:
                    # 将结果图保存到内存
                    self.last_result_img = res_img

                    # 更新 UI
                    res_img_resized = cv2.resize(res_img, (310, 230))
                    res_img_rgb = cv2.cvtColor(res_img_resized, cv2.COLOR_BGR2RGB)
                    pil_res = Image.fromarray(res_img_rgb)
                    self.tk_res_img = ImageTk.PhotoImage(pil_res)
                    self.lbl_result.config(image=self.tk_res_img)

                if label:
                    self.detected_label = label
                    # 可以在这里修改 Prompt 格式
                    self.rag_query = f"Introduce the morphology and habits of {label}."
                    msg = f"Detected: [{label}]\nPrompt ready."
                else:
                    self.detected_label = None
                    self.rag_query = ""
                    msg = "No object detected."

                self.root.after(0, lambda: self.log_to_report(msg))

            except Exception as e:
                # 【修复2】先转存为字符串变量
                err_msg = f"Detection Error: {e}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        threading.Thread(target=task, daemon=True).start()

    def call_rag_server(self):
        if not self.rag_query:
            messagebox.showinfo("Tip", "Click 'Recognize' first.")
            return

        server_url = self.server_entry.get().strip()
        self.log_to_report(f"Connecting to RAG...")

        def task():
            try:
                resp = requests.post(f"{server_url}/chat", json={"query": self.rag_query}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    ans = data.get("answer", "No answer")
                    sources = data.get("sources", [])

                    display = f"=== {self.detected_label} ===\n{ans}\n\n[Sources]:\n"
                    for src in sources:
                        display += f"- {src}\n"

                    self.root.after(0, lambda: self.log_to_report(display, clear=True))
                else:
                    # 【修复3】防止状态码错误导致 lambda 报错（虽然这里没有 e，但保持一致）
                    err_msg = f"Server Error: {resp.status_code}"
                    self.root.after(0, lambda: self.log_to_report(err_msg))
            except Exception as e:
                # 【修复4】这里就是之前报错的地方，先转存为字符串
                err_msg = f"Connection Error: {e}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        threading.Thread(target=task, daemon=True).start()

    def save_image(self):
        if self.last_result_img is None:
            messagebox.showinfo("Tip", "Please recognize an image first.")
            return

        save_dir = "captures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/result_{timestamp}.jpg"

        try:
            cv2.imwrite(filename, self.last_result_img)
            self.log_to_report(f"Image saved to:\n{filename}")
        except Exception as e:
            self.log_to_report(f"Save failed: {e}")

    def on_exit(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def log_to_report(self, text, clear=False):
        self.txt_report.configure(state='normal')
        if clear: self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"{text}\n")
        self.txt_report.see(tk.END)
        self.txt_report.configure(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = MacauEcoSystemApp(root)
    root.mainloop()
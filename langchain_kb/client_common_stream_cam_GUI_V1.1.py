import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import threading
import requests
import time
import os
import cv2
import numpy as np

# 导入推理脚本 (请确保 yolo_infer.py 在同目录下)
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

        # ❗初始化摄像头 (0通常是默认摄像头)
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

        # --- 【新增 UI】：RAG 开关 ---
        self.use_rag_var = tk.BooleanVar(value=True)  # 默认开启 RAG
        self.chk_rag = tk.Checkbutton(btn_frame, text="Use DB (RAG)", variable=self.use_rag_var,
                                      bg=self.bg_color, fg="white", selectcolor="#444",
                                      activebackground=self.bg_color, activeforeground="white")
        self.chk_rag.pack(fill=tk.X, pady=2)

        # Button 1: Recognize
        self.btn_rec = tk.Button(btn_frame, text="Recognize", bg="#28a745", command=self.do_recognition, **btn_opts)
        self.btn_rec.pack(fill=tk.X, pady=2, ipady=3)

        # Button 2: Sci-Pop (Info)
        self.btn_info = tk.Button(btn_frame, text="Ask AI", bg="#fd7e14", command=self.call_rag_server, **btn_opts)
        self.btn_info.pack(fill=tk.X, pady=2, ipady=3)

        # --- 【新增 UI】：重建知识库按钮 ---
        self.btn_rebuild = tk.Button(btn_frame, text="Rebuild DB", bg="#6f42c1", command=self.rebuild_kb, **btn_opts)
        self.btn_rebuild.pack(fill=tk.X, pady=2, ipady=3)

        # Button 3: Save Image
        self.btn_save = tk.Button(btn_frame, text="Save Image", bg="#17a2b8", command=self.save_image, **btn_opts)
        self.btn_save.pack(fill=tk.X, pady=2, ipady=3)

        # Button 4: Exit
        self.btn_exit = tk.Button(btn_frame, text="Exit", bg="#6c757d", command=self.on_exit, **btn_opts)
        self.btn_exit.pack(side=tk.BOTTOM, fill=tk.X, pady=2, ipady=3)

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

        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, self.current_frame)
        self.log_to_report("Analyzing frame...")

        def task():
            try:
                res_img, label = self.detector.detect(temp_path)

                if res_img is not None:
                    self.last_result_img = res_img
                    res_img_resized = cv2.resize(res_img, (310, 230))
                    res_img_rgb = cv2.cvtColor(res_img_resized, cv2.COLOR_BGR2RGB)
                    pil_res = Image.fromarray(res_img_rgb)
                    self.tk_res_img = ImageTk.PhotoImage(pil_res)
                    self.lbl_result.config(image=self.tk_res_img)

                if label:
                    self.detected_label = label
                    self.rag_query = f"Introduce the {label}."
                    msg = f"Detected: [{label}]\nPrompt ready."
                else:
                    self.detected_label = None
                    self.rag_query = ""
                    msg = "No object detected."

                self.root.after(0, lambda: self.log_to_report(msg))

            except Exception as e:
                err_msg = f"Detection Error: {e}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        threading.Thread(target=task, daemon=True).start()

    def call_rag_server(self):
        if not self.rag_query:
            messagebox.showinfo("Tip", "Click 'Recognize' first.")
            return

        server_url = self.server_entry.get().strip()
        use_rag = self.use_rag_var.get()  # 获取 Checkbox 的状态

        mode_str = "RAG Mode" if use_rag else "Pure LLM Mode"
        self.log_to_report(f"Connecting to AI ({mode_str})...", clear=True)

        def task():
            try:
                start_time = time.time()

                # --- 【修改代码】：把 use_rag 状态发给服务器 ---
                payload = {
                    "query": self.rag_query,
                    "use_rag": use_rag
                }

                resp = requests.post(
                    f"{server_url}/chat",
                    json=payload,
                    timeout=60,
                    stream=True
                )

                if resp.status_code == 200:
                    self.root.after(0, lambda: self.log_to_report(f"=== {self.detected_label} ===\n"))
                    first_token_received = False

                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            if not first_token_received:
                                ttft = time.time() - start_time
                                self.root.after(0, lambda t=ttft: self.append_to_report(f" [TTFT: {t:.3f}s]\n"))
                                first_token_received = True
                            self.root.after(0, lambda c=chunk: self.append_to_report(c))

                    total_time = time.time() - start_time
                    self.root.after(0, lambda t=total_time: self.append_to_report(f"\n\n[Done. Total Time: {t:.3f}s]"))

                else:
                    err_msg = f"Server Error: {resp.status_code}"
                    self.root.after(0, lambda: self.log_to_report(err_msg))
            except Exception as e:
                err_msg = f"Connection Error: {e}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        threading.Thread(target=task, daemon=True).start()

    # --- 【新增功能】：调用服务器重建知识库 ---
    def rebuild_kb(self):
        server_url = self.server_entry.get().strip()
        self.log_to_report("Requesting server to rebuild Knowledge Base...\nPlease wait...", clear=True)

        # 暂时禁用按钮防误触
        self.btn_rebuild.config(state=tk.DISABLED)

        def task():
            try:
                # 触发 rebuild，这里我们传空字典，让服务器扫描默认 ./my_docs 文件夹
                # timeout 设长一点，因为重新切分和向量化需要时间
                resp = requests.post(f"{server_url}/rebuild", json={}, timeout=120)

                if resp.status_code == 200:
                    data = resp.json()
                    msg = f"Success: {data.get('message', 'DB Rebuilt.')}"
                else:
                    msg = f"Failed to rebuild (Code {resp.status_code}):\n{resp.text}"
            except Exception as e:
                msg = f"Connection Error during rebuild:\n{e}"

            # 恢复 UI 状态
            self.root.after(0, lambda: self.log_to_report(msg))
            self.root.after(0, lambda: self.btn_rebuild.config(state=tk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

    def append_to_report(self, text):
        self.txt_report.configure(state='normal')
        self.txt_report.insert(tk.END, text)
        self.txt_report.see(tk.END)
        self.txt_report.configure(state='disabled')

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
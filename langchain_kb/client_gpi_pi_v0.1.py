#client, 识别功能仅作展示
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import requests
import time

# ================= 配置 =================
DEFAULT_SERVER = "http://127.0.0.1:8001"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480

# 模拟识别到的鸟类名称（实际项目中这里由 YOLO/AI 模型提供）
DETECTED_OBJECT = "Black-faced Spoonbill"
# 用于发送给 RAG 的查询语句
RAG_QUERY = "Please introduce the morphological characteristics and habits of the Black-faced Spoonbill in Macau."


class MacauEcoSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Macau Eco AI System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        # 设置深色背景
        self.bg_color = "#1e1e1e"
        self.root.configure(bg=self.bg_color)

        # 禁止调整窗口大小（适应嵌入式屏幕）
        self.root.resizable(False, False)

        # 初始化 UI
        self._init_ui()

        # 启动模拟摄像头线程
        self.running = True
        self.update_camera_feed()

    def _init_ui(self):
        # === 1. 顶部栏：服务器地址设置 ===
        top_frame = tk.Frame(self.root, bg=self.bg_color, height=30)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Label(top_frame, text="Server:", bg=self.bg_color, fg="white", font=("Arial", 10)).pack(side=tk.LEFT)
        self.server_entry = tk.Entry(top_frame, bg="#333", fg="white", insertbackground="white", width=30)
        self.server_entry.pack(side=tk.LEFT, padx=5)
        self.server_entry.insert(0, DEFAULT_SERVER)

        tk.Label(top_frame, text="(e.g. http://192.168.1.x:8000)", bg=self.bg_color, fg="gray", font=("Arial", 8)).pack(
            side=tk.LEFT)

        # === 2. 中间区域：视频 + 结果 + 按钮 ===
        middle_frame = tk.Frame(self.root, bg=self.bg_color)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- A. 左侧：实时监控 (Live) ---
        # 绿色边框
        self.live_frame = tk.LabelFrame(middle_frame, text=" Live Monitor ", bg=self.bg_color, fg="#00ff00",
                                        font=("Arial", 10, "bold"), bd=2, relief="solid")
        self.live_frame.place(x=0, y=0, width=320, height=260)

        # 视频显示 Label
        self.lbl_video = tk.Label(self.live_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # --- B. 中间：识别结果 (Result) ---
        # 黄色边框
        self.result_frame = tk.LabelFrame(middle_frame, text=" Recognition Result ", bg=self.bg_color, fg="#ffcc00",
                                          font=("Arial", 10, "bold"), bd=2, relief="solid")
        self.result_frame.place(x=330, y=0, width=320, height=260)

        # 结果显示 Label
        self.lbl_result = tk.Label(self.result_frame, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 加载一张默认的空图片
        self.placeholder_img = self._create_placeholder_image("Waiting...", "gray")
        self.lbl_result.config(image=self.placeholder_img)

        # --- C. 右侧：操作按钮 (Control) ---
        btn_frame = tk.Frame(middle_frame, bg=self.bg_color)
        btn_frame.place(x=660, y=10, width=130, height=260)

        # 定义通用按钮样式
        btn_opts = {"font": ("Arial", 11, "bold"), "fg": "white", "bd": 0, "activeforeground": "white"}

        # 1. 识别 (Recognize) - 绿色
        self.btn_rec = tk.Button(btn_frame, text="Recognize", bg="#28a745", activebackground="#218838",
                                 command=self.do_recognition, **btn_opts)
        self.btn_rec.pack(fill=tk.X, pady=5, ipady=5)

        # 2. 科普 (Sci-Pop / Info) - 橙色 (核心 RAG 功能)
        self.btn_info = tk.Button(btn_frame, text="Sci-Pop (Info)", bg="#fd7e14", activebackground="#e36209",
                                  command=self.call_rag_server, **btn_opts)
        self.btn_info.pack(fill=tk.X, pady=5, ipady=5)

        # 3. 保存 (Save) - 蓝色
        self.btn_save = tk.Button(btn_frame, text="Save Image", bg="#17a2b8", activebackground="#138496",
                                  command=self.save_image, **btn_opts)
        self.btn_save.pack(fill=tk.X, pady=5, ipady=5)

        # 4. 退出 (Exit) - 灰色
        self.btn_exit = tk.Button(btn_frame, text="Exit", bg="#6c757d", activebackground="#5a6268",
                                  command=self.on_exit, **btn_opts)
        self.btn_exit.pack(side=tk.BOTTOM, fill=tk.X, pady=5, ipady=5)

        # === 3. 底部区域：AI 报告 ===
        # 红色/粉色边框
        bottom_frame = tk.LabelFrame(self.root, text=" AI Knowledge Report ", bg=self.bg_color, fg="#ff6666",
                                     font=("Arial", 10, "bold"), bd=2, relief="solid")
        # 修改点 1: fill=tk.X (只横向填充)，去掉 expand=True，防止它抢占中间高度
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # 修改点 2: 必须保留这行，禁止 Frame 根据内容自动缩放
        bottom_frame.pack_propagate(False)

        # 修改点 3: 将高度从 170 改为 110 (或者 100-120 之间)，给中间留出更多空间
        bottom_frame.configure(height=150)

        self.txt_report = scrolledtext.ScrolledText(bottom_frame, bg="#2b2b2b", fg="#e0e0e0",
                                                    font=("Microsoft YaHei", 9), bd=0, padx=5, pady=5)
        self.txt_report.pack(fill=tk.BOTH, expand=True)
        self.txt_report.insert(tk.END, "System Ready. Point camera at target and click 'Recognize' -> 'Sci-Pop'.")
        self.txt_report.configure(state='disabled')

    # ================= 逻辑处理 =================

    def _create_placeholder_image(self, text, color="blue"):
        """创建一个纯色带文字的图片，用于占位"""
        w, h = 310, 230  # 稍微小于 Frame
        img = Image.new('RGB', (w, h), color=color if color != "gray" else "#202020")
        draw = ImageDraw.Draw(img)
        # 简单绘制文字，如果要在树莓派显示中文可能需要加载 ttf 字体
        try:
            # 尝试画一个矩形框模拟检测框
            if color != "gray":
                draw.rectangle([100, 50, 210, 180], outline="#00ff00", width=3)

            draw.text((10, 10), text, fill="white")
        except:
            pass
        return ImageTk.PhotoImage(img)

    def update_camera_feed(self):
        """模拟摄像头实时画面更新"""
        if not self.running: return

        # 这里我们用时间戳生成动态图片模拟视频流
        current_time = time.strftime("%H:%M:%S")
        # 实际项目中，这里应该替换为 cv2.read() 获取的帧

        # 创建模拟的 Live 画面 (深蓝色背景)
        img = Image.new('RGB', (310, 230), color="#001f3f")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"LIVE CAM: {current_time}", fill="white")

        self.tk_live_img = ImageTk.PhotoImage(img)
        self.lbl_video.config(image=self.tk_live_img)

        # 每 100ms 刷新一次
        self.root.after(100, self.update_camera_feed)

    def do_recognition(self):
        """点击识别按钮"""
        self.log_to_report(f"Processing image... Detected: [{DETECTED_OBJECT}]")

        # 模拟：更新结果区域的图片
        # 实际项目中，这里显示 YOLO 推理后的带框图片
        self.tk_res_img = self._create_placeholder_image(f"Result: {DETECTED_OBJECT}", "#333")
        self.lbl_result.config(image=self.tk_res_img)

    def call_rag_server(self):
        """核心：调用 RAG 服务器获取科普信息"""
        server_url = self.server_entry.get().strip()
        if not server_url:
            messagebox.showerror("Error", "Please enter server address!")
            return

        self.log_to_report(f"Connecting to RAG Server ({server_url})...")

        def task():
            try:
                api_url = f"{server_url}/chat"
                # 发送硬编码的查询，实际中可以根据 Detected Object 动态生成
                payload = {"query": RAG_QUERY}

                # 发送请求
                response = requests.post(api_url, json=payload, timeout=20)  # 树莓派推理可能慢，超时设长点

                if response.status_code == 200:
                    data = response.json()
                    ans = data.get("answer", "No answer")
                    sources = data.get("sources", [])

                    # 格式化输出
                    display_text = f"=== AI Description ({DETECTED_OBJECT}) ===\n{ans}\n\n=== Sources ===\n"
                    for src in sources:
                        display_text += f"- {src}\n"

                    self.root.after(0, lambda: self.log_to_report(display_text, clear=True))
                else:
                    err_msg = f"Server Error: {response.status_code}"
                    self.root.after(0, lambda: self.log_to_report(err_msg))

            except Exception as e:
                err_msg = f"Connection Failed: {str(e)}"
                self.root.after(0, lambda: self.log_to_report(err_msg))

        # 在后台线程运行，避免卡死界面
        threading.Thread(target=task, daemon=True).start()

    def save_image(self):
        self.log_to_report("Image saved to ./captures/ (Simulated)")

    def on_exit(self):
        self.running = False
        self.root.destroy()

    def log_to_report(self, text, clear=False):
        """向底部文本框写入信息"""
        self.txt_report.configure(state='normal')
        if clear:
            self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"{text}\n")
        self.txt_report.see(tk.END)
        self.txt_report.configure(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = MacauEcoSystemApp(root)
    root.mainloop()
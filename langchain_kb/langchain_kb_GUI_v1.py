# 如何使用这个 GUI 程序
# 运行程序：保存代码为 gui_rag.py，然后在命令行运行 python gui_rag.py。
# 设置 API：
    # 左上角已经预填了你的 SiliconFlow Key 和 URL（如果不正确请手动修改）。
    # LLM 模型：预填了 Qwen/Qwen3-8B（速度快）。
    # Embed 模型：预填了 BAAI/bge-m3（效果好）。
# 添加资料：
    # 点击 “添加 PDF” 选择你本地的 20200601_DPAA_PAEM_tc.pdf 等文件。
    # 点击 “添加 URL” 粘贴百度百科的链接。
# 构建知识库：
    # 点击蓝色大按钮 “构建/重建 知识库”。
    # 观察下方状态文字，等待进度完成。
    # 注意：因为有 200 多个片段，可能需要 10-30 秒。
# 开始聊天：
    # 在右侧输入框输入 “澳门的黑脸琵鹭数量”，按回车或点击发送。
    # 等待 AI 回答，回答结束后下方会显示红色的 [参考资料] 列表。

import os
# 设置 User-Agent 伪装浏览器
os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
# [关键修改1] 使用标准库的 ScrolledText，解决 state 属性报错问题
from tkinter.scrolledtext import ScrolledText
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 设置 User-Agent
os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class RAGLogic:
    """处理 RAG 核心逻辑的类"""

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.api_key = ""
        self.base_url = ""
        self.llm_model = ""
        self.embed_model = ""

    def load_and_vectorize(self, pdf_paths, urls, api_key, base_url, llm_model, embed_model, log_callback):
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = llm_model
        self.embed_model = embed_model

        raw_docs = []

        # 1. 加载 PDF
        for path in pdf_paths:
            log_callback(f"正在读取 PDF: {os.path.basename(path)}...")
            try:
                loader = PyPDFLoader(path)
                raw_docs.extend(loader.load())
            except Exception as e:
                log_callback(f"错误: 读取 {path} 失败: {str(e)}", "error")

        # 2. 加载 URL
        if urls:
            log_callback(f"正在读取 {len(urls)} 个网页...")
            try:
                loader = WebBaseLoader(urls)
                raw_docs.extend(loader.load())
            except Exception as e:
                log_callback(f"错误: 网页读取失败: {str(e)}", "error")

        if not raw_docs:
            raise ValueError("没有加载到任何有效数据")

        # 3. 切割
        log_callback("正在切割文档...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(raw_docs)
        log_callback(f"共切分为 {len(splits)} 个片段。")

        # 4. 向量化
        log_callback("正在生成向量库 (API交互中)...")
        embeddings = OpenAIEmbeddings(
            model=self.embed_model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            check_embedding_ctx_length=False,
            chunk_size=32  # 防止 413 错误
        )

        self.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # 5. 初始化链
        log_callback("正在初始化 LLM...")
        llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=0.1
        )
        #"""请基于下面的【context】回答用户问题。如果资料中没有提到，请回答“资料中未提及”
        template = """你是一个科普助手，结合【context】回答用户问题。”。

        【context】：
        {context}

        【用户问题】：
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.chain = prompt | llm | StrOutputParser()
        log_callback("知识库构建完成！", "success")

    def query(self, question):
        if not self.retriever:
            raise ValueError("请先构建知识库")

        retrieved_docs = self.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        response_text = self.chain.invoke({"context": context_text, "question": question})
        return response_text, retrieved_docs


class RAGApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="litera")
        self.title("本地知识库助手 (LangChain + SiliconFlow)")
        self.geometry("1000x700")

        self.engine = RAGLogic()
        self.pdf_list = []
        self.url_list = []

        self._init_ui()

    def _init_ui(self):
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        left_panel = ttk.Frame(self, padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew")
        right_panel = ttk.Frame(self, padding=10)
        right_panel.grid(row=0, column=1, sticky="nsew")

        # === 左侧设置 ===
        lf_config = ttk.Labelframe(left_panel, text="API 设置", padding=10)
        lf_config.pack(fill=X, pady=5)

        ttk.Label(lf_config, text="API Key:").pack(anchor=W)
        self.entry_key = ttk.Entry(lf_config, show="*")
        self.entry_key.pack(fill=X, pady=(0, 5))
        self.entry_key.insert(0, "sk-nzxutftishvwoayxihtceizdtdbwmomewcoofwhyritpcjaz")

        ttk.Label(lf_config, text="Base URL:").pack(anchor=W)
        self.entry_url = ttk.Entry(lf_config)
        self.entry_url.pack(fill=X, pady=(0, 5))
        self.entry_url.insert(0, "https://api.siliconflow.cn/v1")

        ttk.Label(lf_config, text="LLM 模型:").pack(anchor=W)
        self.entry_model = ttk.Entry(lf_config)
        self.entry_model.pack(fill=X, pady=(0, 5))
        self.entry_model.insert(0, "Qwen/Qwen3-8B")

        ttk.Label(lf_config, text="Embed 模型:").pack(anchor=W)
        self.entry_embed = ttk.Entry(lf_config)
        self.entry_embed.pack(fill=X, pady=(0, 5))
        self.entry_embed.insert(0, "BAAI/bge-m3")

        # === 知识库列表 ===
        lf_kb = ttk.Labelframe(left_panel, text="知识库文件", padding=10)
        lf_kb.pack(fill=BOTH, expand=YES, pady=5)

        self.lst_files = tk.Listbox(lf_kb, height=10, selectmode=EXTENDED)
        self.lst_files.pack(fill=BOTH, expand=YES, pady=5)

        btn_frame = ttk.Frame(lf_kb)
        btn_frame.pack(fill=X)
        ttk.Button(btn_frame, text="添加 PDF", command=self.add_pdf, bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(btn_frame, text="添加 URL", command=self.add_url, bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(btn_frame, text="清空", command=self.clear_files, bootstyle="secondary-outline").pack(side=RIGHT,
                                                                                                         padx=2)

        self.btn_build = ttk.Button(left_panel, text="构建/重建 知识库", command=self.start_build_kb,
                                    bootstyle="primary")
        self.btn_build.pack(fill=X, pady=10)

        self.lbl_status = ttk.Label(left_panel, text="就绪", bootstyle="secondary", wraplength=200)
        self.lbl_status.pack(side=BOTTOM, fill=X)

        # === 右侧聊天 ===
        # [关键修改2] 使用 padx/pady 替代 padding，修复参数错误
        self.chat_display = ScrolledText(right_panel, padx=10, pady=10, font=("Microsoft YaHei", 10))
        self.chat_display.pack(fill=BOTH, expand=YES)
        self.chat_display.configure(state=DISABLED)

        self.chat_display.tag_config("user", foreground="#007bff", font=("Microsoft YaHei", 11, "bold"))
        self.chat_display.tag_config("ai", foreground="#28a745")
        self.chat_display.tag_config("sys", foreground="gray", font=("Consolas", 9))
        self.chat_display.tag_config("source", foreground="#dc3545", font=("Microsoft YaHei", 8))

        input_frame = ttk.Frame(right_panel, padding=(0, 10, 0, 0))
        input_frame.pack(fill=X)

        self.txt_input = ttk.Entry(input_frame, font=("Microsoft YaHei", 10))
        self.txt_input.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        self.txt_input.bind("<Return>", lambda event: self.start_chat())

        self.btn_send = ttk.Button(input_frame, text="发送", command=self.start_chat, bootstyle="success")
        self.btn_send.pack(side=RIGHT)

    # ... 后续逻辑保持不变 ...

    def log(self, message, tag="sys"):
        self.chat_display.configure(state=NORMAL)
        self.chat_display.insert(END, f"[{tag.upper()}] {message}\n", tag)
        self.chat_display.see(END)
        self.chat_display.configure(state=DISABLED)
        self.lbl_status.config(text=message)

    def append_chat(self, role, text):
        self.chat_display.configure(state=NORMAL)
        tag = "user" if role == "我" else "ai"
        self.chat_display.insert(END, f"\n{role}: \n", tag)
        self.chat_display.insert(END, f"{text}\n", "text")
        self.chat_display.see(END)
        self.chat_display.configure(state=DISABLED)

    def append_sources(self, docs):
        self.chat_display.configure(state=NORMAL)
        self.chat_display.insert(END, "\n--- 参考资料 ---\n", "source")
        seen = set()
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", None)
            info = f"{source}"
            if page is not None: info += f" (P{page + 1})"
            if info not in seen:
                self.chat_display.insert(END, f"[{i + 1}] {info}\n", "source")
                seen.add(info)
        self.chat_display.insert(END, "----------------\n", "source")
        self.chat_display.see(END)
        self.chat_display.configure(state=DISABLED)

    def add_pdf(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        for f in files:
            if f not in self.pdf_list:
                self.pdf_list.append(f)
                self.lst_files.insert(END, f"[PDF] {os.path.basename(f)}")

    def add_url(self):
        top = ttk.Toplevel(self)
        top.title("输入 URL")
        ttk.Label(top, text="网址:").pack(padx=10, pady=5)
        entry = ttk.Entry(top, width=50)
        entry.pack(padx=10, pady=5)

        def confirm():
            url = entry.get().strip()
            if url:
                self.url_list.append(url)
                self.lst_files.insert(END, f"[WEB] {url}")
                top.destroy()

        ttk.Button(top, text="确定", command=confirm).pack(pady=10)

    def clear_files(self):
        self.pdf_list.clear()
        self.url_list.clear()
        self.lst_files.delete(0, END)

    def start_build_kb(self):
        self.btn_build.config(state=DISABLED)
        self.log("开始构建知识库...", "sys")
        api_key = self.entry_key.get()
        base_url = self.entry_url.get()
        llm = self.entry_model.get()
        embed = self.entry_embed.get()

        def task():
            try:
                self.engine.load_and_vectorize(self.pdf_list, self.url_list, api_key, base_url, llm, embed, self.log)
            except Exception as e:
                self.log(f"构建失败: {e}", "error")
                messagebox.showerror("错误", str(e))
            finally:
                self.btn_build.config(state=NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def start_chat(self):
        question = self.txt_input.get().strip()
        if not question: return
        if not self.engine.chain:
            messagebox.showwarning("提示", "请先构建知识库！")
            return
        self.txt_input.delete(0, END)
        self.append_chat("我", question)
        self.btn_send.config(state=DISABLED)

        def task():
            try:
                response, docs = self.engine.query(question)
                self.append_chat("AI", response)
                self.append_sources(docs)
            except Exception as e:
                self.log(f"回答出错: {e}", "error")
            finally:
                self.btn_send.config(state=NORMAL)

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()
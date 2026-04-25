# server-stream for the client
#by huangxy
import os
import shutil
import glob
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse  # 新增：用于流式输出
from pydantic import BaseModel
import uvicorn

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # 更换为 FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 设置 User-Agent
os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ================= 配置 =================
API_KEY = "sk-nzxutftishvwoayxihtceizdtdbwmomewcoofwhyritpcjaz"
BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "./faiss_db"  # 修改路径名


# ================= 数据模型 =================
class ChatRequest(BaseModel):
    query: str


class RebuildRequest(BaseModel):
    pdf_files: Optional[List[str]] = None
    urls: Optional[List[str]] = None


# ================= RAG 核心引擎类 =================
class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None

        # 0.6B 模型较快，适合 RISC-V
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",  # 建议尝试 SiliconFlow 上的 bge-m3，兼容性好且轻量
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            chunk_size = 32  # 这里是正确的
        )

        # 启用 streaming=True
        self.llm = ChatOpenAI(
            model="Qwen/Qwen3-8B",  # 或者 Qwen/Qwen2.5-7B-Instruct
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1,
            streaming=True,
            #chunk_size=32  # <--- 必须添加这一行，建议设置为 64 或更小（如 32）
        )
        #Context: {context}
        self.prompt = ChatPromptTemplate.from_template("""
        You are a professional ecology expert. Please answer the user's question (in English) by combining your own knowledge and the Context provided below.
        Answer Guidelines:      
        1. Prioritize popularizing knowledge (Morphology, Habits, Distribution, present situation, Ecological Role).       
        2. If the Context contain more specific and up-to-date details about Macao (such as specific observation data, locations, etc.), please integrate them into your answer.      
        3. If the Context conflict with your understanding, please follow the Context, as they may be more relevant local documents.       
        4. Your answer should be vivid, interesting, and suitable for popularizing science to the general public.
        [Instruction]: 
        1. Do not use any Markdown formatting.
        2. Titles can be numbered (1., 2., etc.).
        Context: {context}
        Question: {question}
        """)

    def initialize(self):
        """加载或初始化 FAISS"""
        if os.path.exists(DB_PATH):
            print(">>> 正在从本地加载 FAISS 索引...")
            try:
                self.vectorstore = FAISS.load_local(
                    DB_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
                self._setup_retriever()
                print(">>> FAISS 加载成功！")
            except Exception as e:
                print(f">>> 加载失败，准备重建: {e}")
                self.auto_build()
        else:
            self.auto_build()

    def auto_build(self):
        default_pdfs = glob.glob("./my_docs/*.pdf")
        default_urls = ["https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612",
                        "https://avibase.bsc-eoc.org/species.jsp?lang=EN&avibaseid=DFD1DDFF11A7DE43"]
        self.build_kb(default_pdfs, default_urls)

    def build_kb(self, pdf_files: List[str], urls: List[str]):
        """构建/重建知识库 - 手动分批安全版"""
        print(">>> 开始构建知识库 (FAISS)...")

        # 1. 加载数据
        docs = []
        for pdf_path in pdf_files:
            if os.path.exists(pdf_path):
                print(f"    读取 PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"    Error: {e}")

        if urls:
            print(f"    读取 {len(urls)} 个网页...")
            try:
                loader = WebBaseLoader(urls)
                docs.extend(loader.load())
            except Exception as e:
                print(f"    Error: {e}")

        if not docs:
            raise ValueError("没有加载到有效数据")

        # 2. 切割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        total_splits = len(splits)
        print(f"    共切分为 {total_splits} 个片段，准备开始分批向量化...")

        # 3. 【核心修复】手动分批构建 FAISS
        # SiliconFlow 限制 64，我们每批只传 30，确保绝对安全
        batch_size = 30

        # --- 处理第一批以初始化数据库 ---
        first_batch = splits[:batch_size]
        print(f"    正在初始化索引 (1-{len(first_batch)})...")
        self.vectorstore = FAISS.from_documents(first_batch, self.embeddings)

        # --- 循环处理剩余批次 ---
        for i in range(batch_size, total_splits, batch_size):
            batch = splits[i: i + batch_size]
            current_end = min(i + batch_size, total_splits)
            print(f"    正在添加批次 ({i + 1}-{current_end} / 总计 {total_splits})...")

            # 手动调用 add_documents 而不是 from_documents
            self.vectorstore.add_documents(batch)

            # 给 API 和 RISC-V 0.5秒的缓冲时间
            time.sleep(0.5)

        # 4. 保存到磁盘
        self.vectorstore.save_local(DB_PATH)
        self._setup_retriever()
        print(">>> 知识库构建并保存成功！")

    def _setup_retriever(self):
        # k=3 减少上下文长度，加快 RISC-V 处理速度
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

    async def stream_chat(self, query: str):
        """流式回答逻辑 - 根据 Prompt 自动判断是否使用 RAG"""
        if not self.vectorstore:
            yield "Error: Knowledge base not initialized."
            return

        # 1. 【核心判断】检查 Prompt 模板中是否要求传入 context 变量
        # LangChain 的模板对象可以通过 input_variables 获取它需要的变量列表
        use_rag = "context" in self.prompt.input_variables

        context_text = ""
        sources = []

        # 2. 如果需要 RAG，则执行检索逻辑
        if use_rag:
            print(f">>> 检测到 Prompt 包含 {{context}}，启动知识库检索...")
            retrieved_docs = self.retriever.invoke(query)
            context_text = "\n\n".join(d.page_content for d in retrieved_docs)

            # 提取来源
            for doc in retrieved_docs:
                src = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", None)
                if page is not None: src += f" (P{page + 1})"
                sources.append(src)
            sources = list(dict.fromkeys(sources))
        else:
            print(f">>> Prompt 不包含 {{context}}，将直接调用模型知识...")

        # 3. 构造 Chain 并输出回答
        chain = self.prompt | self.llm | StrOutputParser()

        # 构造输入参数字典
        input_data = {"question": query}
        if use_rag:
            input_data["context"] = context_text

        async for chunk in chain.astream(input_data):
            yield chunk

        # 4. 【展示判断】根据是否使用了 RAG 输出不同的来源说明
        if use_rag:
            if sources:
                yield "\n\n[Sources]:\n"
                for s in sources:
                    yield f"- {s}\n"
            else:
                yield "\n\n[Sources]: No relevant local documents found."
        else:
            # 如果没用 RAG，明确告知用户
            yield "\n\n[Source]: General Knowledge (Internal Model Knowledge)"


# ================= FastAPI App =================
app = FastAPI()
engine = RAGEngine()


@app.on_event("startup")
async def startup():
    if not os.path.exists("./my_docs"): os.makedirs("./my_docs")
    engine.initialize()


@app.post("/chat")
async def chat(request: ChatRequest):
    # 使用 StreamingResponse 替代普通返回，彻底解决 30s 超时问题
    return StreamingResponse(engine.stream_chat(request.query), media_type="text/event-stream")



@app.post("/rebuild")
async def rebuild(request: RebuildRequest):
    local_pdfs = glob.glob("./my_docs/*.pdf")
    pdfs = request.pdf_files if request.pdf_files else local_pdfs
    urls = request.urls if request.urls else []
    engine.build_kb(pdfs, urls)
    return {"status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
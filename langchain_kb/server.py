import os
import shutil
import glob
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

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

# ================= 配置 =================
#siliconflow
API_KEY = "sk-xxxxxxxxxx"
BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "./chroma_db"  # 向量数据库持久化路径，/chroma_db使用Embedding模型不一样，需要对数据库进行重建chrome


# ================= 数据模型 (Pydantic) =================
class ChatRequest(BaseModel):
    query: str


class RebuildRequest(BaseModel):
    # 允许客户端动态传入文件列表，如果为空则使用默认配置
    pdf_files: Optional[List[str]] = None
    urls: Optional[List[str]] = None


# ================= RAG 核心引擎类 =================
class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.chain = None

        # 初始化模型配置，
        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-0.6B",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            check_embedding_ctx_length=False,
            chunk_size=32  # 关键修复：避免 413 错误
        )

        self.llm = ChatOpenAI(
            model="Qwen/Qwen3-8B",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are a professional ecology expert. Please answer the user's question (in English) by combining your own knowledge and the references provided below.

        [References]:       
        {context}
        
        [User Question]:        
        {question}
        
        """)

    def initialize(self):
        """启动时尝试加载现有的数据库"""
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            print(">>> 检测到本地向量库，正在加载...")
            self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
            self._setup_chain()
            print(">>> 知识库加载完成！")
        else:
            print(">>> 本地向量库不存在，将使用默认数据构建...")
            # 默认数据配置
            default_pdfs = ["./my_docs/2023_tc.pdf"]  # 请确保文件存在，否则注释掉
            default_urls = ["https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612"]
            self.build_kb(default_pdfs, default_urls)

    def build_kb(self, pdf_files: List[str], urls: List[str]):
        """构建/重建知识库"""
        print(">>> 开始构建知识库...")

        # 1. 清理旧数据库
        if os.path.exists(DB_PATH):
            print("    清理旧数据库文件...")
            # Chroma 在运行时可能会锁定文件，强制删除可能会报错，这里做简单处理
            self.vectorstore = None  # 释放引用
            try:
                shutil.rmtree(DB_PATH)
            except Exception as e:
                print(f"    警告: 删除旧库失败 (可能正被占用): {e}")

        # 2. 加载数据
        docs = []
        # PDF
        for pdf_path in pdf_files:
            if os.path.exists(pdf_path):
                print(f"    读取 PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"    Error: {e}")

        # URL
        if urls:
            print(f"    读取 {len(urls)} 个网页...")
            try:
                loader = WebBaseLoader(urls)
                docs.extend(loader.load())
            except Exception as e:
                print(f"    Error: {e}")

        if not docs:
            raise ValueError("没有加载到有效数据，无法构建知识库")

        # 3. 切割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        print(f"    共切分为 {len(splits)} 个片段，正在向量化...")

        # 4. 向量化并持久化存储
        # 指定 persist_directory 后，Chroma 会自动保存到磁盘
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )

        self._setup_chain()
        print(">>> 知识库构建并保存成功！")

    def _setup_chain(self):
        """设置检索器和问答链"""
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chain = self.prompt | self.llm | StrOutputParser()

    # 修改 RAGEngine 中的 chat 函数，增加一个简单的逻辑判断
    def chat(self, query: str):
        if not self.vectorstore:
            raise HTTPException(status_code=500, detail="知识库未初始化")

        # 1. 检索（k值可以根据需要调整，k=5 通常足够了）
        retrieved_docs = self.retriever.invoke(query)

        # 检查是否真的找回了有用的内容
        if not retrieved_docs:
            context_text = "（未在本地文档中找到相关特定信息）"
        else:
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 2. 生成结果
        answer = self.chain.invoke({"context": context_text, "question": query})

        # 3. 整理来源
        sources = []
        for doc in retrieved_docs:
            src = os.path.basename(doc.metadata.get("source", "未知"))  # 只保留文件名，不显示长路径
            page = doc.metadata.get("page", None)
            if page is not None:
                src += f" (P{page + 1})"
            sources.append(src)

        sources = list(dict.fromkeys(sources))

        return {
            "answer": answer,
            "sources": sources
        }


# ================= FastAPI App =================
app = FastAPI(title="RAG Knowledge Base Server")
engine = RAGEngine()


# 在服务器启动时初始化
@app.on_event("startup")
async def startup_event():
    # 确保有一个 my_docs 文件夹防止报错
    if not os.path.exists("./my_docs"): os.makedirs("./my_docs")
    engine.initialize()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return engine.chat(request.query)


@app.post("/rebuild")
async def rebuild_endpoint(request: RebuildRequest):
    """
    重建知识库接口。
    """
    # 2. 【核心修改】自动扫描 ./my_docs 文件夹下所有的 .pdf 文件
    # 无论文件名叫什么，只要在文件夹里，都会被读取
    local_pdfs = glob.glob("./my_docs/*.pdf")

    # 如果客户端传了文件名列表，就用客户端的；否则用本地扫描到的列表
    pdfs = request.pdf_files if request.pdf_files is not None else local_pdfs

    # URL 还是以客户端传入的为主，如果没有传，就用默认的
    default_urls = ["https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612"]
    urls = request.urls if request.urls is not None else default_urls

    print(f">>> 准备重建: 扫描到 {len(pdfs)} 个本地 PDF 文件")

    try:
        engine.build_kb(pdfs, urls)
        return {"status": "success", "message": f"知识库已重建，包含 {len(pdfs)} 个文件和 {len(urls)} 个网页"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8001)

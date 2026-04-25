

import os
import shutil
import glob
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# LangChain 相关导入
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= 配置 =================
API_KEY = " "sk-xxxxxxxxxx"
BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "./faiss_db"


# ================= 数据模型 =================
class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True  # 开关：默认 True(使用 RAG)，传 False 则纯 LLM 测试


class RebuildRequest(BaseModel):
    pdf_files: Optional[List[str]] = None
    urls: Optional[List[str]] = None


# ================= RAG 核心引擎 =================
class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None

        # 初始化向量模型 (BGE-M3)
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            chunk_size=32
        )

        # 初始化大语言模型 (Qwen2.5-7B)
        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1,
            streaming=True,
        )

        # === 对比方案 A：纯 LLM Prompt (不包含 Context) ===
        self.prompt_llm_only = ChatPromptTemplate.from_template("""
        You are a professional ecology expert. Answer the user's question (in English) based strictly on your OWN pre-trained general knowledge.
        DO NOT use any external documents.

        1. Focus on Morphology, Habits, Distribution, and Ecological Role.
        2. Integrate specific Macao data if you have any in your pre-trained memory.
        3. No Markdown formatting. Use numbered lists (1., 2.) for titles.

        Question: {question}
        """)

        # === 对比方案 B：LLM + RAG Prompt (包含 Context) ===
        self.prompt_rag = ChatPromptTemplate.from_template("""
        You are a professional ecology expert. Answer the user's question (in English) based MAINLY on the following Context provided below.
        If the Context doesn't contain the exact answer, you can supplement with your general knowledge, but state clearly what comes from the context.

        1. Focus on Morphology, Habits, Distribution, and Ecological Role.
        2. No Markdown formatting. Use numbered lists (1., 2.) for titles.

        Context:
        {context}

        Question: {question}
        """)

    def initialize(self):
        """启动初始化"""
        if os.path.exists(os.path.join(DB_PATH, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(
                    DB_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
                self._setup_retriever()
                print(">>> 成功加载本地 FAISS 索引")
            except Exception as e:
                print(f"加载本地索引失败: {e}，正在重新构建...")
                self.auto_build_default()
        else:
            self.auto_build_default()

    def _setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def auto_build_default(self):
        """自动使用默认文件构建知识库"""
        pdfs = glob.glob("./my_docs/*.pdf")
        urls = ["https://avibase.bsc-eoc.org/species.jsp?lang=EN&avibaseid=DFD1DDFF11A7DE43"]
        self.build_kb(pdfs, urls)

    def build_kb(self, pdf_files: List[str], urls: List[str]):
        """核心知识库构建逻辑 (Rebuild)"""
        print(f">>> 开始构建知识库: PDF={len(pdf_files)}, URLs={len(urls)}")

        # 1. 加载文档
        docs = []
        for p in pdf_files:
            if os.path.exists(p):
                try:
                    docs.extend(PyPDFLoader(p).load())
                except Exception as e:
                    print(f"加载 PDF 失败 {p}: {e}")

        if urls:
            try:
                loader = WebBaseLoader(urls)
                docs.extend(loader.load())
            except Exception as e:
                print(f"加载 URL 失败: {e}")

        if not docs:
            print(">>> 错误: 没有有效文档，构建中止")
            return False

        # 2. 文本切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        print(f">>> 文档切分完成，共 {len(splits)} 个片段")

        # 3. 向量化 (分批处理应对 API 限制)
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)  # 清理旧库

        batch_size = 30
        try:
            # 首次初始化 FAISS
            self.vectorstore = FAISS.from_documents(splits[:batch_size], self.embeddings)

            # 分批追加
            for i in range(batch_size, len(splits), batch_size):
                self.vectorstore.add_documents(splits[i: i + batch_size])
                time.sleep(0.4)  # 避免触发 API 速率限制

            # 4. 保存到本地并更新检索器
            self.vectorstore.save_local(DB_PATH)
            self._setup_retriever()
            print(">>> 知识库重建并保存成功")
            return True
        except Exception as e:
            print(f">>> 向量化过程发生错误: {e}")
            return False

    async def stream_chat(self, query: str, use_rag: bool):
        """流式对话输出"""
        try:
            if use_rag:
                # ==========================
                # 模式：LLM + RAG (检索增强)
                # ==========================
                if not self.retriever:
                    yield "Database not initialized. Please rebuild the knowledge base first.\n"
                    return

                # 1. 检索相似文档片段
                retrieved_docs = await asyncio.to_thread(self.retriever.invoke, query)
                context = "\n\n".join(d.page_content for d in retrieved_docs)
                sources = list(set([os.path.basename(d.metadata.get("source", "Web")) for d in retrieved_docs]))

                # 2. 使用 RAG Prompt 将上下文传给大模型
                chain = self.prompt_rag | self.llm | StrOutputParser()
                yield "[Mode: LLM + RAG]\n\n"
                async for chunk in chain.astream({"context": context, "question": query}):
                    yield chunk

                if sources:
                    yield "\n\n[RAG Sources Used]: " + ", ".join(sources) + "\n"

            else:
                # ==========================
                # 模式：只用纯 LLM
                # ==========================
                # 不做检索，直接使用 LLM 专用的 Prompt
                chain = self.prompt_llm_only | self.llm | StrOutputParser()
                yield "[Mode: Pure LLM]\n\n"
                async for chunk in chain.astream({"question": query}):
                    yield chunk
                yield "\n\n[Note]: Answered purely from pre-trained knowledge, no database used.\n"

        except Exception as e:
            yield f"\n[Error]: {str(e)}\n"


# ================= FastAPI 应用设定 =================
app = FastAPI()
engine = RAGEngine()


@app.on_event("startup")
async def startup():
    if not os.path.exists("./my_docs"):
        os.makedirs("./my_docs")
    # 启动时初始化引擎（加载现有库或从默认文档生成）
    engine.initialize()


@app.post("/chat")
async def chat(request: ChatRequest):
    """对话接口，支持切换 use_rag"""
    return StreamingResponse(
        engine.stream_chat(request.query, request.use_rag),
        media_type="text/plain"
    )


@app.post("/rebuild")
async def rebuild(request: RebuildRequest):
    """
    重建知识库接口。
    可以通过 body 传入自定义的 pdf_files 或 urls。如果不传，则扫描默认文件夹。
    """
    # 如果请求没填文件，就扫描本地文件夹
    pdfs = request.pdf_files if request.pdf_files else glob.glob("./my_docs/*.pdf")
    urls = request.urls if request.urls else []

    if not pdfs and not urls:
        raise HTTPrException(status_code=400, detail="No PDFs or URLs provided/found.")

    # 开启线程执行构建，避免长时间阻塞 FastAPI 的主事件循环
    success = await asyncio.to_thread(engine.build_kb, pdfs, urls)

    if success:
        return {
            "status": "success",
            "message": f"Knowledge base rebuilt successfully with {len(pdfs)} PDFs and {len(urls)} URLs."
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to rebuild knowledge base. Check server logs.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

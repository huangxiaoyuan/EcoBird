# server.py
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
API_KEY = "sk-xxxxxxxxxx"
BASE_URL = "https://api.siliconflow.cn/v1"
DB_PATH = "./faiss_db"  # 修改路径名


# ================= 数据模型 =================
class ChatRequest(BaseModel):
    query: str


class RebuildRequest(BaseModel):
    pdf_files: Optional[List[str]] = None
    urls: Optional[List[str]] = None


# ================= RAG 核心引擎 =================
class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None

        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            chunk_size=32
        )

        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0.1,
            streaming=True,
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are a professional ecology expert. Answer the user's question (in English) based on your general knowledge (not any Context).
        1. Focus on Morphology, Habits, Distribution, and Ecological Role.
        2. Integrate specific Macao data if you have any in your pre-trained memory.
        3. No Markdown formatting. Use numbered lists (1., 2.) for titles.
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
                self.auto_build_default()
        else:
            self.auto_build_default()

    def _setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def auto_build_default(self):
        """自动使用默认文件构建"""
        pdfs = glob.glob("./my_docs/*.pdf")
        urls = ["https://avibase.bsc-eoc.org/species.jsp?lang=EN&avibaseid=DFD1DDFF11A7DE43"]
        self.build_kb(pdfs, urls)

    def build_kb(self, pdf_files: List[str], urls: List[str]):
        """核心构建逻辑"""
        print(f">>> 开始构建知识库: PDF={len(pdf_files)}, URLs={len(urls)}")

        # 1. 加载
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
            print(">>> 错误: 没有有效文档")
            return False

        # 2. 切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. 向量化 (分批处理应对 API 限制)
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)  # 清理旧库

        batch_size = 30
        self.vectorstore = FAISS.from_documents(splits[:batch_size], self.embeddings)

        for i in range(batch_size, len(splits), batch_size):
            self.vectorstore.add_documents(splits[i: i + batch_size])
            time.sleep(0.4)

        # 4. 保存
        self.vectorstore.save_local(DB_PATH)
        self._setup_retriever()
        print(">>> 知识库构建成功")
        return True

    async def stream_chat(self, query: str):
        if not self.retriever:
            yield "Database not initialized."
            return

        try:
            # 检索
            retrieved_docs = await asyncio.to_thread(self.retriever.invoke, query)
            context = "\n\n".join(d.page_content for d in retrieved_docs)
            sources = list(set([os.path.basename(d.metadata.get("source", "Web")) for d in retrieved_docs]))

            # 生成
            chain = self.prompt | self.llm | StrOutputParser()
            async for chunk in chain.astream({"context": context, "question": query}):
                yield chunk

            if sources:
                yield "\n\nSources: " + ", ".join(sources)
        except Exception as e:
            yield f"\n[Error]: {str(e)}"


# ================= FastAPI App =================
app = FastAPI()
engine = RAGEngine()


@app.on_event("startup")
async def startup():
    if not os.path.exists("./my_docs"): os.makedirs("./my_docs")
    engine.initialize()


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(engine.stream_chat(request.query), media_type="text/plain")


@app.post("/rebuild")
async def rebuild(request: RebuildRequest):
    # 如果请求没填文件，就扫描本地文件夹
    pdfs = request.pdf_files if request.pdf_files else glob.glob("./my_docs/*.pdf")
    urls = request.urls if request.urls else []

    # 开启线程执行构建，避免阻塞 FastAPI
    success = await asyncio.to_thread(engine.build_kb, pdfs, urls)

    if success:
        return {"status": "success", "message": f"Knowledge base rebuilt with {len(pdfs)} PDFs."}
    else:
        raise HTTPException(status_code=500, detail="Failed to rebuild knowledge base.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

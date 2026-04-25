import os
# 1. 设置 User-Agent 消除警告并伪装成浏览器
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # 使用 OpenAI 标准接口
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================= 1. 配置云端 API =================
# 使用你之前提供的 SiliconFlow Key
API_KEY = "xxxxxxxxxxxxxxxx"
BASE_URL = "https://api.siliconflow.cn/v1"

# ================= 2. 加载数据 (PDF + 网页) =================
print(">>> [1/5] 正在加载文档...")
docs = []

# --- A. 加载 PDF ---
pdf_path = "20200601_DPAA_PAEM_tc.pdf" # 替换为你的文件名
if os.path.exists(pdf_path):
    print(f"    正在读取 PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())
else:
    print(f"    (跳过) 未找到 PDF 文件: {pdf_path}")

# --- B. 加载 网页 ---
urls = ["https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612"]
print(f"    正在读取网页: {urls[0]}...")
try:
    loader = WebBaseLoader(urls)
    docs.extend(loader.load())
except Exception as e:
    print(f"    网页读取失败: {e}")

if not docs:
    print("!!! 错误: 未加载到任何数据，请检查文件或网络。")
    exit()

# ================= 3. 文档切割 =================
print(f">>> [2/5] 正在切割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 切分大小
    chunk_overlap=100  # 重叠部分
)
splits = text_splitter.split_documents(docs)
print(f"    共切分为 {len(splits)} 个片段。")

# ================= 4. 向量化 (Embeddings) =================
print(">>> [3/5] 正在调用云端 API 生成向量 (Embeddings)...")

# 【关键点】不使用 Ollama，而是使用云端 API 进行向量化
# SiliconFlow 提供了 BAAI/bge-m3 (很强的中文向量模型)
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False # 关闭长度检查以避免报错
)

# 创建向量库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ================= 5. 定义 LLM 和 Prompt =================
print(">>> [4/5] 初始化云端大模型...")

# 【关键点】使用 ChatOpenAI 连接云端模型
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B", # 使用 SiliconFlow 上的 DeepSeek 模型
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.1
)

template = """你是一个专业的科普助手。请基于下面的【参考资料】回答问题。如果资料中没有提到，请回答“资料中未提及”。

【参考资料】：
{context}

【用户问题】：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ================= 6. 构建并运行链 =================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ================= 7. 提问 =================
query = "澳门的黑脸琵鹭"
print(f"\n>>> [5/5] 用户提问: {query}")
print("-" * 30)

# 流式输出
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)
print("\n" + "-" * 30)
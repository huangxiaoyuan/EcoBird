#huangxy
#列表化输入:
#pdf_files = ["a.pdf", "b.pdf"]: 现在你可以把所有 PDF 文件名都放在这个列表里，代码会循环读取。
#url_list = [...]: 网址也是同理。WebBaseLoader([url1, url2]) 本身就支持一次性读入多个 URL。

import os

# 设置 User-Agent 伪装浏览器
os.environ[
    "USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= 1. 配置云端 API =================
API_KEY = "sk-nzxutftishvwoayxihtceizdtdbwmomewcoofwhyritpcjaz"
BASE_URL = "https://api.siliconflow.cn/v1"

# ================= 2. 加载多源数据 (列表模式) =================
print(">>> [1/5] 正在加载多源文档...")
docs = []

# --- A. 定义 PDF 列表 ---
# 你可以在这里填入多个文件名
pdf_files = [
    #"PM_PDF/20200601_DPAA_PAEM_tc.pdf",
    "./my_docs/2023_tc.pdf",  # 如果有更多文件，去掉注释加在这里
    #"PM_PDF/澳門環境保護規劃（2021-2025）202112-EnvPlanningBook_PB_TC.pdf"
]

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"    正在读取 PDF: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"    读取 {pdf_path} 失败: {e}")
    else:
        print(f"    (跳过) 未找到文件: {pdf_path}")

# --- B. 定义 网址 列表 ---
# 你可以在这里填入多个网址
url_list = [
    "https://baike.baidu.com/item/%E9%BB%91%E8%84%B8%E7%90%B5%E9%B9%AD/347612",
    "https://datazone.birdlife.org/species/factsheet/black-faced-spoonbill-platalea-minor", # 示例
]

if url_list:
    print(f"    正在读取 {len(url_list)} 个网页...")
    try:
        # WebBaseLoader 原生支持传入列表
        loader = WebBaseLoader(url_list)
        docs.extend(loader.load())
    except Exception as e:
        print(f"    网页读取部分失败: {e}")

if not docs:
    print("!!! 错误: 未加载到任何数据，请检查文件路径或网络。")
    exit()

# ================= 3. 文档切割 =================
print(f">>> [2/5] 正在切割文档...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"    共切分为 {len(splits)} 个片段。")

# ================= 4. 向量化 =================
print(">>> [3/5] 生成向量库...")
# 【修复点】增加 chunk_size=32
# 这告诉 LangChain 每次只发 32 个片段给服务器，避免触发 64 的限制
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=32  # <--- 关键修复：设置为小于 64 的数字 (例如 32 或 50)
)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
# 这里 k=4 表示检索最相关的 4 个片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ================= 5. 定义模型 =================
print(">>> [4/5] 初始化大模型...")
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.1
)

template = """你是一个科普专家，请结合下面的【参考资料】回答用户问题。如果资料中没有提到，请回答“资料中未提及”。

【参考资料】：
{context}

【用户问题】：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ================= 6. 执行问答并显示来源 =================
query = "澳门的黑脸琵鹭"
print(f"\n>>> [5/5] 用户提问: {query}")
print("-" * 50)

# --- 关键修改：手动分步执行，以便获取来源信息 ---

# 第一步：先检索文档
print("正在检索相关资料...", end="", flush=True)
retrieved_docs = retriever.invoke(query)
print(f" 找到 {len(retrieved_docs)} 份参考资料。\n")

# 格式化文档内容供 AI 阅读
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 第二步：构建完整的 Prompt 并发送给 AI
chain = prompt | llm | StrOutputParser()

print("=== AI 回答 ===")
# 流式打印回答
full_response = ""
for chunk in chain.stream({"context": context_text, "question": query}):
    print(chunk, end="", flush=True)
    full_response += chunk
print("\n")

# 第三步：打印参考来源列表
print("=== 参考资料目录 ===")
seen_sources = set()  # 用于去重，防止同一个文件的不同页码重复显示太多次
i = 1
for doc in retrieved_docs:
    # 获取元数据 source (文件名或URL) 和 page (页码, 只有PDF有)
    source = doc.metadata.get("source", "未知来源")
    page = doc.metadata.get("page", None)

    # 构建显示字符串
    source_info = source
    if page is not None:
        source_info += f" (第 {page + 1} 页)"

    # 打印（这里做了简单的去重，如果你想看所有片段来源，可以去掉 if 判断）
    if source_info not in seen_sources:
        print(f"[{i}] {source_info}")
        # 也可以打印一小段原文预览，验证是否准确
        # print(f"    预览: {doc.page_content[:30].replace('\n', ' ')}...")
        seen_sources.add(source_info)
        i += 1
print("-" * 50)
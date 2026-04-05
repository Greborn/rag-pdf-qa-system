import os
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# ============================================================
# 这是一个“极简本地 PDF RAG 问答系统”示例。
# 你可以把它理解成：
# 1. 先读取本地 PDF
# 2. 把 PDF 内容切成很多小段
# 3. 把每一段变成向量并存入 FAISS
# 4. 用户提问时，先去 FAISS 找最相关的文本片段
# 5. 再把“问题 + 检索到的上下文”交给外部大模型，让它生成答案
#
# 整个项目故意写在一个文件里，方便学习、演示和后续继续修改。
# ============================================================


# =========================
# 一、基础路径配置
# =========================
# 使用当前 app.py 所在目录作为项目根目录。
# 这样无论你把项目复制到哪里，路径都能跟着项目一起走，
# 不依赖项目外部的固定目录。
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PDF_DIR = os.path.join(PROJECT_DIR, "pdfs")
VECTOR_DB_DIR = os.path.join(PROJECT_DIR, "faiss_index")

# 公开版仓库统一使用项目内目录，避免暴露本地绝对路径。
PDF_DIR = PROJECT_PDF_DIR

# 默认 PDF 路径先留空，由页面自动读取可选文件列表。
DEFAULT_PDF = ""


# =========================
# 二、RAG 参数配置
# =========================
# 用户原始要求：
# - chunk_size = 500
# - chunk_overlap = 50
# - 检索最相关的 3 个文本块
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# 默认 embedding 模型。
# 这是一个经典、轻量、容易跑通的开源向量模型。
# 第一次运行时，如果本机没有缓存，通常会自动下载。
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 默认 reranker 模型。
# 开启重排后，会先让 FAISS 召回更多候选片段，再用 cross-encoder 做二阶段排序。
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_FETCH_K = 8


# =========================
# 三、外部大模型 API 预留配置
# =========================
# 这里先预留接口，不强制你立刻接入。
# 当你后面要接 OpenAI 兼容接口、Claude、DeepSeek、通义等时，
# 可以直接在 Streamlit 页面里填写，或者在这里改默认值。
DEFAULT_API_BASE = os.getenv("RAG_API_BASE", "")
DEFAULT_API_KEY = os.getenv("RAG_API_KEY", "")
DEFAULT_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "")


# =========================
# 四、页面基础设置
# =========================
st.set_page_config(page_title="极简本地 PDF RAG", page_icon="📄", layout="wide")
st.title("极简本地 PDF RAG 问答系统")
st.caption("本地 PDF + 文本切分 + HuggingFace Embedding + FAISS + 外部大模型接口预留")


# =========================
# 五、准备项目目录
# =========================
# 启动时自动确保项目内需要的目录存在。
# 这样用户第一次运行时，不需要手动创建 pdfs 或 faiss_index 文件夹。
os.makedirs(PROJECT_PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# =========================
# 六、Session State 初始化
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = DEFAULT_PDF

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_retrieval_mode" not in st.session_state:
    st.session_state.last_retrieval_mode = "基础检索（FAISS）"

if "last_reranker_error" not in st.session_state:
    st.session_state.last_reranker_error = ""


# =========================
# 七、工具函数：读取 PDF 列表
# =========================
def get_pdf_files(pdf_dir: str):
    """
    获取指定目录下的所有 PDF 文件。

    返回值：
        一个列表，列表里的每个元素都是 PDF 的完整路径。
    """
    if not os.path.exists(pdf_dir):
        return []

    files = []
    for name in os.listdir(pdf_dir):
        if name.lower().endswith(".pdf"):
            files.append(os.path.join(pdf_dir, name))

    files.sort()
    return files


# =========================
# 八、工具函数：加载并切分 PDF
# =========================
def load_and_split_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int):
    """
    用 PyPDFLoader 读取本地 PDF，然后用 RecursiveCharacterTextSplitter 做文本切分。

    参数：
        pdf_path: PDF 文件路径
        chunk_size: 每个切片的长度
        chunk_overlap: 切片之间的重叠长度

    返回值：
        切分后的 documents 列表
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    return split_docs


# =========================
# 九、工具函数：构建向量库
# =========================
def build_vectorstore(pdf_path: str, chunk_size: int, chunk_overlap: int, embedding_model_name: str):
    """
    读取 PDF -> 切分 -> 向量化 -> 写入 FAISS。

    这里是整个 RAG 的核心准备流程。
    """
    split_docs = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 保存到本地，方便后续复用。
    # 注意：如果你换了 PDF、chunk_size、chunk_overlap，
    # 最稳妥的做法是重新构建，而不是直接复用旧索引。
    vectorstore.save_local(VECTOR_DB_DIR)
    return vectorstore, split_docs


@st.cache_resource(show_spinner=False)
def load_reranker(model_name: str):
    return CrossEncoder(model_name)


def rerank_documents(question: str, docs: list, reranker_model_name: str, final_top_k: int):
    """
    使用 cross-encoder 对 FAISS 初筛后的候选片段做二阶段重排。
    """
    if not docs:
        return []

    reranker = load_reranker(reranker_model_name)
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked_pairs = sorted(
        zip(docs, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )
    return [doc for doc, _ in ranked_pairs[:final_top_k]]


def retrieve_documents(vectorstore, question: str, top_k: int, use_reranker: bool, reranker_model_name: str, fetch_k: int):
    """
    统一处理检索逻辑：
    - 关闭重排：直接使用 FAISS 相似度检索
    - 开启重排：先扩大召回，再用 cross-encoder 重排
    """
    actual_fetch_k = max(int(top_k), int(fetch_k))
    candidate_docs = vectorstore.similarity_search(question, k=actual_fetch_k)

    if not use_reranker:
        return candidate_docs[:int(top_k)], "基础检索（FAISS）", ""

    try:
        reranked_docs = rerank_documents(question, candidate_docs, reranker_model_name, int(top_k))
        return reranked_docs, "二阶段检索（FAISS + Cross-Encoder 重排）", ""
    except Exception as e:
        fallback_docs = candidate_docs[:int(top_k)]
        return fallback_docs, "基础检索（FAISS，重排加载失败后回退）", str(e)


# =========================
# 十、工具函数：组装 Prompt
# =========================
def build_prompt(question: str, contexts: list):
    """
    把检索到的文本块和用户问题拼成 Prompt。

    这个 Prompt 设计得尽量简单，适合做项目展示时讲解 RAG 的基本流程。
    """
    context_text = "\n\n".join([
        f"【片段{i + 1}】\n{doc.page_content}" for i, doc in enumerate(contexts)
    ])

    prompt = f"""
你是一个基于检索增强生成（RAG）的问答助手。
请优先根据下面提供的参考内容回答问题，不要脱离参考内容胡乱发挥。
如果参考内容不足以回答，请明确说明“参考资料中没有足够信息支持这个问题的准确回答”。

参考内容：
{context_text}

用户问题：
{question}

请用中文回答，并尽量做到：
1. 结论清晰
2. 条理清楚
3. 如果能定位到参考片段，请结合片段内容解释
""".strip()

    return prompt, context_text


# =========================
# 十一、工具函数：调用外部大模型 API
# =========================
def mask_api_key(api_key: str):
    """
    在页面中展示 API Key 状态时做简单脱敏。
    """
    if not api_key:
        return "未填写"
    if len(api_key) <= 8:
        return "已填写（长度较短，已隐藏）"
    return f"{api_key[:4]}...{api_key[-4:]}"


def call_external_llm(api_base: str, api_key: str, model_name: str, prompt: str):
    """
    这里预留一个“外部大模型接口”。

    当前实现采用常见的 OpenAI 兼容接口格式：
    POST /chat/completions

    这样后面你要接很多平台时会比较方便：
    - OpenAI 兼容服务
    - 一些中转平台
    - 某些国产大模型平台的兼容接口

    如果你后面要改成 Claude 原生接口，也可以在这里替换。
    """
    if not api_base or not api_key or not model_name:
        return (
            "当前还没有配置外部大模型 API，所以系统现在只能完成“检索”和“Prompt 组装”，"
            "还不能真正调用大模型生成最终答案。\n\n"
            "请在左侧填写：API Base URL、API Key、Model Name。"
        )

    base_url = api_base.rstrip("/")
    candidate_urls = [f"{base_url}/chat/completions"]
    if not base_url.endswith("/v1"):
        candidate_urls.append(f"{base_url}/v1/chat/completions")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个严谨的中文问答助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    errors = []
    for url in candidate_urls:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            response.encoding = "utf-8"

            try:
                result = response.json()
            except ValueError:
                preview = response.text[:300].strip()
                if not preview:
                    preview = "接口返回了空内容。"
                return (
                    "调用外部大模型失败：接口返回的不是有效 JSON。\n\n"
                    f"请求地址：{url}\n"
                    f"状态码：{response.status_code}\n"
                    f"Content-Type：{response.headers.get('Content-Type', '未知')}\n"
                    f"返回内容预览：{preview}"
                )

            answer = result["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            errors.append(f"{url} -> {e}")

    return "调用外部大模型失败：\n" + "\n".join(errors)


# =========================
# 十二、侧边栏配置区
# =========================
with st.sidebar:
    st.header("参数配置")
    st.caption("这个项目优先用于演示本地 PDF RAG 的完整流程。")

    pdf_files = get_pdf_files(PDF_DIR)
    pdf_options = [""] + pdf_files

    default_index = 0
    if st.session_state.current_pdf in pdf_options:
        default_index = pdf_options.index(st.session_state.current_pdf)

    selected_pdf = st.selectbox(
        "选择 PDF 文件",
        options=pdf_options,
        index=default_index,
        format_func=lambda x: os.path.basename(x) if x else "请选择一个 PDF 文件"
    )

    if selected_pdf:
        st.session_state.current_pdf = selected_pdf

    chunk_size = st.number_input("Chunk Size（切片大小）", min_value=50, max_value=4000, value=CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input("Chunk Overlap（切片重叠）", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=10)
    top_k = st.number_input("Top-K（检索数量）", min_value=1, max_value=10, value=TOP_K, step=1)

    embedding_model_name = st.text_input("Embedding 模型名称", value=EMBEDDING_MODEL_NAME)

    st.markdown("---")
    st.subheader("检索优化配置")
    use_reranker = st.checkbox("启用二阶段重排", value=True)
    reranker_model_name = st.text_input("Reranker 模型名称", value=RERANKER_MODEL_NAME, disabled=not use_reranker)
    reranker_fetch_k = st.number_input(
        "初筛候选数（Fetch-K）",
        min_value=int(top_k),
        max_value=20,
        value=max(int(top_k), RERANKER_FETCH_K),
        step=1,
        disabled=not use_reranker
    )
    st.caption("关闭时使用基础 FAISS 检索；开启后会先扩大召回，再用 cross-encoder 重排。")

    st.markdown("---")
    st.subheader("外部大模型 API 配置")
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE, placeholder="例如：https://api.openai.com/v1")
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password", placeholder="请输入你的 API Key")
    model_name = st.text_input("Model Name", value=DEFAULT_MODEL_NAME, placeholder="例如：gpt-4o-mini")

    api_ready = bool(api_base and api_key and model_name)
    st.markdown("接口状态：" + ("已配置，可直接问答" if api_ready else "未配置，仅可演示检索与 Prompt 组装"))
    st.caption(f"API Key 状态：{mask_api_key(api_key)}")

    st.markdown("---")
    if st.button("构建 / 重建向量库", use_container_width=True):
        if not selected_pdf:
            st.error("请先把 PDF 放到项目目录下的 `pdfs` 文件夹里，然后在这里选择一个 PDF 文件。")
        else:
            with st.spinner("正在读取 PDF、切分文本、生成向量并构建 FAISS 索引，请稍候..."):
                try:
                    vectorstore, split_docs = build_vectorstore(
                        pdf_path=selected_pdf,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        embedding_model_name=embedding_model_name
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_pdf = selected_pdf
                    st.success(f"向量库构建完成，共切分出 {len(split_docs)} 个文本块。")
                except Exception as e:
                    st.error(f"构建向量库失败：{e}")

    if st.button("清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.session_state.last_sources = []
        st.session_state.last_retrieval_mode = "基础检索（FAISS）"
        st.session_state.last_reranker_error = ""
        st.success("聊天记录已清空。")

    st.markdown("---")
    st.info(
        "项目目录说明：\n"
        f"- PDF 目录：`{PDF_DIR}`\n"
        f"- FAISS 索引目录：`{VECTOR_DB_DIR}`\n\n"
        "建议你后续做实验时重点观察：\n"
        "1. Chunk Size = 50 / 500 / 2000\n"
        "2. Top-K = 1 / 3 / 10\n"
        "3. 关闭/开启二阶段重排后的回答差异"
    )


st.markdown("---")
st.subheader("项目当前状态")

pdf_count = len(get_pdf_files(PDF_DIR))
api_ready = bool(api_base and api_key and model_name)
current_retrieval_mode = "二阶段检索（FAISS + Cross-Encoder 重排）" if use_reranker else "基础检索（FAISS）"

col1, col2, col3, col4 = st.columns(4)
col1.metric("可选 PDF 数量", pdf_count)
col2.metric("当前 Top-K", int(top_k))
col3.metric("API 状态", "已配置" if api_ready else "未配置")
col4.metric("检索模式", "重排开启" if use_reranker else "基础检索")

if st.session_state.current_pdf:
    st.info(f"当前已选择 PDF：`{os.path.basename(st.session_state.current_pdf)}`")
else:
    st.warning("当前还没有选中 PDF。请先在左侧选择文档。")

st.caption(f"当前问答将使用：{current_retrieval_mode}")

if st.session_state.vectorstore is None:
    st.caption("当前还没有构建向量库。你可以先点击左侧“构建 / 重建向量库”。")
else:
    st.success("当前向量库已就绪，可以直接提问。")

st.markdown("**推荐演示问题**")
st.markdown("- 这份 PDF 主要讲了什么？")
st.markdown("- 请总结这份文档的核心功能。")
st.markdown("- 文档中关于安装/配置/使用流程是怎么说明的？")
st.markdown("- 文档里提到的芯片/型号/参数是什么？")


# =========================
# 十三、主区域：聊天记录展示
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# 十四、主区域：用户提问处理
# =========================
user_question = st.chat_input("请输入你的问题，例如：这份 PDF 主要讲了什么？")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            warning_text = "请先在左侧选择 PDF，并点击“构建 / 重建向量库”。"
            st.markdown(warning_text)
            st.session_state.messages.append({"role": "assistant", "content": warning_text})
        else:
            try:
                relevant_docs, retrieval_mode, reranker_error = retrieve_documents(
                    vectorstore=st.session_state.vectorstore,
                    question=user_question,
                    top_k=int(top_k),
                    use_reranker=use_reranker,
                    reranker_model_name=reranker_model_name,
                    fetch_k=int(reranker_fetch_k)
                )
                prompt, context_text = build_prompt(user_question, relevant_docs)
                answer = call_external_llm(api_base, api_key, model_name, prompt)

                st.session_state.last_context = context_text
                st.session_state.last_sources = relevant_docs
                st.session_state.last_retrieval_mode = retrieval_mode
                st.session_state.last_reranker_error = reranker_error

                if not api_ready:
                    st.info("当前未配置外部大模型 API，因此本次仅完成了检索与 Prompt 组装演示。")

                st.caption(f"本次检索模式：{retrieval_mode}")
                if reranker_error:
                    st.warning(f"重排未生效，已自动回退到基础检索：{reranker_error}")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_text = f"问答过程出错：{e}"
                st.markdown(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})


# =========================
# 十五、主区域：检索结果展示
# =========================
st.markdown("---")
st.subheader("本次检索到的参考片段")

st.caption(f"最近一次检索模式：{st.session_state.last_retrieval_mode}")
if st.session_state.last_reranker_error:
    st.caption(f"最近一次重排回退原因：{st.session_state.last_reranker_error}")

if st.session_state.last_context:
    with st.expander("查看本次组装后的 Prompt 上下文", expanded=False):
        st.text(st.session_state.last_context)

if st.session_state.last_sources:
    for i, doc in enumerate(st.session_state.last_sources):
        with st.expander(f"参考片段 {i + 1}", expanded=False):
            st.write(doc.page_content)
            st.caption(f"元数据：{doc.metadata}")
else:
    st.caption("当前还没有检索结果。请先把 PDF 放进项目目录下的 `pdfs` 文件夹，构建向量库后再提问。")


# =========================
# 十六、页脚说明
# =========================
st.markdown("---")
st.caption(
    "说明：本项目是一个用于学习和简历展示的极简 RAG Demo。"
    "当前默认采用本地 Embedding + 本地 FAISS + 外部大模型回答。"
)
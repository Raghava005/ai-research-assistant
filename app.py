import re
import urllib.parse
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st
import tempfile

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG
# -----------------------------

SEARCH_RESULTS = 5
PASSAGES_PER_PAGE = 3
TOP_PASSAGES = 4
TIMEOUT = 8
DB_PATH = "chroma_db"

web_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

rag_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=rag_embedding
)


# -----------------------------
# DOCUMENT UPLOAD PROCESSING
# -----------------------------

def process_uploaded_files(files):

    docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())

            loader = PyPDFLoader(tmp.name)

            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    temp_db = Chroma.from_documents(
        documents=chunks,
        embedding=rag_embedding
    )

    return temp_db


# -----------------------------
# WEB SEARCH FUNCTIONS
# -----------------------------

def unwrap_ddg(url):
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except:
        pass
    return url


def search_web(query):

    urls = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=SEARCH_RESULTS):

            url = r.get("href") or r.get("url")

            if url:
                urls.append(unwrap_ddg(url))

    return urls


def fetch_text(url):

    headers = {"User-Agent": "Mozilla/5.0"}

    try:

        r = requests.get(url, timeout=TIMEOUT, headers=headers)

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script","style","nav","footer","header"]):
            tag.extract()

        paragraphs = [p.get_text(" ",strip=True) for p in soup.find_all("p")]

        text = " ".join(paragraphs)

        text = re.sub(r"\s+"," ",text)

        return text

    except:
        return ""


def chunk_text(text, size=120):

    words = text.split()

    chunks=[]

    for i in range(0,len(words),size):
        chunks.append(" ".join(words[i:i+size]))

    return chunks


def cosine(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10)


# -----------------------------
# WEB RESEARCH PIPELINE
# -----------------------------

def web_research(query):

    urls = search_web(query)

    docs=[]

    for u in urls:

        txt = fetch_text(u)

        if not txt:
            continue

        chunks = chunk_text(txt)

        for c in chunks[:PASSAGES_PER_PAGE]:
            docs.append({"url":u,"passage":c})

    texts=[d["passage"] for d in docs]

    if not texts:
        return []

    embeddings=web_model.encode(texts)

    q_emb=web_model.encode([query])[0]

    sims=[cosine(e,q_emb) for e in embeddings]

    idx=np.argsort(sims)[::-1][:TOP_PASSAGES]

    return [docs[i] for i in idx]


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("AI Research Assistant (Hybrid RAG + LLM)")

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

query = st.text_input("Ask a research question")


if st.button("Search"):

    temp_db = None

    if uploaded_files:
        st.subheader("📤 Processing Uploaded Documents")
        temp_db = process_uploaded_files(uploaded_files)


    if temp_db:
        search_db = temp_db
    else:
        search_db = db


    generic_keywords = [
        "what is this paper about",
        "summarize",
        "summary",
        "main idea",
        "content of the paper",
        "explain this paper"
    ]

    is_generic = any(k in query.lower() for k in generic_keywords)

    if is_generic:
        rag_results = search_db.similarity_search(query, k=8)
    else:
        rag_results = search_db.similarity_search(query, k=3)


    if rag_results:

        context = "\n".join([doc.page_content for doc in rag_results])

        st.subheader("📄 Research Paper Results")

        for r in rag_results:
            st.write(r.page_content)
            st.write("---")

    else:
        st.write("No strong match found in documents.")


    st.subheader("🌐 Web Research")

    web_results = web_research(query)

    if web_results:

        for p in web_results:
            st.write(p["passage"])
            st.write(p["url"])
            st.write("---")

    else:
        st.write("No web results found.")
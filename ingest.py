from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os


# Folder where your research papers are stored
DATA_PATH = "documents"

# Vector database folder
DB_PATH = "chroma_db"


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {DATA_PATH}")

    return documents


def split_documents_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    return chunks


def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def save_to_database(chunks: list[Document]):

    embeddings = create_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    db.persist()

    print("Vector database saved successfully")


if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents_text(docs)

    save_to_database(chunks)
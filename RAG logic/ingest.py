import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_documents():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Documents")
    docs = []

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_dir, filename), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        raise ValueError("No text files found in Documents folder.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
    print("Documents ingested successfully.")

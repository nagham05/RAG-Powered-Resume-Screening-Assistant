import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR, RESUME_DIR


def load_resumes(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "candidate": filename.replace(".pdf", ""),
                        "source": file_path,
                    },
                )
            )
    return documents


def build_index():
    docs = load_resumes(RESUME_DIR)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved {len(chunks)} chunks from {len(docs)} resumes to {INDEX_DIR}/")


if __name__ == "__main__":
    build_index()
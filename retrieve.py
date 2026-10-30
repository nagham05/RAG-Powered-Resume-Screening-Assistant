from collections import defaultdict
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR, EMBEDDING_MODEL, TOP_K


def load_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


vectorstore = load_vectorstore()


def retrieve(query, k=TOP_K):
    """Return {candidate: [chunk_text, ...]} for the top-k chunks."""
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
    candidate_chunks = defaultdict(list)
    for doc in docs:
        name = doc.metadata.get("candidate", "Unknown")
        candidate_chunks[name].append(doc.page_content)
    return candidate_chunks


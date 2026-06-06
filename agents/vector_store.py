import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# all-MiniLM-L6-v2 is ~90MB with CPU-only torch — fits Render free tier
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_store = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def build_store(docs, save_path="data/processed/faiss_index"):
    os.makedirs(save_path, exist_ok=True)
    embeddings = get_embeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    return store

def load_store(index_path="data/processed/faiss_index"):
    global _store
    if _store is None:
        embeddings = get_embeddings()
        _store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS store loaded into memory (singleton).")
    return _store




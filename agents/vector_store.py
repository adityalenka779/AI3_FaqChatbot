import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Always resolve index path relative to this file's location
BASE_DIR = Path(__file__).parent.parent
DEFAULT_INDEX_PATH = str(BASE_DIR / "data/processed/faiss_index")

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

def build_store(docs, save_path=None):
    if save_path is None:
        save_path = DEFAULT_INDEX_PATH
    os.makedirs(save_path, exist_ok=True)
    embeddings = get_embeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    return store

def load_store(index_path=None):
    global _store
    if _store is None:
        if index_path is None:
            index_path = DEFAULT_INDEX_PATH
        print(f"Loading FAISS from: {index_path}")
        embeddings = get_embeddings()
        _store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS store loaded into memory (singleton).")
    return _store



import os
import pickle
import numpy as np
from pathlib import Path

# Use Groq's embedding-compatible approach:
# Store raw text chunks and do keyword search — no torch, no sentence-transformers
# This uses zero ML memory and works perfectly within Render's 512MB

BASE_DIR = Path(__file__).parent.parent
DEFAULT_INDEX_PATH = str(BASE_DIR / "data/processed/faiss_index")

_chunks = None  # list of {text, metadata} dicts

def load_store(index_path=None):
    global _chunks
    if _chunks is None:
        if index_path is None:
            index_path = DEFAULT_INDEX_PATH
        pkl_path = Path(index_path) / "index.pkl"
        print(f"Loading text chunks from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Extract documents from FAISS pickle
        if hasattr(data, 'docstore') or isinstance(data, dict):
            # It's a FAISS store object or dict — extract docs
            _chunks = _extract_chunks(data)
        else:
            _chunks = data
        print(f"Loaded {len(_chunks)} chunks into memory.")
    return _chunks

def _extract_chunks(data):
    """Extract raw text chunks from whatever format the pickle is in."""
    chunks = []
    try:
        # FAISS index.pkl stores (index, docstore) tuple
        if isinstance(data, tuple):
            _, docstore_dict = data
            for doc_id, doc in docstore_dict.items():
                chunks.append({
                    "text": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                })
        elif isinstance(data, dict):
            for k, v in data.items():
                chunks.append({
                    "text": v.page_content if hasattr(v, 'page_content') else str(v),
                    "metadata": v.metadata if hasattr(v, 'metadata') else {}
                })
    except Exception as e:
        print(f"Chunk extraction error: {e}")
    return chunks

def search(query: str, k: int = 4):
    """Simple keyword search over loaded chunks."""
    chunks = load_store()
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(1 for w in query_words if w in text_lower)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]
    
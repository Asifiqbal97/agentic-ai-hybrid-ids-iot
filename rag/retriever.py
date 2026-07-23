# =============================================================================
# rag/retriever.py — Query FAISS vector store at inference time
# Called by orchestrator.py when RAG_ENABLED = True
# =============================================================================

import os
import pickle
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAG_DB_PATH

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install: pip install faiss-cpu sentence-transformers")
    sys.exit(1)

# Global cache — loaded once, reused across calls
_index    = None
_chunks   = None
_embedder = None


def _load():
    """Load FAISS index, chunks, and embedder (once)."""
    global _index, _chunks, _embedder

    if _index is not None:
        return  # already loaded

    index_path  = os.path.join(RAG_DB_PATH, "index.faiss")
    chunks_path = os.path.join(RAG_DB_PATH, "chunks.pkl")
    name_path   = os.path.join(RAG_DB_PATH, "embedder_name.txt")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"RAG vector store not found at {RAG_DB_PATH}\n"
            f"Run: python rag/builder.py"
        )

    _index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    with open(name_path) as f:
        embedder_name = f.read().strip()

    _embedder = SentenceTransformer(embedder_name)
    print(f"[RAG] Loaded vector store: {_index.ntotal} chunks")


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve top_k relevant chunks for a query.
    Returns formatted string ready for prompt injection.
    """
    _load()

    query_vec = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = _index.search(query_vec, top_k)

    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx < len(_chunks):
            retrieved.append(f"[{i+1}] {_chunks[idx]}")

    if not retrieved:
        return ""

    return "## Relevant Security Knowledge:\n" + "\n\n".join(retrieved)


if __name__ == "__main__":
    # Quick test
    result = retrieve("MQTT DDoS attack mitigation")
    print(result)

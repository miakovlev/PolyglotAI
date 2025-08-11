import os
import re
import math
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from .utils import read_env, ensure_dir

try:
    import faiss  # type: ignore
except Exception:
    raise RuntimeError("faiss-cpu is required. Install via: pip install faiss-cpu.")


def simple_chunk(text: str, chunk_chars: int = 1000, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


class InMemoryIndex:
    def __init__(self, embed_model: str = None):
        self.embed_model = embed_model or read_env("EMBED_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=read_env("OPENAI_API_KEY"))
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.dim = 0

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = []
        B = 64
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            embs.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
        X = np.vstack(embs).astype("float32")
        return self._normalize(X)

    def build(self, segments: List[Dict[str, Any]]):
        """
        segments: [{text, start?, end?, speaker?}]
        """
        self.meta = segments
        texts = [s["text"] for s in segments]
        X = self.embed(texts)
        self.dim = X.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X)

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        qv = self.embed([q])
        D, I = self.index.search(qv, top_k)
        hits = []
        for idx in I[0]:
            if idx == -1:
                continue
            hits.append(self.meta[idx])
        return hits

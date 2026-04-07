"""
retrieval.py  -  In-memory ChromaDB vector index for Baseline B RAG.
Built once at startup, reused across all requests.
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from config import CORPUS_FILE, TOP_K

_collection = None


def _build_index() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection

    with open(CORPUS_FILE, "r") as f:
        corpus = json.load(f)

    client = chromadb.EphemeralClient()
    ef = embedding_functions.DefaultEmbeddingFunction()
    col = client.create_collection("va_corpus", embedding_function=ef)

    col.add(
        documents=[p["text"] for p in corpus],
        ids=[p["id"] for p in corpus],
        metadatas=[{"title": p["title"]} for p in corpus],
    )
    _collection = col
    return col


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    col = _build_index()
    results = col.query(query_texts=[query], n_results=k)
    return results["documents"][0]

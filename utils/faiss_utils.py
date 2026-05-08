"""
FAISS index loading and similarity search utilities.
"""
import faiss
import numpy as np


def load_faiss_index(index_path):
    return faiss.read_index(index_path)


def search_similar(index, query_embedding, k=5):
    """
    Search top-k similar vectors. Returns indices only (backward compatible).
    """
    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return indices[0]


def search_similar_with_scores(index, query_embedding, k=5):
    """
    Search top-k similar vectors. Returns (indices, distances) for re-ranking.
    Distances are cosine similarity scores (higher = more similar).
    """
    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]
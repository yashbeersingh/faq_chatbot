import faiss
import numpy as np

class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query_vec, top_k):
        scores, indices = self.index.search(
            query_vec.reshape(1, -1), top_k
        )
        return scores[0], indices[0]

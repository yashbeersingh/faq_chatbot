from sentence_transformers import SentenceTransformer
from utils.config import MODEL_NAME

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

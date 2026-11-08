"""The local sentence-embedding model used by the semantic cache."""

import numpy as np
from sentence_transformers import SentenceTransformer

from cache_lab.lm import pick_device

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_id: str = EMBED_MODEL):
        # Pass the device explicitly: sentence-transformers would otherwise pick MPS on a Mac, which crashes
        # on some macOS/torch combinations.
        self.model = SentenceTransformer(model_id, device=pick_device())

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

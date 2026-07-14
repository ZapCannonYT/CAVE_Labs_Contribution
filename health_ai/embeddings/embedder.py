"""
embedder.py — Singleton embedding model for Health AI v3.

Model: all-MiniLM-L6-v2 (90 MB, 384-dimensional, L2-normalised output)

Why a singleton:
    Loading SentenceTransformer takes ~1–2 seconds. We load it once at
    server startup and reuse the same instance for every request.

Thread safety:
    SentenceTransformer.encode() releases the GIL during inference so
    concurrent calls are safe. The singleton creation is protected by a Lock.
"""

import numpy as np
from threading import Lock
from sentence_transformers import SentenceTransformer

from health_ai.config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIM
from health_ai.core.logger import get_logger
from health_ai.core.exceptions import EmbeddingError

log = get_logger(__name__)


class EmbeddingModel:
    """
    Singleton wrapper around SentenceTransformer.

    Usage:
        embedder = EmbeddingModel()
        vectors = embedder.embed(["text one", "text two"])
        # → np.ndarray shape (2, 384), dtype float32, L2-normalised
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
        return cls._instance

    def _ensure_loaded(self):
        if self._loaded:
            return
        log.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self._loaded = True
            log.info(f"Embedding model ready (dim={EMBEDDING_DIM})")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def embed(self, texts: list[str] | str) -> np.ndarray:
        """
        Embed one or more strings.

        Args:
            texts: A single string or a list of strings.

        Returns:
            np.ndarray of shape (N, 384), dtype float32, L2-normalised.
            Each row is the embedding for the corresponding input text.

        Raises:
            EmbeddingError: if encoding fails.
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")

        # Filter out empty strings — they produce zero vectors which pollute results
        texts = [t.strip() for t in texts]
        texts = [t if t else "[empty]" for t in texts]

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-normalise → dot product == cosine sim
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}") from e

    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single string and return a plain Python list of floats.
        Convenient for JSON serialisation in API responses.
        """
        arr = self.embed([text])
        return arr[0].tolist()

"""
chunker.py — Word-based sliding window text chunker for Health AI v3.

Strategy:
    Split on whitespace (words), use a sliding window with overlap.
    This is intentionally simple — it works reliably for medical text
    without requiring any NLP tokeniser as a dependency.

Chunk size: 500 words
Overlap:    100 words  (ensures context is not lost at chunk boundaries)
"""

import uuid
from dataclasses import dataclass, field
from typing import List

from health_ai.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from health_ai.core.logger import get_logger
from health_ai.core.exceptions import ChunkingError

log = get_logger(__name__)


@dataclass
class Chunk:
    """A single chunk of text with its associated metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}


class TextChunker:
    """
    Splits a document's text into overlapping word-window chunks.

    Args:
        chunk_size:  Number of words per chunk (default 500).
        overlap:     Number of words shared between consecutive chunks (default 100).
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ChunkingError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap   # how far we advance each iteration

    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        """
        Chunk a document string into a list of Chunk objects.

        Args:
            text:          The raw document text to split.
            base_metadata: Dict of metadata to attach to every chunk
                           (e.g. filename, source_type, doc_type).

        Returns:
            List of Chunk objects. Empty list if text is empty after stripping.

        Raises:
            ChunkingError: if text is not a string.
        """
        if not isinstance(text, str):
            raise ChunkingError(f"Expected str, got {type(text).__name__}")

        if base_metadata is None:
            base_metadata = {}

        text = text.strip()
        if not text:
            log.warning("TextChunker received empty text — returning empty chunk list.")
            return []

        words = text.split()
        total_words = len(words)
        chunks: List[Chunk] = []

        start = 0
        chunk_index = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            metadata = {
                **base_metadata,
                "chunk_index": chunk_index,
                "word_start": start,
                "word_end": end,
                "total_words": total_words,
            }

            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            chunk_index += 1

            # If we've reached the end, stop — no infinite loop on short docs
            if end == total_words:
                break

            start += self.step

        log.debug(
            f"Chunked {total_words} words → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks

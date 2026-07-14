"""
chunker.py — Word-based, sentence-boundary, and XML-aware text chunker for Health AI v3.

This chunker runs efficiently on edge/mobile devices:
1. Zero heavy NLP dependencies (no SpaCy or NLTK).
2. Uses single-pass native regex scanning for minimal memory footprint.
3. Automatically aligns standard text splits to sentence or line boundaries.
4. Detects and respects XML structures, preventing tag mutilation and extracting tag paths.
"""

import re
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
    Splits a document's text into overlapping, boundary-aligned, or XML-aware chunks.

    Args:
        chunk_size:  Target number of words per chunk.
        overlap:     Maximum number of words shared between consecutive chunks.
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
        self.step = chunk_size - overlap

    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Chunk]:
        """
        Chunk a document string into a list of Chunk objects.

        Args:
            text:          The raw document text to split.
            base_metadata: Dict of metadata to attach to every chunk.

        Returns:
            List of Chunk objects.
        """
        if not isinstance(text, str):
            raise ChunkingError(f"Expected str, got {type(text).__name__}")

        if base_metadata is None:
            base_metadata = {}

        text = text.strip()
        if not text:
            log.warning("TextChunker received empty text — returning empty chunk list.")
            return []

        # Detect XML structures (e.g. starts with tag structure or contains tags)
        is_xml = text.startswith("<") and ">" in text

        if is_xml:
            return self._chunk_xml(text, base_metadata)
        else:
            return self._chunk_standard(text, base_metadata)

    def _chunk_xml(self, text: str, base_metadata: dict) -> List[Chunk]:
        """
        XML-Aware Chunking Strategy (Edge-Optimized).
        Tokenizes XML tags and text content in a single pass to avoid breaking elements.
        """
        tokens = re.findall(r'<[^>]+>|[^<]+', text)
        chunks: List[Chunk] = []
        current_chunk_tokens: List[str] = []
        current_word_count = 0
        path_stack: List[str] = []
        chunk_start_path = ""

        for token in tokens:
            stripped = token.strip()
            if not stripped:
                if current_chunk_tokens:
                    current_chunk_tokens.append(token)
                continue

            if token.startswith("<"):
                # Handle tags
                if token.startswith("</"):
                    # End tag: pop path stack matching the tag name
                    tag_match = re.match(r'^</([a-zA-Z0-9_\-]+)', token)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        if tag_name in path_stack:
                            while path_stack:
                                popped = path_stack.pop()
                                if popped == tag_name:
                                    break
                else:
                    # Start tag
                    tag_match = re.match(r'^<([a-zA-Z0-9_\-]+)', token)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        is_self_closing = token.endswith("/>")
                        if not is_self_closing:
                            path_stack.append(tag_name)

                current_chunk_tokens.append(token)
            else:
                # Handle text blocks
                token_words = token.split()
                token_word_count = len(token_words)

                # Split chunk if word count exceeded
                if current_word_count > 0 and (current_word_count + token_word_count) > self.chunk_size:
                    chunk_text = "".join(current_chunk_tokens).strip()
                    path_str = "/" + "/".join(path_stack)

                    metadata = {
                        **base_metadata,
                        "chunk_type": "xml",
                        "xml_path": chunk_start_path or path_str or "/",
                        "chunk_index": len(chunks),
                    }
                    chunks.append(Chunk(text=chunk_text, metadata=metadata))

                    current_chunk_tokens = []
                    current_word_count = 0
                    chunk_start_path = path_str

                current_chunk_tokens.append(token)
                current_word_count += token_word_count

        if current_chunk_tokens:
            chunk_text = "".join(current_chunk_tokens).strip()
            if chunk_text:
                path_str = "/" + "/".join(path_stack)
                metadata = {
                    **base_metadata,
                    "chunk_type": "xml",
                    "xml_path": chunk_start_path or path_str or "/",
                    "chunk_index": len(chunks),
                }
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

        log.debug(f"XML Chunked into {len(chunks)} chunks.")
        return chunks

    def _chunk_standard(self, text: str, base_metadata: dict) -> List[Chunk]:
        """
        Sentence-Boundary & Line-Aware Chunking Strategy.
        Groups sentences and lines dynamically to avoid cutoffs.
        """
        # Split on sentence boundaries (. ? !) followed by spacing, or newlines
        raw_blocks = re.split(r'(?<=[.!?])\s+|\n+', text)
        blocks = [b.strip() for b in raw_blocks if b.strip()]

        if not blocks:
            return []

        chunks: List[Chunk] = []
        current_blocks: List[str] = []
        current_word_count = 0
        block_idx = 0

        while block_idx < len(blocks):
            block = blocks[block_idx]
            block_words = block.split()
            block_word_count = len(block_words)

            # Fallback: if a single block/sentence is excessively long, split by words
            if block_word_count > self.chunk_size:
                if current_blocks:
                    chunk_text = " ".join(current_blocks)
                    metadata = {
                        **base_metadata,
                        "chunk_type": "standard",
                        "chunk_index": len(chunks),
                    }
                    chunks.append(Chunk(text=chunk_text, metadata=metadata))
                    current_blocks = []
                    current_word_count = 0

                start_w = 0
                while start_w < block_word_count:
                    end_w = min(start_w + self.chunk_size, block_word_count)
                    chunk_words = block_words[start_w:end_w]
                    chunk_text = " ".join(chunk_words)
                    metadata = {
                        **base_metadata,
                        "chunk_type": "standard",
                        "chunk_index": len(chunks),
                    }
                    chunks.append(Chunk(text=chunk_text, metadata=metadata))
                    start_w += self.step

                block_idx += 1
                continue

            # Group blocks
            if current_word_count > 0 and (current_word_count + block_word_count) > self.chunk_size:
                chunk_text = " ".join(current_blocks)
                metadata = {
                    **base_metadata,
                    "chunk_type": "standard",
                    "chunk_index": len(chunks),
                }
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                # Slide back blocks to respect self.overlap (target overlap in words)
                overlap_words = 0
                overlap_blocks = 0
                for rev_block in reversed(current_blocks):
                    rev_count = len(rev_block.split())
                    if overlap_words + rev_count > self.overlap:
                        break
                    overlap_words += rev_count
                    overlap_blocks += 1

                if overlap_blocks > 0:
                    current_blocks = current_blocks[-overlap_blocks:]
                    current_word_count = overlap_words
                else:
                    current_blocks = []
                    current_word_count = 0

            current_blocks.append(block)
            current_word_count += block_word_count
            block_idx += 1

        if current_blocks:
            chunk_text = " ".join(current_blocks)
            metadata = {
                **base_metadata,
                "chunk_type": "standard",
                "chunk_index": len(chunks),
            }
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        log.debug(f"Standard Chunked into {len(chunks)} chunks.")
        return chunks

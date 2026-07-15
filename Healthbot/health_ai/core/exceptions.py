"""
exceptions.py — Custom exception hierarchy for Health AI v3.

Raising specific exceptions makes error handling in the API layer clean
and avoids swallowing unexpected errors silently.
"""


class HealthAIError(Exception):
    """Base class for all Health AI exceptions."""


class ModelNotFoundError(HealthAIError):
    """Raised when the GGUF model file does not exist at the configured path."""


class ModelLoadError(HealthAIError):
    """Raised when the LLM fails to load (corrupt file, out of RAM, etc.)."""


class EmbeddingError(HealthAIError):
    """Raised when the embedding model fails to encode text."""


class OCRError(HealthAIError):
    """Raised when all OCR engines fail to extract text from an image."""


class UnsupportedFileTypeError(HealthAIError):
    """Raised when an uploaded file has an extension we cannot process."""


class EmptyDocumentError(HealthAIError):
    """Raised when a document produces zero text after extraction."""


class ChunkingError(HealthAIError):
    """Raised when the chunker receives invalid input."""


class GenerationError(HealthAIError):
    """Raised when the LLM generation step fails."""

"""
Abstract base for all embedding models (text and modality).

Implementations may support text encoding, modality encoding (image/video/audio),
or both. Callers use the same interface; unsupported methods raise NotImplementedError.
"""

from __future__ import annotations

from abc import ABC
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract interface for any embedding model used in the pipeline.

    - Text: override embed_text() to encode a single string into a vector.
    - Modality: override embed_modality_batch() to encode files (image/video/audio)
      into vectors. Modality key is e.g. "image", "video", "audio".

    Subclasses override only the methods they support; default implementations
    raise NotImplementedError.
    """

    def embed_text(self, text: str) -> List[float]:
        """
        Encode a single text string into an embedding vector.

        Raises NotImplementedError if this embedder does not support text.
        """
        raise NotImplementedError("This embedder does not support text encoding")

    def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of text strings. Default implementation calls embed_text in a loop.
        Subclasses may override for batched encoding.
        """
        return [self.embed_text(t) for t in texts]

    def embed_modality_batch(
        self, modality_key: str, paths: List[str]
    ) -> List[List[float]]:
        """
        Encode a batch of modality assets (file paths) for the given modality.

        - modality_key: one of "image", "video", "audio"
        - paths: list of file paths or URIs
        - Returns: one embedding vector per path (same order); invalid paths may
          be represented as zero vectors or omitted depending on implementation.

        Raises NotImplementedError if this embedder does not support modality.
        """
        raise NotImplementedError("This embedder does not support modality encoding")

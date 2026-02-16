"""
Embedding models: abstract base and implementations for text and modality.

Use BaseEmbedder as the common interface; SentenceTransformerEmbedder for text,
ImageBindEmbedder for image/video/audio.
"""

from fimmia.embedding_models.base import BaseEmbedder
from fimmia.embedding_models.imagebind_embedder import ImageBindEmbedder
from fimmia.embedding_models.text_embedder import SentenceTransformerEmbedder


def get_default_modality_embedder(device: str = "cuda") -> BaseEmbedder:
    """
    Factory returning the default modality embedder (ImageBind).
    """
    return ImageBindEmbedder(device=device)


def get_default_text_embedder(
    model_name: str = "intfloat/e5-mistral-7b-instruct",
    max_seq_length: int = 4096,
    device: str | None = None,
) -> BaseEmbedder:
    """
    Factory returning the default text embedder (SentenceTransformer).
    """
    return SentenceTransformerEmbedder(
        model_name=model_name,
        max_seq_length=max_seq_length,
        device=device,
    )


__all__ = [
    "BaseEmbedder",
    "ImageBindEmbedder",
    "SentenceTransformerEmbedder",
    "get_default_modality_embedder",
    "get_default_text_embedder",
]

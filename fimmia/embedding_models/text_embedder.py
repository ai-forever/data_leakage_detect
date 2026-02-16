"""
Text embedding implementation using SentenceTransformer.
"""

from __future__ import annotations

from typing import List

import torch

from fimmia.embedding_models.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Text embedder backed by SentenceTransformer.

    Exposes embed_text() and embed_text_batch(); embed_modality_batch() is not supported.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        max_seq_length: int = 4096,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = max_seq_length
        if device is not None:
            self._model = self._model.to(device)

    def embed_text(self, text: str) -> List[float]:
        with torch.inference_mode():
            return self._model.encode([text]).tolist()[0]

    def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with torch.inference_mode():
            return self._model.encode(texts).tolist()

    def embed_modality_batch(
        self, modality_key: str, paths: List[str]
    ) -> List[List[float]]:
        return super().embed_modality_batch(modality_key, paths)

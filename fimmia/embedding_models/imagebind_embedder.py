"""
Modality embedding implementation using ImageBind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from fimmia.embedding_models.base import BaseEmbedder


@dataclass
class ImageBindEmbedder(BaseEmbedder):
    """
    Modality embedder backed by facebookresearch/ImageBind.

    Implements embed_modality_batch() for image, video, and audio.
    embed_batch() is an alias for backward compatibility with ModalityEmbedder callers.
    """

    device: str = "cuda"

    def __post_init__(self) -> None:
        try:
            from imagebind import data as ib_data  # type: ignore[import]
            from imagebind.models import imagebind_model  # type: ignore[import]
            from imagebind.models.imagebind_model import ModalityType  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "ImageBind is not installed. Please install it, e.g. "
                "pip install 'git+https://github.com/facebookresearch/ImageBind.git'"
            ) from e

        self._ib_data = ib_data
        self._ModalityType = ModalityType
        self._model = imagebind_model.imagebind_huge(pretrained=True).to(self.device)
        self._model.eval()

        self._modality_map: Dict[str, Tuple[object, object]] = {
            "image": (ModalityType.VISION, ib_data.load_and_transform_vision_data),
            "video": (ModalityType.VISION, ib_data.load_and_transform_video_data),
            "audio": (ModalityType.AUDIO, ib_data.load_and_transform_audio_data),
        }

    def embed_text(self, text: str) -> List[float]:
        return super().embed_text(text)

    def embed_modality_batch(
        self, modality_key: str, paths: List[str]
    ) -> List[List[float]]:
        if not paths:
            return []

        modality_key = modality_key.lower()
        if modality_key not in self._modality_map:
            raise ValueError(
                f"Unsupported modality_key for ImageBindEmbedder: {modality_key!r}"
            )

        ib_modality, loader_fn = self._modality_map[modality_key]

        valid_indices: List[int] = []
        valid_paths: List[str] = []
        for i, p in enumerate(paths):
            if p is None:
                continue
            if isinstance(p, float) and np.isnan(p):
                continue
            valid_indices.append(i)
            valid_paths.append(str(p))

        if not valid_paths:
            return [[] for _ in paths]

        inputs_single = loader_fn(valid_paths, self.device)
        ib_inputs = {ib_modality: inputs_single}

        with torch.no_grad():
            outputs = self._model(ib_inputs)[ib_modality]

        outputs = outputs.detach().cpu()

        result: List[List[float]] = [[] for _ in paths]
        for idx_in_batch, idx_in_all in enumerate(valid_indices):
            result[idx_in_all] = outputs[idx_in_batch].tolist()

        return result

    def embed_batch(self, modality_key: str, paths: List[str]) -> List[List[float]]:
        """Alias for embed_modality_batch(); kept for backward compatibility."""
        return self.embed_modality_batch(modality_key, paths)

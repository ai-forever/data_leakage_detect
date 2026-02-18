"""
FiMMIA: Framework for Multimodal Membership Inference Attacks

A modular framework for membership inference attacks against multimodal large language models.
Supports image, audio, and video modalities.
"""

# Core training and inference
from fimmia.train import (
    train,
    ModelArguments,
    DataTrainingArguments,
    DefaultTrainingArguments,
)

# Model initialization
from fimmia.fimmia_models import init_model, get_all_models

# Neighbor generation
from fimmia.neighbors import NeighborsGenerator, Args as NeighborsArgs

# Embedding models
from fimmia.embedding_models import (
    BaseEmbedder,
    ImageBindEmbedder,
    SentenceTransformerEmbedder,
    get_default_modality_embedder,
    get_default_text_embedder,
)

# Utilities
from fimmia.utils.data import create_data_collator
from fimmia.utils.metrics import get_metrics_from_df, get_df_with_predictions
from fimmia.utils.mds_dataset import get_streaming_ds

__version__ = "1.0.0"

__all__ = [
    # Training
    "train",
    "ModelArguments",
    "DataTrainingArguments",
    "DefaultTrainingArguments",
    # Models
    "init_model",
    "get_all_models",
    # Neighbor generation
    "NeighborsGenerator",
    "NeighborsArgs",
    # Embedding models
    "BaseEmbedder",
    "ImageBindEmbedder",
    "SentenceTransformerEmbedder",
    "get_default_modality_embedder",
    "get_default_text_embedder",
    # Utilities
    "create_data_collator",
    "get_metrics_from_df",
    "get_df_with_predictions",
    "get_streaming_ds",
]

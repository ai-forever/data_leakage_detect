# Pipeline Guide

This guide provides a detailed walkthrough of the FiMMIA training and inference pipeline.

## Overview

The FiMMIA pipeline consists of several stages:

1. **SFT-LoRA Finetuning** (optional): Finetune the target MLLM
2. **Neighbor Generation**: Create semantically similar samples
3. **Embedding Generation**: Extract features from text and modalities
4. **Loss Computation**: Calculate model losses
5. **MDS Dataset Preparation**: Merge embeddings and losses
6. **Attack Model Training**: Train the FiMMIA classifier
7. **Inference**: Evaluate on test data

## Stage 1: SFT-LoRA Finetuning (Optional)

If you want to finetune the target MLLM before running attacks:

```bash
# For image modality
fimmia sft image \
  --train_df_path="data/train.csv" \
  --test_df_path="data/test.csv" \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir="models/sft/image" \
  --num_train_epochs=5

# For video modality
fimmia sft video \
  --train_df_path="data/train.csv" \
  --test_df_path="data/test.csv" \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir="models/sft/video" \
  --num_train_epochs=5

# For audio modality
fimmia sft audio \
  --train_df_path="data/train.csv" \
  --test_df_path="data/test.csv" \
  --model_id="Qwen/Qwen2-Audio-7B-Instruct" \
  --output_dir="models/sft/audio" \
  --num_train_epochs=5
```

**Output:** Finetuned model saved in `output_dir`.

## Stage 2: Neighbor Generation

Generate semantically similar neighbor samples for each input:

```bash
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/train.csv" \
  --max_text_len=4000 \
  --n=25 \
  --mask_size=0.1
```

**Parameters:**
- `model_path`: T5-based model for generating neighbors
- `dataset_path`: Input dataset CSV
- `max_text_len`: Maximum text length to process
- `n`: Number of neighbors per sample
- `mask_size`: Ratio of text to mask

**Output:** Updated CSV with `neighbors` column added.

## Stage 3: Embedding Generation

Generate embeddings for text and optional modality:

### Text-Only

```bash
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --max_seq_length=4096 \
  --device="cuda"
```

### With Modality

```bash
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --max_seq_length=4096 \
  --modality_key=video \
  --device="cuda" \
  --part_size=5000
```

**Parameters:**
- `df_path`: Dataset CSV (must have `neighbors` column)
- `embed_model`: Text embedder model
- `modality_key`: Modality column name (omit for text-only)
- `device`: Computation device

**Output:**
- `embeds/part_*.csv`: Text embeddings
- `{modality}_embeds/part_*.csv`: Modality embeddings (if `modality_key` set)

## Stage 4: Loss Computation

Compute model losses on original and neighbor samples:

### Image Modality

```bash
fimmia loss image \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv" \
  --part_size=5000
```

### Video Modality

```bash
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv" \
  --part_size=5000
```

### Audio Modality

```bash
fimmia loss audio \
  --model_id="Qwen/Qwen2-Audio-7B-Instruct" \
  --model_name="Qwen2-Audio-7B-Instruct" \
  --label=0 \
  --df_path="data/train.csv" \
  --part_size=5000
```

**Parameters:**
- `model_id`: HuggingFace model ID
- `model_name`: Model name for organizing outputs
- `label`: Dataset label (0 for non-members, 1 for members)
- `df_path`: Dataset CSV

**Output:** `loss/{model_name}/part_*.csv` with loss values.

## Stage 5: MDS Dataset Preparation

Merge embeddings and losses into MDS format:

```bash
fimmia mds-dataset \
  --save_dir="data/mds_dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/train.csv" \
  --labels="0,1" \
  --modality_key="video" \
  --shuffle=0 \
  --single_file=1
```

**Parameters:**
- `save_dir`: Output directory for MDS dataset
- `model_name`: Model name (must match loss computation)
- `origin_df_path`: Original dataset CSV
- `labels`: Comma-separated labels (e.g., "0,1")
- `modality_key`: Modality column name (omit for text-only)

**Output:** MDS dataset in `save_dir` ready for training.

## Stage 6: Attack Model Training

Train the FiMMIA attack model:

### Text-Only Model

```bash
fimmia train \
  --train_dataset_path="data/mds_dataset/train" \
  --val_dataset_path="data/mds_dataset/val" \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --embedding_size=4096 \
  --output_dir="models/fimmia_baseline" \
  --num_train_epochs=10 \
  --learning_rate=5e-5
```

### Multimodal Model

```bash
fimmia train \
  --train_dataset_path="data/mds_dataset/train" \
  --val_dataset_path="data/mds_dataset/val" \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --embedding_size=4096 \
  --modality_embedding_size=1024 \
  --output_dir="models/fimmia_multimodal" \
  --num_train_epochs=10 \
  --learning_rate=5e-5 \
  --sigmas_path="data/sigmas.json" \
  --sigmas_type="std"
```

**Parameters:**
- `model_name`: Use `FiMMIABaseLineModelLossNormSTDV2` for text-only, `FiMMIAModalityAllModelLossNormSTDV2` for multimodal
- `embedding_size`: Text embedding dimension
- `modality_embedding_size`: Modality embedding dimension (only for multimodal)

**Output:** Trained model in `output_dir`.

## Stage 7: Inference

Run inference on test data:

```bash
fimmia infer \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --model_path="models/fimmia_baseline" \
  --test_path="data/mds_dataset/test" \
  --save_path="results/predictions.csv" \
  --save_metrics_path="results/metrics.csv"
```

**Output:**
- `predictions.csv`: Predictions for each sample
- `metrics.csv`: Evaluation metrics (AUC, TPR@FPR, etc.)

## Complete Pipeline Script

Here's a complete pipeline script:

```bash
#!/bin/bash

# Stage 1: Neighbors
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/train.csv" \
  --max_text_len=4000

# Stage 2: Embeddings
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --modality_key=video \
  --device="cuda"

# Stage 3: Losses
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv"

# Stage 4: MDS Dataset
fimmia mds-dataset \
  --save_dir="data/mds_dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/train.csv" \
  --labels="0,1" \
  --modality_key="video"

# Stage 5: Training
fimmia train \
  --train_dataset_path="data/mds_dataset/train" \
  --val_dataset_path="data/mds_dataset/val" \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --embedding_size=4096 \
  --modality_embedding_size=1024 \
  --output_dir="models/fimmia" \
  --num_train_epochs=10

# Stage 6: Inference
fimmia infer \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --model_path="models/fimmia" \
  --test_path="data/mds_dataset/test" \
  --save_path="results/predictions.csv" \
  --save_metrics_path="results/metrics.csv"
```

## Tips and Best Practices

1. **Text-only vs Multimodal**: Omit `--modality_key` for text-only datasets. Use `FiMMIABaseLineModelLossNormSTDV2` for text-only.

2. **Batch Processing**: Set `--part_size` appropriately for large datasets to avoid memory issues.

3. **Model Selection**: Choose the right model architecture based on your data:
   - `FiMMIABaseLineModelLossNormSTDV2`: Text-only
   - `FiMMIAModalityAllModelLossNormSTDV2`: Multimodal

4. **Normalization**: Use `--sigmas_path` and `--sigmas_type` for better performance with multimodal models.

5. **GPU Memory**: Use `--part_size` to control memory usage during embedding and loss computation.

For more details, see the [Tutorials](../tutorials/training.md).

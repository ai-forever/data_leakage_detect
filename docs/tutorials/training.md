# Training Tutorial

This tutorial walks you through training a FiMMIA attack model step by step.

## Prerequisites

- FiMMIA installed (see [Installation](../installation.md))
- Dataset in the required format
- GPU access (recommended)

## Dataset Preparation

Your dataset should be a CSV file with columns: `input`, `answer`, optional modality column (`image`/`video`/`audio`), and `ds_name`.

Example:
```csv
input,answer,video,ds_name
"Question: <video> What happens?",A,path/to/video.mp4,ruEnvAQA
```

## Step 1: Generate Neighbors

First, generate semantically similar neighbor samples:

```bash
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/train.csv" \
  --max_text_len=4000 \
  --n=25
```

This adds a `neighbors` column to your dataset.

## Step 2: Generate Embeddings

Generate embeddings for text and modality:

```bash
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --max_seq_length=4096 \
  --modality_key=video \
  --device="cuda" \
  --part_size=5000
```

This creates:
- `embeds/part_*.csv`: Text embeddings
- `video_embeds/part_*.csv`: Video embeddings

## Step 3: Compute Losses

Compute model losses for both members (label=1) and non-members (label=0):

```bash
# For members
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=1 \
  --df_path="data/train.csv"

# For non-members
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/test.csv"
```

## Step 4: Prepare MDS Dataset

Merge embeddings and losses:

```bash
fimmia mds-dataset \
  --save_dir="data/mds_dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/train.csv" \
  --labels="0,1" \
  --modality_key="video" \
  --shuffle=1
```

## Step 5: Train Attack Model

Train the FiMMIA model:

```bash
fimmia train \
  --train_dataset_path="data/mds_dataset/train" \
  --val_dataset_path="data/mds_dataset/val" \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --embedding_size=4096 \
  --modality_embedding_size=1024 \
  --output_dir="models/fimmia" \
  --num_train_epochs=10 \
  --learning_rate=5e-5 \
  --optim="adafactor" \
  --warmup_ratio=0.03
```

## Monitoring Training

Training progress is logged to TensorBoard. View with:

```bash
tensorboard --logdir=models/fimmia
```

## Model Selection

Choose the right model architecture:

- **Text-only**: `FiMMIABaseLineModelLossNormSTDV2`
  - Use when `--modality_key` is not set
  - Only requires `--embedding_size`

- **Multimodal**: `FiMMIAModalityAllModelLossNormSTDV2`
  - Use when `--modality_key` is set
  - Requires both `--embedding_size` and `--modality_embedding_size`

## Hyperparameter Tuning

Key hyperparameters to tune:

- `--learning_rate`: Start with 5e-5, adjust based on convergence
- `--num_train_epochs`: Monitor validation metrics to avoid overfitting
- `--warmup_ratio`: Typically 0.03-0.1
- `--max_grad_norm`: Prevents gradient explosion (e.g., 10.0)

## Next Steps

After training, proceed to [Inference Tutorial](inference.md) to evaluate your model.

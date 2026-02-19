# Quick Start

This guide will help you run your first membership inference attack with FiMMIA in just a few steps.

## Prerequisites

- FiMMIA installed (see [Installation](installation.md))
- A dataset in the required format (see [Data Format](#data-format))
- GPU access (recommended for faster processing)

## Data Format

Your dataset should be a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `input` | Text input with optional modality placeholders | "Question: <image> What is this?" |
| `answer` | Expected answer | "A" |
| `image` / `video` / `audio` | Modality column (optional) | Path to media file |
| `ds_name` | Dataset name | "ruEnvAQA" |

## Quick Example: Text-Only Attack

### Step 1: Generate Neighbors

```bash
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/train.csv" \
  --max_text_len=4000
```

This creates semantically similar neighbor samples for each input.

### Step 2: Generate Embeddings

```bash
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --max_seq_length=4096 \
  --device="cuda"
```

This generates text embeddings for inputs and neighbors.

### Step 3: Compute Losses

```bash
fimmia loss image \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv"
```

This computes model losses on original and neighbor samples.

### Step 4: Prepare MDS Dataset

```bash
fimmia mds-dataset \
  --save_dir="data/mds_dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/train.csv" \
  --labels="0,1" \
  --modality_key="image"
```

This merges embeddings and losses into a training-ready format.

### Step 5: Train Attack Model

```bash
fimmia train \
  --train_dataset_path="data/mds_dataset/train" \
  --val_dataset_path="data/mds_dataset/val" \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --embedding_size=4096 \
  --output_dir="models/fimmia_model" \
  --num_train_epochs=10
```

### Step 6: Run Inference

```bash
fimmia infer \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --model_path="models/fimmia_model" \
  --test_path="data/mds_dataset/test" \
  --save_path="results/predictions.csv" \
  --save_metrics_path="results/metrics.csv"
```

## Quick Example: Python API

You can also use FiMMIA programmatically:

```python
from fimmia import (
    NeighborsGenerator,
    NeighborsArgs,
    get_default_text_embedder,
    train,
    ModelArguments,
    DataTrainingArguments,
    DefaultTrainingArguments,
)

# Generate neighbors
args = NeighborsArgs(
    dataset_path="data/train.csv",
    model_path="ai-forever/FRED-T5-1.7B",
    max_text_len=4000
)
ng = NeighborsGenerator(args)
df = ng.predict(args.dataset_path)

# Get embedding model
text_embedder = get_default_text_embedder(
    model_name="intfloat/e5-mistral-7b-instruct"
)

# Train model
model_args = ModelArguments(
    model_name="FiMMIABaseLineModelLossNormSTDV2",
    embedding_size=4096
)
data_args = DataTrainingArguments(
    train_dataset_path="data/mds_dataset/train",
    val_dataset_path="data/mds_dataset/val"
)
training_args = DefaultTrainingArguments(
    output_dir="models/fimmia_model",
    num_train_epochs=10
)

trainer = train(model_args, data_args, training_args)
```

## Next Steps

- Learn more about the [CLI commands](usage/cli-reference.md)
- Explore the [Python API](usage/python-api.md)
- Follow a complete [Training Tutorial](tutorials/training.md)
- Check out [Code Examples](tutorials/examples.md)

# CLI Reference

Complete reference for all FiMMIA command-line interface commands.

## Overview

FiMMIA provides a structured CLI with subcommands for different operations. The main command is `fimmia`, with subcommands for training, inference, and data processing.

## Main Commands

### `fimmia train`

Train a FiMMIA attack model.

**Usage:**
```bash
fimmia train [OPTIONS]
```

**Options:**
- `--train_dataset_path` (required): Path to training MDS dataset
- `--val_dataset_path`: Path to validation MDS dataset
- `--model_name` (default: "BaseLineModel"): FiMMIA model architecture name
  - Options: `FiMMIABaseLineModelLossNormSTDV2`, `FiMMIAModalityAllModelLossNormSTDV2`
- `--embedding_size` (default: 4096): Text embedding dimension
- `--modality_embedding_size` (default: 1024): Modality embedding dimension
- `--output_dir` (required): Directory to save the trained model
- `--num_train_epochs` (default: 10): Number of training epochs
- `--optim` (default: "adamw_torch_fused"): Optimizer name
- `--learning_rate` (default: 2e-5): Learning rate
- `--max_grad_norm`: Maximum gradient norm for clipping
- `--warmup_ratio` (default: 0.03): Warmup ratio
- `--sigmas_path`: Path to normalization parameters JSON file
- `--sigmas_type`: Type of normalization ("std" or other)

**Example:**
```bash
fimmia train \
  --train_dataset_path="data/mds/train" \
  --val_dataset_path="data/mds/val" \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --embedding_size=4096 \
  --modality_embedding_size=1024 \
  --output_dir="models/fimmia" \
  --num_train_epochs=10
```

### `fimmia infer`

Run inference with a trained FiMMIA model.

**Usage:**
```bash
fimmia infer [OPTIONS]
```

**Options:**
- `--model_name` (required): FiMMIA model architecture name
- `--model_path` (required): Path to trained model directory
- `--test_path` (required): Path to test MDS dataset
- `--save_path` (required): Path to save predictions CSV
- `--save_metrics_path` (required): Path to save metrics CSV
- `--sigmas_path`: Path to normalization parameters JSON file
- `--sigmas_type`: Type of normalization

**Example:**
```bash
fimmia infer \
  --model_name="FiMMIABaseLineModelLossNormSTDV2" \
  --model_path="models/fimmia" \
  --test_path="data/mds/test" \
  --save_path="results/predictions.csv" \
  --save_metrics_path="results/metrics.csv"
```

### `fimmia neighbors`

Generate semantically similar neighbor samples.

**Usage:**
```bash
fimmia neighbors [OPTIONS]
```

**Options:**
- `--dataset_path` (required): Path to input dataset CSV
- `--model_path` (required): Path to T5 model for neighbor generation
- `--mask_token` (default: "<extra_id_0>"): Mask token
- `--end_token` (default: "<extra_id_1>"): End token
- `--max_length_ratio` (default: 5): Maximum length ratio
- `--prefix_token` (default: "<SC1>"): Prefix token
- `--n` (default: 25): Number of neighbors per sample
- `--mask_size` (default: 0.1): Mask size ratio
- `--max_masks` (default: 40): Maximum number of masks
- `--min_masks` (default: 5): Minimum number of masks
- `--num_return_sequences` (default: 1): Number of return sequences
- `--max_text_len` (default: 3000): Maximum text length
- `--user_answer` (default: 0): User answer mode (0 or 1)
- `--modality_column`: Modality column name
- `--modality_output_dir`: Output directory for modality neighbors

**Example:**
```bash
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/train.csv" \
  --max_text_len=4000
```

### `fimmia embeds`

Generate embeddings for text and optional modality.

**Usage:**
```bash
fimmia embeds [OPTIONS]
```

**Options:**
- `--df_path` (required): Path to dataset CSV
- `--embed_model` (default: "intfloat/e5-mistral-7b-instruct"): Text embedder model
- `--max_seq_length` (default: 4096): Maximum sequence length
- `--user_answer` (default: 0): User answer mode
- `--modality_key`: Modality column name (image/video/audio)
- `--ignore_modality_neighbors` (default: 0): Ignore modality neighbors
- `--device` (default: "cuda"): Device for computation
- `--part_size` (default: 5000): Lines per output part file
- `--run_single_file` (default: 1): Process single file (1) or batches (0)

**Example:**
```bash
fimmia embeds \
  --df_path="data/train.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --max_seq_length=4096 \
  --modality_key=video \
  --device="cuda"
```

### `fimmia loss`

Compute model losses for different modalities.

**Usage:**
```bash
fimmia loss {image|video|audio} [OPTIONS]
```

**Common Options:**
- `--model_id` (required): HuggingFace model ID
- `--model_name` (required): Model name for storing results
- `--label` (required): Dataset label (0 or 1)
- `--df_path` (required): Path to dataset CSV
- `--part_size` (default: 5000): Lines per part file

**Example (Image):**
```bash
fimmia loss image \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv"
```

**Example (Video):**
```bash
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/train.csv"
```

**Example (Audio):**
```bash
fimmia loss audio \
  --model_id="Qwen/Qwen2-Audio-7B-Instruct" \
  --model_name="Qwen2-Audio-7B-Instruct" \
  --label=0 \
  --df_path="data/train.csv"
```

### `fimmia sft`

SFT-LoRA finetuning for different modalities.

**Usage:**
```bash
fimmia sft {image|video|audio} [OPTIONS]
```

**Common Options:**
- `--train_df_path` (required): Path to training dataset
- `--test_df_path` (required): Path to test dataset
- `--model_id` (required): HuggingFace model ID
- `--output_dir` (required): Output directory for finetuned model
- `--num_train_epochs` (default: 5): Number of training epochs
- `--per_device_train_batch_size` (default: 4): Training batch size
- `--per_device_eval_batch_size` (default: 4): Evaluation batch size
- `--learning_rate` (default: 2e-5): Learning rate
- `--max_grad_norm` (default: 0.3): Maximum gradient norm
- `--warmup_ratio` (default: 0.03): Warmup ratio

**Example:**
```bash
fimmia sft image \
  --train_df_path="data/train.csv" \
  --test_df_path="data/test.csv" \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir="models/sft" \
  --num_train_epochs=5
```

### `fimmia mds-dataset`

Prepare MDS dataset from embeddings and losses.

**Usage:**
```bash
fimmia mds-dataset [OPTIONS]
```

**Options:**
- `--save_dir` (required): Directory to save MDS dataset
- `--model_name` (required): Model name
- `--origin_df_path` (required): Path to original dataset CSV
- `--shuffle` (default: 0): Shuffle data (0 or 1)
- `--labels` (required): Comma-separated list of labels (e.g., "0,1")
- `--modality_key`: Modality column name
- `--single_file` (default: 1): Process single file (1) or batches (0)
- `--sigmas_path`: Path to normalization parameters
- `--sigmas_fields`: Fields to normalize

**Example:**
```bash
fimmia mds-dataset \
  --save_dir="data/mds_dataset" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/train.csv" \
  --labels="0,1" \
  --modality_key="video"
```

### `fimmia attribute`

Run gradient-based feature attribution.

**Usage:**
```bash
fimmia attribute [OPTIONS]
```

**Options:**
- `--model_dir` (required): Path to FiMMIA model directory
- `--mds_dataset_path` (required): Path to MDS dataset
- `--model_cls` (required): Model class name
- `--embedding_size` (default: 4096): Embedding dimension
- `--modality_embedding_size` (default: 1024): Modality embedding dimension
- `--add_attribution_noise` (default: False): Add stochastic perturbations
- `--create_graphs` (default: True): Create attribution graphs

**Example:**
```bash
fimmia attribute \
  --model_dir="models/fimmia" \
  --mds_dataset_path="data/mds_dataset" \
  --model_cls="BaseLineModelV2" \
  --embedding_size=4096
```

## Shift Detection Commands

### `shift-attack`

Run baseline membership inference attacks.

**Usage:**
```bash
shift-attack [OPTIONS]
```

**Options:**
- `--dataset` (required): Dataset name
  - Options: `wikimia`, `bookmia`, `laion_mi`, `laion_mi_image`, `custom`, etc.
- `--attack` (default: "bag_of_words"): Attack method
  - Options: `date_detection`, `bag_of_words`, `greedy_selection`, `bag_of_visual_words`, `bag_of_audio_words`
- `--fpr_budget` (default: 1.0): FPR budget for TPR@FPR metric
- `--plot_roc`: Plot ROC curve
- `--hypersearch`: Redo hyperparameter search
- `--custom_data_path`: Path to custom dataset (if `--dataset=custom`)
- `--custom_feature_column`: Feature column name (if `--dataset=custom`)
- `--custom_label_column`: Label column name (if `--dataset=custom`)

**Example:**
```bash
shift-attack \
  --dataset=bookmia \
  --attack=bag_of_words \
  --fpr_budget=1.0 \
  --plot_roc
```

## Getting Help

For help on any command, use the `--help` flag:

```bash
fimmia --help
fimmia train --help
shift-attack --help
```

# Inference Tutorial

This tutorial shows how to run inference with a trained FiMMIA model.

## Prerequisites

- Trained FiMMIA model (see [Training Tutorial](training.md))
- Test dataset prepared in MDS format

## Step 1: Prepare Test Data

Prepare test data following the same pipeline as training:

```bash
# Generate neighbors for test set
fimmia neighbors \
  --model_path="ai-forever/FRED-T5-1.7B" \
  --dataset_path="data/test.csv" \
  --max_text_len=4000

# Generate embeddings
fimmia embeds \
  --df_path="data/test.csv" \
  --embed_model="intfloat/e5-mistral-7b-instruct" \
  --modality_key=video \
  --device="cuda"

# Compute losses
fimmia loss video \
  --model_id="Qwen/Qwen2.5-VL-3B-Instruct" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --label=0 \
  --df_path="data/test.csv"

# Prepare MDS dataset
fimmia mds-dataset \
  --save_dir="data/mds_test" \
  --model_name="Qwen2.5-VL-3B-Instruct" \
  --origin_df_path="data/test.csv" \
  --labels="0" \
  --modality_key="video"
```

## Step 2: Run Inference

Run inference with your trained model:

```bash
fimmia infer \
  --model_name="FiMMIAModalityAllModelLossNormSTDV2" \
  --model_path="models/fimmia" \
  --test_path="data/mds_test" \
  --save_path="results/predictions.csv" \
  --save_metrics_path="results/metrics.csv"
```

**Important:** The `--model_name` must match the architecture used during training.

## Step 3: Analyze Results

The inference outputs two files:

### Predictions CSV

Contains predictions for each sample:
- `label`: True label
- `prediction`: Predicted label
- `score`: Confidence score

### Metrics CSV

Contains evaluation metrics:
- `AUC`: Area Under ROC Curve
- `TPR@1%FPR`: True Positive Rate at 1% False Positive Rate
- `TPR@5%FPR`: True Positive Rate at 5% False Positive Rate
- `Accuracy`: Classification accuracy

## Python API Inference

You can also run inference programmatically:

```python
from fimmia import init_model, ModelArguments
from fimmia.utils.metrics import get_df_with_predictions
from transformers import Trainer, TrainingArguments
from fimmia.utils.data import create_data_collator

# Initialize model
model_args = ModelArguments(
    model_name="FiMMIAModalityAllModelLossNormSTDV2",
    model_path="models/fimmia",
    embedding_size=4096,
    modality_embedding_size=1024
)
model = init_model(model_args)

# Create data collator
data_collator = create_data_collator(
    model_name="FiMMIAModalityAllModelLossNormSTDV2"
)

# Create trainer
training_args = TrainingArguments(
    output_dir="models/fimmia",
    eval_strategy="no"
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator
)

# Get predictions
df_pred = get_df_with_predictions("data/mds_test", trainer)
df_pred.to_csv("results/predictions.csv", index=False)

# Calculate metrics
from fimmia.utils.metrics import get_metrics_from_df
metrics = get_metrics_from_df(df_pred)
metrics.to_csv("results/metrics.csv", index=False)
```

## Interpreting Results

### AUC Score

- **> 0.7**: Good attack performance
- **0.6-0.7**: Moderate performance
- **< 0.6**: Poor performance (may need more training data or tuning)

### TPR@FPR

- **TPR@1%FPR**: True positive rate when false positive rate is 1%
- Higher is better
- Useful for understanding attack performance at low false positive rates

## Common Issues

### Model Architecture Mismatch

**Error:** Model architecture doesn't match saved model

**Solution:** Ensure `--model_name` matches the architecture used during training.

### Missing Normalization Parameters

**Error:** Missing sigmas for normalization

**Solution:** Provide `--sigmas_path` and `--sigmas_type` if used during training.

### Memory Issues

**Error:** Out of memory during inference

**Solution:** Process data in smaller batches or use a smaller model.

## Next Steps

- Analyze results and compare with baselines
- Try different model architectures
- Experiment with hyperparameters
- See [Examples](examples.md) for more use cases

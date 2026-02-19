# Python API

FiMMIA provides a comprehensive Python API for programmatic usage. This guide covers the main components and usage patterns.

## Core Components

### Training

```python
from fimmia import (
    train,
    ModelArguments,
    DataTrainingArguments,
    DefaultTrainingArguments,
)

# Configure model
model_args = ModelArguments(
    model_name="FiMMIAModalityAllModelLossNormSTDV2",
    embedding_size=4096,
    modality_embedding_size=1024,
    sigmas_path="data/sigmas.json",
    sigmas_type="std"
)

# Configure data
data_args = DataTrainingArguments(
    train_dataset_path="data/mds/train",
    val_dataset_path="data/mds/val"
)

# Configure training
training_args = DefaultTrainingArguments(
    output_dir="models/fimmia",
    num_train_epochs=10,
    learning_rate=5e-5,
    optim="adafactor",
    max_grad_norm=10,
    warmup_ratio=0.03
)

# Train model
trainer = train(model_args, data_args, training_args)
```

### Neighbor Generation

```python
from fimmia import NeighborsGenerator, NeighborsArgs

# Configure neighbor generation
args = NeighborsArgs(
    dataset_path="data/train.csv",
    model_path="ai-forever/FRED-T5-1.7B",
    max_text_len=4000,
    n=25,
    mask_size=0.1
)

# Generate neighbors
ng = NeighborsGenerator(args)
df_with_neighbors = ng.predict(args.dataset_path)
```

### Embedding Models

```python
from fimmia import (
    get_default_text_embedder,
    get_default_modality_embedder,
    BaseEmbedder
)

# Get text embedder
text_embedder = get_default_text_embedder(
    model_name="intfloat/e5-mistral-7b-instruct",
    max_seq_length=4096,
    device="cuda"
)

# Get modality embedder
modality_embedder = get_default_modality_embedder(device="cuda")

# Use embedders
text_emb = text_embedder.encode_text("Sample text")
image_emb = modality_embedder.encode_image("path/to/image.jpg")
```

### Model Initialization

```python
from fimmia import init_model, ModelArguments

# Initialize model
model_args = ModelArguments(
    model_name="FiMMIABaseLineModelLossNormSTDV2",
    embedding_size=4096,
    model_path="models/fimmia"  # For loading pretrained
)

model = init_model(model_args)
```

### Metrics and Evaluation

```python
from fimmia import get_metrics_from_df, get_df_with_predictions
from transformers import Trainer

# Get predictions
df_pred = get_df_with_predictions("data/mds/test", trainer)

# Calculate metrics
metrics = get_metrics_from_df(df_pred)
print(metrics)
```

## Advanced Usage

### Custom Embedding Models

```python
from fimmia.embedding_models import BaseEmbedder, SentenceTransformerEmbedder

# Create custom embedder
class CustomEmbedder(BaseEmbedder):
    def encode_text(self, text: str):
        # Your implementation
        pass

# Or use existing implementations
embedder = SentenceTransformerEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_seq_length=512
)
```

### Custom Data Collators

```python
from fimmia.utils.data import create_data_collator

# Create data collator
data_collator = create_data_collator(
    model_name="FiMMIAModalityAllModelLossNormSTDV2",
    sigmas_path="data/sigmas.json",
    sigmas_type="std"
)
```

### Streaming Datasets

```python
from fimmia.utils.mds_dataset import get_streaming_ds

# Create streaming dataset
train_ds = get_streaming_ds(
    paths=["data/mds/train"],
    shuffle=True
)

val_ds = get_streaming_ds(
    paths=["data/mds/val"],
    shuffle=False
)
```

## Complete Example

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

# Step 1: Generate neighbors
neighbor_args = NeighborsArgs(
    dataset_path="data/train.csv",
    model_path="ai-forever/FRED-T5-1.7B",
    max_text_len=4000
)
ng = NeighborsGenerator(neighbor_args)
df = ng.predict(neighbor_args.dataset_path)

# Step 2: Get embedding models
text_embedder = get_default_text_embedder(
    model_name="intfloat/e5-mistral-7b-instruct"
)

# Step 3: Train model
model_args = ModelArguments(
    model_name="FiMMIABaseLineModelLossNormSTDV2",
    embedding_size=4096
)
data_args = DataTrainingArguments(
    train_dataset_path="data/mds/train",
    val_dataset_path="data/mds/val"
)
training_args = DefaultTrainingArguments(
    output_dir="models/fimmia",
    num_train_epochs=10
)

trainer = train(model_args, data_args, training_args)
```

## Shift Detection API

```python
from shift_detection import (
    bag_of_words_basic,
    date_detection_basic,
    greedy_selection_basic,
    bag_of_visual_words_basic,
    bag_of_audio_words_basic,
)

# Run bag of words attack
bag_of_words_basic(
    X=text_features,
    y=labels,
    dataset_name="bookmia",
    fpr_budget=1.0,
    plot_roc=True
)

# Run visual attack
bag_of_visual_words_basic(
    X=image_paths,
    y=labels,
    dataset_name="laion_mi_image",
    fpr_budget=1.0,
    plot_roc=True,
    hypersearch=False
)
```

For more details, see the [API Reference](../api/fimmia.md).

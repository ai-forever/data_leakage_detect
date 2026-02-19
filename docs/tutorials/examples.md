# Code Examples

This page contains practical code examples for common FiMMIA use cases.

## Complete Training Pipeline

```python
from fimmia import (
    NeighborsGenerator,
    NeighborsArgs,
    train,
    ModelArguments,
    DataTrainingArguments,
    DefaultTrainingArguments,
)

# Step 1: Generate neighbors
neighbor_args = NeighborsArgs(
    dataset_path="data/train.csv",
    model_path="ai-forever/FRED-T5-1.7B",
    max_text_len=4000,
    n=25
)
ng = NeighborsGenerator(neighbor_args)
df = ng.predict(neighbor_args.dataset_path)

# Step 2: Train model
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
    num_train_epochs=10,
    learning_rate=5e-5
)

trainer = train(model_args, data_args, training_args)
```

## Custom Embedding Pipeline

```python
from fimmia import get_default_text_embedder, get_default_modality_embedder
import pandas as pd

# Load data
df = pd.read_csv("data/train.csv")

# Initialize embedders
text_embedder = get_default_text_embedder(
    model_name="intfloat/e5-mistral-7b-instruct",
    max_seq_length=4096
)
modality_embedder = get_default_modality_embedder(device="cuda")

# Generate embeddings
text_embeddings = []
modality_embeddings = []

for _, row in df.iterrows():
    # Text embedding
    text_emb = text_embedder.encode_text(row['input'])
    text_embeddings.append(text_emb)
    
    # Modality embedding (if available)
    if 'video' in row and pd.notna(row['video']):
        modality_emb = modality_embedder.encode_video(row['video'])
        modality_embeddings.append(modality_emb)

# Save embeddings
embeddings_df = pd.DataFrame({
    'text_embedding': text_embeddings,
    'modality_embedding': modality_embeddings if modality_embeddings else None
})
embeddings_df.to_csv("embeddings.csv", index=False)
```

## Batch Processing

```python
from fimmia.utils.mds_dataset import get_streaming_ds
from transformers import Trainer
import torch

# Create streaming dataset
train_ds = get_streaming_ds(
    paths=["data/mds/train/part_0", "data/mds/train/part_1"],
    shuffle=True
)

# Process in batches
batch_size = 32
for i in range(0, len(train_ds), batch_size):
    batch = train_ds[i:i+batch_size]
    # Process batch
    pass
```

## Evaluation Metrics

```python
from fimmia.utils.metrics import get_metrics_from_df, get_df_with_predictions
from transformers import Trainer
import pandas as pd

# Get predictions
df_pred = get_df_with_predictions("data/mds/test", trainer)

# Calculate metrics
metrics = get_metrics_from_df(df_pred)
print(f"AUC: {metrics['AUC'].iloc[0]}")
print(f"TPR@1%FPR: {metrics['TPR@1%FPR'].iloc[0]}")

# Detailed analysis
from sklearn.metrics import classification_report, confusion_matrix

y_true = df_pred['label']
y_pred = df_pred['prediction']

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

## Shift Detection Examples

### Text Attack

```python
from shift_detection import bag_of_words_basic
import pandas as pd

# Load data
df = pd.read_csv("data/dataset.csv")
X = df['text'].tolist()
y = df['label'].tolist()

# Run attack
bag_of_words_basic(
    X=X,
    y=y,
    dataset_name="bookmia",
    fpr_budget=1.0,
    plot_roc=True,
    hypersearch=False
)
```

### Visual Attack

```python
from shift_detection import bag_of_visual_words_basic
import pandas as pd

# Load data
df = pd.read_csv("data/images.csv")
image_paths = df['image_path'].tolist()
y = df['label'].tolist()

# Run attack
bag_of_visual_words_basic(
    X=image_paths,
    y=y,
    dataset_name="laion_mi_image",
    fpr_budget=1.0,
    plot_roc=True,
    hypersearch=True
)
```

### Audio Attack

```python
from shift_detection import bag_of_audio_words_basic
import pandas as pd

# Load data
df = pd.read_csv("data/audio.csv")
audio_paths = df['audio_path'].tolist()
y = df['label'].tolist()

# Run attack
bag_of_audio_words_basic(
    audio_paths=audio_paths,
    y=y,
    dataset_name="custom",
    fpr_budget=1.0,
    plot_roc=True
)
```

## Model Comparison

```python
from fimmia import init_model, ModelArguments
import torch

# Compare different model architectures
models = [
    "FiMMIABaseLineModelLossNormSTDV2",
    "FiMMIAModalityAllModelLossNormSTDV2"
]

results = {}
for model_name in models:
    model_args = ModelArguments(
        model_name=model_name,
        embedding_size=4096,
        modality_embedding_size=1024 if "ModalityAll" in model_name else None
    )
    model = init_model(model_args)
    
    # Evaluate model
    # ... evaluation code ...
    
    results[model_name] = {
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
    }

print(results)
```

## Custom Data Collator

```python
from fimmia.utils.data import create_data_collator
from transformers import default_data_collator

# Create custom collator with normalization
data_collator = create_data_collator(
    model_name="FiMMIAModalityAllModelLossNormSTDV2",
    sigmas_path="data/sigmas.json",
    sigmas_type="std"
)

# Or use default
default_collator = default_data_collator
```

## Working with Different Modalities

### Image Modality

```python
from fimmia import get_default_modality_embedder

embedder = get_default_modality_embedder(device="cuda")
image_emb = embedder.encode_image("path/to/image.jpg")
```

### Video Modality

```python
video_emb = embedder.encode_video("path/to/video.mp4")
```

### Audio Modality

```python
audio_emb = embedder.encode_audio("path/to/audio.wav")
```

## Notebook Integration

See the example notebook at `examples/Finetune_and_Inference_example.ipynb` for a complete Jupyter notebook example.

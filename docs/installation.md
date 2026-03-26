# Installation

This guide will help you install FiMMIA and its dependencies.

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training and inference)
- 16GB+ RAM (recommended)

## Installation Methods

### Development Installation (Recommended)

For development and contributions, install in editable mode:

```bash
git clone https://github.com/ai-forever/data_leakage_detect.git
cd data_leakage_detect
uv pip install -e .
```

This will:
- Install the `fimmia` and `shift_detection` packages
- Install all required dependencies
- Make CLI commands (`fimmia` and `shift-attack`) available in your PATH

### Standard Installation

For regular usage:

```bash
uv pip install git+https://github.com/ai-forever/data_leakage_detect.git
```

## Dependencies

FiMMIA requires the following key dependencies:

- **PyTorch** (2.7.1) - Deep learning framework
- **Transformers** (4.56.2) - HuggingFace transformers library
- **Sentence Transformers** (4.1.0) - Text embeddings
- **ImageBind** - Multimodal embeddings (installed from GitHub)
- **Pandas, NumPy** - Data processing
- **Scikit-learn** - Machine learning utilities

See `requirements.txt` for the complete list of dependencies.

## Verifying Installation

After installation, verify that FiMMIA is correctly installed:

```bash
# Check CLI commands
fimmia --help
shift-attack --help

# Check Python import
python -c "import fimmia; print(fimmia.__version__)"
```

## GPU Setup

For GPU acceleration, ensure you have:

1. **CUDA Toolkit** installed (compatible with PyTorch 2.7.1)
2. **cuDNN** installed
3. **NVIDIA drivers** up to date

Verify GPU availability:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Troubleshooting

### ImageBind Installation Issues

If ImageBind fails to install, try:

```bash
uv pip install git+https://github.com/facebookresearch/ImageBind.git
```

### CUDA/GPU Issues

If you encounter CUDA-related errors:

1. Verify PyTorch CUDA compatibility: `python -c "import torch; print(torch.version.cuda)"`
2. Reinstall PyTorch with correct CUDA version if needed
3. Check GPU memory availability: `nvidia-smi`

### Import Errors

If you get import errors after installation:

1. Ensure you're using the correct Python environment
2. Verify installation: `uv pip list | grep fimmia`
3. Reinstall in editable mode: `uv pip install -e .`

## Documentation

The documentation is built using **Zensical** (a backward-compatible alternative to MkDocs). To build locally:

```bash
uv pip install -r docs/requirements.txt
zensical build
zensical serve  # To preview locally
```

See [Migration Notes](MIGRATION.md) for details about the switch from MkDocs to Zensical.

## Next Steps

Once installed, proceed to the [Quick Start Guide](quickstart.md) to run your first membership inference attack.

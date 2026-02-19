# Documentation

This directory contains the source files for the FiMMIA documentation.

## Building Locally

### Using Zensical (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Build documentation
zensical build

# Serve locally (with live reload)
zensical serve
```

The documentation will be available at `http://127.0.0.1:8000`

### Fallback: Using MkDocs 1.x

If Zensical is not yet available, you can temporarily use MkDocs 1.x:

```bash
# Install MkDocs 1.x
pip install "mkdocs<2.0" mkdocs-material mkdocstrings[python]

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
```

**Note**: MkDocs 1.x is unmaintained and will not receive updates. Zensical is the recommended path forward.

## Project Structure

```
docs/
├── index.md              # Homepage
├── installation.md       # Installation guide
├── quickstart.md         # Quick start guide
├── MIGRATION.md          # Migration notes (MkDocs → Zensical)
├── usage/                # Usage documentation
│   ├── cli-reference.md
│   ├── python-api.md
│   └── pipeline.md
├── api/                  # API reference
│   ├── fimmia.md
│   └── shift_detection.md
├── tutorials/            # Tutorials
│   ├── training.md
│   ├── inference.md
│   └── examples.md
└── images/               # Documentation images
    ├── FiMMIA_system_overview.png
    └── FiMMIA_Inference.png
```

## Configuration

The documentation is configured in `mkdocs.yml` at the repository root. This file is compatible with both Zensical and MkDocs 1.x.

## Deployment

The documentation is automatically built and deployed on [Read the Docs](https://readthedocs.org) when changes are pushed to the repository.

See `.readthedocs.yml` in the repository root for Read the Docs configuration.

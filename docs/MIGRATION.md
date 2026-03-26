# Documentation Migration to Zensical

This project uses **Zensical** instead of MkDocs for documentation generation.

## Why Zensical?

As of 2026, MkDocs 2.0 introduced breaking changes that are incompatible with Material for MkDocs:
- No plugin system
- Breaking theme changes
- Incompatible with Material for MkDocs
- New TOML configuration format
- MkDocs 1.x is unmaintained

**Zensical** is a backward-compatible alternative designed to work with existing MkDocs projects and Material for MkDocs. It maintains compatibility with:
- Existing `mkdocs.yml` configuration files
- Material for MkDocs theme
- All Python Markdown extensions
- Project structure and URLs

## What Changed

- **Build tool**: `mkdocs` → `zensical`
- **Build commands**: `mkdocs build` → `zensical build`, `mkdocs serve` → `zensical serve`
- **Configuration**: `mkdocs.yml` remains unchanged (fully compatible)

## Local Development

```bash
# Install dependencies
uv pip install -r docs/requirements.txt

# Build documentation
zensical build

# Serve locally
zensical serve
```

## Read the Docs

Read the Docs should automatically detect and use Zensical when building. The `.readthedocs.yml` file has been updated to use `zensical build` instead of `mkdocs build`.

**Note**: If Zensical is not yet available on Read the Docs, you may need to:
1. Use a custom build environment
2. Or temporarily use MkDocs 1.x (though it's unmaintained)

Check the [Zensical documentation](https://zensical.com) for the latest installation and deployment instructions.

## References

- [MkDocs 2.0 Announcement](https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/)
- [Zensical Documentation](https://zensical.com) (when available)

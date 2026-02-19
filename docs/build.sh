#!/bin/bash
# Build script for local documentation development

set -e

echo "Installing documentation dependencies..."
pip install -r docs/requirements.txt

echo "Building documentation..."
zensical build

echo "Documentation built successfully in 'site/' directory"
echo "To preview, run: zensical serve"

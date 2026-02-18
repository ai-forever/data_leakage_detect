#!/usr/bin/env python
"""
Legacy job launcher for backward compatibility.

DEPRECATED: This script is maintained for backward compatibility.
For new code, please use the structured CLI:
  - fimmia <command> [args...]  (e.g., fimmia train --train_dataset_path=...)
  - shift-attack [args...]       (e.g., shift-attack --dataset=bookmia --attack=bag_of_words)

See README.md for more information.
"""
import importlib
import argparse
import sys
import warnings
from pathlib import Path


sys.path.append(str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy job runner (deprecated - use structured CLI instead)",
        epilog=(
            "Note: This script is deprecated. Use 'fimmia <command>' or 'shift-attack' instead. "
            "See README.md for migration guide."
        ),
    )
    parser.add_argument("--script", type=str, required=True, help="path to script")
    return parser.parse_known_args()[0]


def main():
    # Show deprecation warning
    warnings.warn(
        "job_launcher.py is deprecated. Please use the structured CLI instead:\n"
        "  - fimmia <command> [args...] for FiMMIA operations\n"
        "  - shift-attack [args...] for shift detection attacks\n"
        "See README.md for details.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    args = parse_args()
    module = importlib.import_module(args.script)
    module.main()


if __name__ == "__main__":
    main()

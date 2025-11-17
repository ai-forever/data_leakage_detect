#!/usr/bin/env python
import importlib
import argparse
import sys
import os


sys.path.append(os.path.split(__file__)[0])


def parse_args():
    parser = argparse.ArgumentParser(description=f"job runner arguments")
    parser.add_argument("--script", type=str, required=True, help="path to script")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    module = importlib.import_module(args.script)
    module.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
FiMMIA Command Line Interface

Structured CLI for FiMMIA operations with subcommands.
"""

import sys
import argparse
import importlib


def _import_and_run(module_path, args):
    """Import a module and run its main() function with remaining args."""
    module = importlib.import_module(module_path)
    original_argv = sys.argv[:]
    sys.argv = [module_path] + args
    module.main()
    sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(
        description="FiMMIA: Framework for Multimodal Membership Inference Attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train FiMMIA attack model", allow_abbrev=False
    )
    train_parser.set_defaults(
        module="fimmia.train",
    )

    # Inference command
    infer_parser = subparsers.add_parser(
        "infer", help="Run FiMMIA inference", allow_abbrev=False
    )
    infer_parser.set_defaults(
        module="fimmia.fimmia_inference",
    )

    # Neighbors command
    neighbors_parser = subparsers.add_parser(
        "neighbors", help="Generate neighbors", allow_abbrev=False
    )
    neighbors_parser.set_defaults(
        module="fimmia.neighbors",
    )

    # Embeddings command
    embeds_parser = subparsers.add_parser(
        "embeds", help="Generate embeddings", allow_abbrev=False
    )
    embeds_parser.set_defaults(
        module="fimmia.embeds_joint",
    )

    # Loss calculation command
    loss_parser = subparsers.add_parser(
        "loss", help="Compute losses", allow_abbrev=False
    )
    loss_subparsers = loss_parser.add_subparsers(dest="modality", help="Modality type")

    loss_image_parser = loss_subparsers.add_parser(
        "image", help="Compute losses for image modality", allow_abbrev=False
    )
    loss_image_parser.set_defaults(module="fimmia.image.loss_calc")

    loss_audio_parser = loss_subparsers.add_parser(
        "audio", help="Compute losses for audio modality", allow_abbrev=False
    )
    loss_audio_parser.set_defaults(module="fimmia.audio.loss_calc_qwen2")

    loss_video_parser = loss_subparsers.add_parser(
        "video", help="Compute losses for video modality", allow_abbrev=False
    )
    loss_video_parser.set_defaults(module="fimmia.video.loss_calc_qwen25")

    # SFT finetuning command
    sft_parser = subparsers.add_parser(
        "sft", help="SFT-LoRA finetuning", allow_abbrev=False
    )
    sft_subparsers = sft_parser.add_subparsers(dest="modality", help="Modality type")

    sft_image_parser = sft_subparsers.add_parser(
        "image", help="SFT finetuning for image", allow_abbrev=False
    )
    sft_image_parser.set_defaults(module="fimmia.sft_finetune_image")

    sft_video_parser = sft_subparsers.add_parser(
        "video", help="SFT finetuning for video", allow_abbrev=False
    )
    sft_video_parser.set_defaults(module="fimmia.video.train_qwen25vl")

    sft_audio_parser = sft_subparsers.add_parser(
        "audio", help="SFT finetuning for audio", allow_abbrev=False
    )
    sft_audio_parser.set_defaults(module="fimmia.audio.train_qwen2")

    # Attribute command
    attribute_parser = subparsers.add_parser(
        "attribute", help="Gradient attribution", allow_abbrev=False
    )
    attribute_parser.set_defaults(module="fimmia.attribute_fimmia")

    # MDS dataset command
    mds_parser = subparsers.add_parser(
        "mds-dataset", help="Prepare MDS dataset", allow_abbrev=False
    )
    mds_parser.set_defaults(module="fimmia.utils.mds_dataset")

    # Parse known args to get command
    args, remaining = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Get the module path
    module_path = args.module

    # Run the module's main() with remaining arguments
    _import_and_run(module_path, remaining)


if __name__ == "__main__":
    main()

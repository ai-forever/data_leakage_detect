"""
Unified text + modality embedding pipeline.

Computes text embeddings and optional modality embeddings via the embedding_models
abstraction (BaseEmbedder) in a single pass over the dataset, and writes
embeds/part_*.csv and, when a modality column is present, {modality_key}_embeds/part_*.csv
so that existing MDS and training code works unchanged.
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

from fimmia.embedding_models import (
    BaseEmbedder,
    get_default_modality_embedder,
    get_default_text_embedder,
)


@dataclass
class Args:
    df_path: str
    # Text encoder
    embed_model: str = "intfloat/e5-mistral-7b-instruct"
    max_seq_length: int = 4096
    user_answer: int = 0
    # Modality (optional; when absent, only text embeds are produced)
    modality_key: Optional[str] = None  # e.g. "image", "video", "audio"
    device: str = "cuda"
    # Output
    part_size: int = 5000
    run_single_file: int = 1


def prc_df(
    df_path: str,
    text_embedder: BaseEmbedder,
    part_size: int,
    user_answer: int,
    modality_key: Optional[str],
    device: str,
    save_dir: Optional[str] = None,
    modality_embedder: Optional[BaseEmbedder] = None,
) -> List[str]:
    """
    Single pass: for each (row, neighbor) compute text and optional modality
    embeddings, then write embeds/part_*.csv and, if modality_key present,
    {modality_key}_embeds/part_*.csv with aligned row counts.
    """
    df = pd.read_csv(df_path)
    base_dir = df_path[:-4] if save_dir is None else save_dir
    embeds_dir = os.path.join(base_dir, "embeds")
    os.makedirs(embeds_dir, exist_ok=True)

    has_modality = (
        modality_key is not None
        and str(modality_key).strip()
        and modality_key in df.columns
    )
    if has_modality:
        modality_embeds_dir = os.path.join(base_dir, f"{modality_key}_embeds")
        os.makedirs(modality_embeds_dir, exist_ok=True)
        if modality_embedder is None:
            modality_embedder = get_default_modality_embedder(device=device)
    else:
        modality_embeds_dir = None
        modality_embedder = None

    lines_text: List[dict] = []
    lines_modality: List[dict] = []  # same length as lines_text when has_modality
    num_part = 0
    res: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"prc {df_path}"):
        new_row = dict(row)
        neighbors = eval(new_row.pop("neighbors"))

        row_input = row.input + " " + row.answer
        input_embeds = text_embedder.embed_text(row_input)
        if has_modality and modality_embedder is not None:
            mod_path = row[modality_key]
            input_mod_embeds = modality_embedder.embed_modality_batch(
                modality_key, [mod_path]
            )[0]
        is_add = True

        for neighbor in set(neighbors):
            line = deepcopy(new_row)
            line["neighbor"] = neighbor
            if user_answer:
                neighbor_text = row.input + " " + neighbor
            else:
                neighbor_text = neighbor + " " + row.answer
            line["neighbor_embeds"] = text_embedder.embed_text(neighbor_text)
            first_in_row = is_add
            if is_add:
                is_add = False
                line["input_embeds"] = input_embeds
            else:
                line["input_embeds"] = None
            lines_text.append(line)

            if has_modality and modality_embedder is not None:
                neighbor_mod_embeds = modality_embedder.embed_modality_batch(
                    modality_key, [mod_path]
                )[0]
                line_mod = {
                    f"neighbor_{modality_key}_embeds": neighbor_mod_embeds,
                    f"input_{modality_key}_embeds": input_mod_embeds
                    if first_in_row
                    else None,
                }
                lines_modality.append(line_mod)

        if part_size < len(lines_text):
            print("Save part:", num_part, "Saved:", len(lines_text))
            fp = os.path.join(embeds_dir, f"part_{num_part}.csv")
            pd.DataFrame(lines_text).to_csv(fp, index=False)
            res.append(fp)
            if has_modality:
                fp_mod = os.path.join(modality_embeds_dir, f"part_{num_part}.csv")
                pd.DataFrame(lines_modality).to_csv(fp_mod, index=False)
            num_part += 1
            lines_text = []
            lines_modality = []

    if len(lines_text):
        print("Save part:", num_part, "Saved:", len(lines_text))
        fp = os.path.join(embeds_dir, f"part_{num_part}.csv")
        pd.DataFrame(lines_text).to_csv(fp, index=False)
        res.append(fp)
        if has_modality:
            fp_mod = os.path.join(modality_embeds_dir, f"part_{num_part}.csv")
            pd.DataFrame(lines_modality).to_csv(fp_mod, index=False)

    return res


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    text_embedder = get_default_text_embedder(
        model_name=args.embed_model,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    # Load modality embedder once before dataset processing when modality is used
    modality_embedder: Optional[BaseEmbedder] = None
    if args.modality_key and str(args.modality_key).strip():
        modality_embedder = get_default_modality_embedder(device=args.device)

    save_dir = f"{args.df_path[:-4]}_ng_parts"

    res: List[str] = []
    if args.run_single_file:
        processed_paths = prc_df(
            df_path=args.df_path,
            text_embedder=text_embedder,
            part_size=args.part_size,
            user_answer=args.user_answer,
            modality_key=args.modality_key,
            device=args.device,
            modality_embedder=modality_embedder,
        )
        res.extend(processed_paths)
    else:
        for df_path in glob(f"{save_dir}/*.csv"):
            processed_paths = prc_df(
                df_path=df_path,
                text_embedder=text_embedder,
                part_size=args.part_size,
                user_answer=args.user_answer,
                modality_key=args.modality_key,
                device=args.device,
                modality_embedder=modality_embedder,
            )
            res.extend(processed_paths)

    print("Processed:")
    for path in res:
        print(path)


if __name__ == "__main__":
    main()

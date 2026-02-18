"""
Unified text + modality embedding pipeline.

Computes text embeddings and optional modality embeddings via the embedding_models
abstraction (BaseEmbedder) in a single pass over the dataset, and writes
embeds/part_*.csv and, when a modality column is present, {modality_key}_embeds/part_*.csv
so that existing MDS and training code works unchanged.
"""

from pathlib import Path
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
    # If 1, explicitly ignore any modality neighbor information and always use the
    # original modality input for all neighbors. Currently kept for API symmetry
    # with the loss calculators; embeds_joint does not use modality_neighbors.
    ignore_modality_neighbors: int = 0
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
    ignore_modality_neighbors: int = 0,
) -> List[str]:
    """
    Single pass: for each (row, neighbor) compute text and optional modality
    embeddings, then write embeds/part_*.csv and, if modality_key present,
    {modality_key}_embeds/part_*.csv with aligned row counts.
    """
    df = pd.read_csv(df_path)
    base_dir = Path(df_path).with_suffix("") if save_dir is None else Path(save_dir)
    embeds_dir = base_dir / "embeds"
    embeds_dir.mkdir(parents=True, exist_ok=True)

    has_modality = (
        modality_key is not None
        and str(modality_key).strip()
        and modality_key in df.columns
    )
    if has_modality:
        modality_embeds_dir = base_dir / f"{modality_key}_embeds"
        modality_embeds_dir.mkdir(parents=True, exist_ok=True)
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

        # Prepare modality information for this row: we always embed the original
        # modality input once for the "input_*_embeds" column, and for neighbors
        # we either:
        #   * use per-neighbor modality paths from `modality_neighbors` when
        #     available and ignore_modality_neighbors == 0, or
        #   * fall back to the original modality path for all neighbors.
        if has_modality and modality_embedder is not None:
            base_mod_path = row[modality_key]
            input_mod_embeds = modality_embedder.embed_modality_batch(
                modality_key, [base_mod_path]
            )[0]

            mod_list = None
            if (
                not ignore_modality_neighbors
                and "modality_neighbors" in row
                and pd.notna(row.get("modality_neighbors"))
            ):
                try:
                    raw_mod_list = (
                        eval(row["modality_neighbors"])
                        if isinstance(row["modality_neighbors"], str)
                        else row["modality_neighbors"]
                    )
                    if isinstance(raw_mod_list, list) and len(raw_mod_list) == len(
                        neighbors
                    ):
                        mod_list = raw_mod_list
                except Exception:
                    mod_list = None

            if mod_list is not None:
                neighbor_specs = [(n, mod_list[i]) for i, n in enumerate(neighbors)]
            else:
                neighbor_specs = [(n, base_mod_path) for n in set(neighbors)]
        else:
            base_mod_path = None
            input_mod_embeds = None
            neighbor_specs = [(n, None) for n in set(neighbors)]

        is_add = True

        for neighbor, neighbor_mod_path in neighbor_specs:
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
                    modality_key, [neighbor_mod_path]
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
            fp = embeds_dir / f"part_{num_part}.csv"
            pd.DataFrame(lines_text).to_csv(str(fp), index=False)
            res.append(str(fp))
            if has_modality:
                fp_mod = modality_embeds_dir / f"part_{num_part}.csv"
                pd.DataFrame(lines_modality).to_csv(str(fp_mod), index=False)
            num_part += 1
            lines_text = []
            lines_modality = []

    if len(lines_text):
        print("Save part:", num_part, "Saved:", len(lines_text))
        fp = embeds_dir / f"part_{num_part}.csv"
        pd.DataFrame(lines_text).to_csv(str(fp), index=False)
        res.append(str(fp))
        if has_modality:
            fp_mod = modality_embeds_dir / f"part_{num_part}.csv"
            pd.DataFrame(lines_modality).to_csv(str(fp_mod), index=False)

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
            ignore_modality_neighbors=args.ignore_modality_neighbors,
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
                ignore_modality_neighbors=args.ignore_modality_neighbors,
            )
            res.extend(processed_paths)

    print("Processed:")
    for path in res:
        print(path)


if __name__ == "__main__":
    main()

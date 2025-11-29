from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from transformers import HfArgumentParser
from sentence_transformers import SentenceTransformer
from glob import glob
import torch
import os
import pandas as pd


@dataclass
class Args:
    df_path: str
    embed_model: str = "intfloat/e5-mistral-7b-instruct"
    part_size: int = 5000
    max_seq_length: int = 4096
    run_single_file: int = 1
    user_answer: int = 0


def get_embeds(text, embed_model):
    with torch.inference_mode():
        embeds = embed_model.encode([text]).tolist()[0]
    return embeds


def prc_df(df_path, embed_model, part_size=1000, save_dir=None, user_answer=0):
    if save_dir is None:
        save_dir = os.path.join(df_path[:-4], "embeds")
        os.makedirs(save_dir, exist_ok=True)
    lines = []
    df = pd.read_csv(df_path)
    num_part = 0
    res = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"prc {df_path}"):
        new_row = dict(row)
        neighbors = eval(new_row.pop("neighbors"))
        row_input = row.input + " " + row.answer
        input_embeds = get_embeds(row_input, embed_model)
        is_add = True
        for neighbor in set(neighbors):
            line = deepcopy(new_row)
            line["neighbor"] = neighbor
            if user_answer:
                neighbor = row.input + " " + neighbor
            else:
                neighbor = neighbor + " " + row.answer
            line["neighbor_embeds"] = get_embeds(neighbor, embed_model)
            if is_add:
                is_add = False
                line["input_embeds"] = input_embeds
            else:
                line["input_embeds"] = None
            lines.append(line)
        if part_size < len(lines):
            print("Save part:", num_part, "Saved:", len(lines))
            fp = os.path.join(save_dir, f"part_{num_part}.csv")
            pd.DataFrame(lines).to_csv(fp, index=False)
            res.append(fp)
            num_part += 1
            lines = []
    if len(lines):
        print("Save part:", num_part, "Saved:", len(lines))
        fp = os.path.join(save_dir, f"part_{num_part}.csv")
        pd.DataFrame(lines).to_csv(fp, index=False)
        res.append(fp)
    return res


def main():
    parser = HfArgumentParser((Args, ))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    embed_model = SentenceTransformer(args.embed_model)
    embed_model.max_seq_length = args.max_seq_length

    save_dir = f"{args.df_path[:-4]}_ng_parts"

    res = []
    if args.run_single_file:
        processed_paths = prc_df(
            df_path=args.df_path,
            embed_model=embed_model,
            part_size=args.part_size,
            user_answer=args.user_answer
        )
        res.extend(processed_paths)
    else:
        for df_path in glob(f"{save_dir}/*.csv"):
            processed_paths = prc_df(
                df_path=df_path,
                embed_model=embed_model,
                part_size=args.part_size,
                user_answer=args.user_answer
            )
            res.extend(processed_paths)

    print("Processed:")
    for path in res:
        print(path)


if __name__ == "__main__":
    main()

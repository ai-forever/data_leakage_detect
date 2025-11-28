from glob import glob
from tqdm import tqdm
from streaming.base import MDSWriter
from fimmia.utils.mds_dataset import get_streaming_ds
from fimmia.utils.utils import load_json
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from transformers import default_data_collator
import numpy as np
import pandas as pd
import os


def merge_part(embeds_dir, image_embeds_dir, loss_dir, num_part, model_name, label=0):
    df_embeds = pd.read_csv(os.path.join(embeds_dir, f"part_{num_part}.csv"))
    df_image_embeds = pd.read_csv(os.path.join(image_embeds_dir, f"part_{num_part}.csv"))
    df_loss = pd.read_csv(os.path.join(loss_dir, f"part_{num_part}.csv"))
    assert len(df_loss) == len(df_image_embeds) == len(df_embeds)
    df_loss["label"] = label
    df_loss["neighbor_embeds"] = df_embeds.neighbor_embeds
    df_loss["input_embeds"] = df_embeds.input_embeds
    df_loss["neighbor_image_embeds"] = df_image_embeds.neighbor_image_embeds
    df_loss["input_image_embeds"] = df_image_embeds.input_image_embeds
    df_loss["model_name"] = model_name
    df_loss["hash"] = [
        str(hash(f"{x.input}{x.answer}{x.image}{x.ds_name}{x.label}{model_name}")) for _, x in df_loss.iterrows()]
    df_loss["num_part"] = num_part
    return df_loss


def iterate_df_parts(path, model_name, label):
    embeds_dir = os.path.join(path[:-4], "embeds")
    image_embeds_dir = os.path.join(path[:-4], "image_embeds")
    loss_dir = os.path.join(path[:-4], f"loss/{model_name}")
    num_parts = len(glob(f"{embeds_dir}/*.csv"))
    for num_part in tqdm(range(num_parts), total=num_parts, desc=f"prc df {path}"):
        part = merge_part(embeds_dir, image_embeds_dir, loss_dir, num_part, model_name, label=label)
        yield part


def prepare_row(row):
    if isinstance(row.input_embeds, str):
        row.input_embeds = eval(row.input_embeds)
    if isinstance(row.neighbor_embeds, str):
        row.neighbor_embeds = eval(row.neighbor_embeds)
    if isinstance(row.neighbor_image_embeds, str):
        row.neighbor_image_embeds = eval(row.neighbor_image_embeds)
    if isinstance(row.input_image_embeds, str):
        row.input_image_embeds = eval(row.input_image_embeds)
    row.input_embeds = np.array(row.input_embeds, dtype=float)
    row.neighbor_embeds = np.array(row.neighbor_embeds, dtype=float)
    row.input_image_embeds = np.array(row.input_image_embeds, dtype=float)
    row.neighbor_image_embeds = np.array(row.neighbor_image_embeds, dtype=float)
    return row


def fix_row(input_row, row):
    row.input_loss = input_row.input_loss
    row.input_image_embeds = input_row.input_image_embeds
    row.input_embeds = input_row.input_embeds
    return row


def row_to_sample(row, sigmas):
    key = f'{row.ds_name}_{row.model_name}'
    line = {
        "label": row.label,
        "loss_input": row.input_loss - row.neighbor_loss,
        "embedding_input": row.input_embeds - row.neighbor_embeds,
        "image_embedding_input": row.input_image_embeds - row.neighbor_image_embeds,
        "ds_name": row.ds_name,
        "neighbor": row.neighbor,
        "input": row.input,
        "answer": row.answer,
        "image": row.image,
        "hash": row.hash,
        "model_name": row.model_name,
        "num_part": row.num_part,
        "mean": sigmas.get(key, {"mean": 0})["mean"],
        "std": sigmas.get(key, {"std": 1})["std"],
    }
    return line


def write_part(df, ds_writer, sigmas):
    for id_group, (_, group) in enumerate(df.groupby("hash")):
        input_row = prepare_row(group.iloc[0])
        rows = []
        for _, row in group.iterrows():
            rows.append(fix_row(input_row=input_row, row=row))
            row = prepare_row(row)
            line = row_to_sample(row, sigmas)
            ds_writer.write(line)


def build_mds(origin_ds_path, save_dir, model_names, labels, single_file=False, sigmas=None):
    columns = {
        "label": "int32",
        "loss_input": "float16",
        "mean": "float16",
        "std": "float16",
        "embedding_input": "ndarray",
        "image_embedding_input": "ndarray",
        "ds_name": "str",
        "neighbor": "str",
        "input": "str",
        "answer": "str",
        "image": "str",
        "hash": "str",
        "model_name": "str",
        "num_part": "int32"
    }
    if sigmas is None:
        sigmas = {}
    ds_name = os.path.split(origin_ds_path)[-1][:-4]
    ds_dir = os.path.split(origin_ds_path)[0]
    if single_file:
        ds_file_paths = [origin_ds_path]
    else:
        ds_file_paths = glob(os.path.join(ds_dir, f"{ds_name}_ng_parts/*.csv"))
    with MDSWriter(out=save_dir, columns=columns, exist_ok=True) as ds_writer:
        for df_path in tqdm(ds_file_paths, total=len(ds_file_paths)):
            for model_name, label in zip(model_names, labels):
                for df in iterate_df_parts(df_path, model_name, label):
                    write_part(df=df, ds_writer=ds_writer, sigmas=sigmas)
                print("-------------------------------\n")


def get_mean_std(path):
    ds = get_streaming_ds(path, shuffle=False)
    losses = defaultdict(list)
    for x in ds:
        key = f'{x["ds_name"]}_{x["model_name"]}'
        losses[key].append(x["loss_input"])
    sigmas = defaultdict(dict)
    for key, loss in losses.items():
        loss = np.array(loss)
        scaler = StandardScaler()
        scaler.fit(loss.reshape(-1, 1))
        sigmas[key]["mean"] = float(scaler.mean_[0])
        sigmas[key]["std"] = float(scaler.var_[0])
    return sigmas


def get_min_max(path):
    ds = get_streaming_ds(path, shuffle=False)
    losses = defaultdict(list)
    for x in ds:
        key = f'{x["ds_name"]}_{x["model_name"]}'
        losses[key].append(x["loss_input"])
    sigmas = defaultdict(dict)
    for key, loss in losses.items():
        loss = np.array(loss)
        sigmas[key]["min_loss"] = min(loss)
        sigmas[key]["loss_diff"] = max(loss) - sigmas[key]["min_loss"]
    return sigmas


def create_data_collator_std(sigmas_path, use_image=False):
    sigmas = load_json(sigmas_path)

    def data_collator(features):
        new_features = []
        for x in features:
            key = f'{x["ds_name"]}_{x["model_name"]}'
            if key not in sigmas:
                print("No key", key, "in", sigmas_path, "Use default.")
                key = f'{x["ds_name"]}_no_leak'
            mean = sigmas[key]["mean"]
            std = sigmas[key]["std"]
            new_features.append({
                "loss_input": x["loss_input"],
                "embedding_input": x["embedding_input"],
                "label": x["label"],
                "mean": mean,
                "std": std
            })
            if use_image:
                new_features[-1]["image_embedding_input"] = x["image_embedding_input"]
        return default_data_collator(new_features)

    return data_collator


def create_data_collator_minmax(sigmas_path, use_image=False):
    sigmas = load_json(sigmas_path)

    def data_collator(features):
        new_features = []
        for x in features:
            key = f'{x["ds_name"]}_{x["model_name"]}'
            if key not in sigmas:
                print("No key", key, "in", sigmas_path, "Use default.")
                key = f'{x["ds_name"]}_no_leak'
            min_loss = sigmas[key]["min_loss"]
            loss_diff = sigmas[key]["loss_diff"]
            new_features.append({
                "loss_input": x["loss_input"],
                "embedding_input": x["embedding_input"],
                "label": x["label"],
                "min_loss": min_loss,
                "loss_diff": loss_diff
            })
            if use_image:
                new_features[-1]["image_embedding_input"] = x["image_embedding_input"]
        return default_data_collator(new_features)

    return data_collator


def create_default_data_collator(use_image=False):

    def baseline_data_collator(features):
        new_features = []
        for x in features:
            new_features.append({
                "loss_input": x["loss_input"],
                "embedding_input": x["embedding_input"],
                "label": x["label"],
            })
            if use_image:
                new_features[-1]["image_embedding_input"] = x["image_embedding_input"]
        return default_data_collator(new_features)
    return baseline_data_collator


def create_data_collator(model_name, sigmas_path=None, sigmas_type=None):
    model_name = model_name.lower()
    if "baseline" in model_name:
        use_image = False
    else:
        use_image = True

    if sigmas_type == "std":
        data_collator = create_data_collator_std(sigmas_path, use_image)
    elif sigmas_type == "linear":
        data_collator = create_data_collator_minmax(sigmas_path, use_image)
    else:
        data_collator = create_default_data_collator(use_image)
    return data_collator

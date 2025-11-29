from streaming.base import MDSWriter, StreamingDataset, Stream
from torchdata.datapipes.iter import IterableWrapper
from glob import glob
from tqdm import tqdm
from dataclasses import dataclass
from transformers import HfArgumentParser
from fimmia.utils.utils import load_json
import os
import pandas as pd
import numpy as np


@dataclass
class DatasetBuilderArguments:
    save_dir: str
    model_name: str
    origin_df_path: str
    labels: str = "0"
    shuffle: int = 0
    modality_key: str = "image"
    single_file: int = 0
    sigmas_fields: str = None
    sigmas_path: str = None


def get_streaming_ds(paths, shuffle=False):
    streams = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        streams.append(Stream(local=path))
    ds = StreamingDataset(streams=streams, batch_size=1, shuffle=shuffle)
    ds = IterableWrapper(ds)
    return ds


class MDSDataset:
    def __init__(
            self, mds_dataset_paths, model_name, labels,
            origin_df_path=None, shuffle=False, modality_key="image", sigmas_fields=None
    ):
        self.mds_dataset_paths = mds_dataset_paths
        if isinstance(self.mds_dataset_paths, str):
            self.mds_dataset_paths = [self.mds_dataset_paths]
        self.origin_df_path = origin_df_path
        self.stream = None
        self.shuffle = shuffle
        self.modality_key = modality_key
        self.model_name = model_name
        self.labels = list(map(int, labels.split(",")))
        self.sigmas_fields = sigmas_fields
        if self.sigmas_fields is not None:
            self.sigmas_fields = self.sigmas_fields.split(",")
        self.columns = {
            "label": "int32",
            "ds_name": "str",
            "neighbor": "str",
            "input": "str",
            "answer": "str",
            modality_key: "str",
            "loss_input": "float16",
            "embedding_input": "ndarray",
            "hash": "str",
            "model_name": "str",
            "num_part": "int32"
        }
        if self.sigmas_fields is not None:
            for key in self.sigmas_fields:
                self.columns[key] = "float16"
        self.neighbor_modality_key = f"neighbor_{self.modality_key}_embeds"
        self.input_modality_key = f"input_{self.modality_key}_embeds"
        self.modality_exists = False

    def load(self):
        self.stream = get_streaming_ds(self.mds_dataset_paths, shuffle=False)

    def iterate_df_parts(self, df_path, label):
        working_dir = df_path[:-4]
        embeds_dir = os.path.join(working_dir, "embeds")
        modality_embeds_dir = os.path.join(working_dir, f"{self.modality_key}_embeds")
        loss_dir = os.path.join(working_dir, "loss", self.model_name, "leak" if label else "no_leak")
        num_parts = len(glob(f"{embeds_dir}/*.csv"))
        for num_part in tqdm(range(num_parts), total=num_parts, desc=f"prc df {df_path}"):
            yield self.merge_part(embeds_dir, modality_embeds_dir, loss_dir, num_part, label)

    def merge_part(self, embeds_dir, modality_embeds_dir, loss_dir, num_part, label):
        df_embeds = pd.read_csv(os.path.join(embeds_dir, f"part_{num_part}.csv"))

        df_loss = pd.read_csv(os.path.join(loss_dir, f"part_{num_part}.csv"))
        df_loss["label"] = label
        df_loss["neighbor_embeds"] = df_embeds.neighbor_embeds
        df_loss["input_embeds"] = df_embeds.input_embeds
        df_loss["model_name"] = self.model_name
        df_loss["hash"] = [
            str(hash(f"{x.input}{x.answer}{self.get_modality_input(x)}{x.ds_name}{x.label}{self.model_name}"))
            for _, x in df_loss.iterrows()]
        df_loss["num_part"] = num_part

        modality_embeds_path = os.path.join(modality_embeds_dir, f"part_{num_part}.csv")
        if self.modality_exists or os.path.exists(modality_embeds_path):
            self.modality_exists = True
            self.columns[f"{self.modality_key}_embedding_input"] = "ndarray"
            df_modality_embeds = pd.read_csv(modality_embeds_path)
            df_loss[self.neighbor_modality_key] = df_modality_embeds[self.neighbor_modality_key]
            df_loss[self.input_modality_key] = df_modality_embeds[self.input_modality_key]

        return df_loss

    def prepare_row(self, row):
        if isinstance(row["input_embeds"], str):
            row["input_embeds"] = eval(row["input_embeds"])
        if isinstance(row["neighbor_embeds"], str):
            row["neighbor_embeds"] = eval(row["neighbor_embeds"])
        if self.modality_exists:
            if isinstance(row[self.neighbor_modality_key], str):
                row[self.neighbor_modality_key] = eval(row[self.neighbor_modality_key])
            row[self.neighbor_modality_key] = np.array(row[self.neighbor_modality_key], dtype=float)
            if isinstance(row[self.input_modality_key], str):
                row[self.input_modality_key] = eval(row[self.input_modality_key])
            row[self.input_modality_key] = np.array(row[self.input_modality_key], dtype=float)
        row["input_embeds"] = np.array(row["input_embeds"], dtype=float)
        row["neighbor_embeds"] = np.array(row["neighbor_embeds"], dtype=float)

        return row

    def fix_row(self, input_row, row):
        row.input_loss = input_row.input_loss
        if self.modality_exists:
            row[self.input_modality_key] = input_row[self.input_modality_key]
        row.input_embeds = input_row.input_embeds
        return row

    def get_modality_input(self, row):
        if self.modality_key in ["image", "video"]:
            modality_input = row[self.modality_key]
        else:
            modality_input = row.get("audio")
            if isinstance(modality_input, float) or modality_input is None:
                modality_input = row.get("audio_1")
            if isinstance(modality_input, float) or modality_input is None:
                modality_input = row.get("audio_2")
        return modality_input

    def row_to_sample(self, row, sigmas):
        line = {
            "label": row.label,
            "loss_input": row.neighbor_loss - row.input_loss,
            "embedding_input": row.neighbor_embeds - row.input_embeds,
            "ds_name": row.ds_name,
            "neighbor": row.neighbor,
            "input": row.input,
            "answer": row.answer,
            self.modality_key: self.get_modality_input(row),
            "hash": row.hash,
            "model_name": row.model_name,
            "num_part": row.num_part,
            # "mean": sigmas.get(key, {"mean": 0})["mean"],
            # "std": sigmas.get(key, {"std": 1})["std"],
        }
        if self.modality_exists:
            modality_key = f"{self.modality_key}_embedding_input"
            line[modality_key] = row[self.neighbor_modality_key] - row[self.input_modality_key]
        if sigmas is not None:
            key = f'{row.ds_name}_{row.model_name}'
            for key in self.sigmas_fields:
                line[key] = sigmas.get(key, {key: 1})[key]
        return line

    def build(self, single_file=False, sigmas=None):
        save_dir = self.mds_dataset_paths[0]
        ds_name = os.path.split(self.origin_df_path)[-1][:-4]

        if single_file:
            ds_file_paths = [self.origin_df_path]
        else:
            ds_file_paths = glob(os.path.join(os.path.split(self.origin_df_path)[0], f"{ds_name}_ng_parts/*.csv"))
        with MDSWriter(out=save_dir, columns=self.columns, exist_ok=True) as ds_writer:
            pbar = tqdm(ds_file_paths, total=len(ds_file_paths))
            for df_path in pbar:
                pbar.set_description(f"process: {df_path}")
                for label in self.labels:
                    for df in self.iterate_df_parts(df_path, label):
                        self.write_part(df=df, ds_writer=ds_writer, sigmas=sigmas)

    def write_part(self, df, ds_writer, sigmas):
        for _, group in df.groupby("hash"):
            input_row = self.prepare_row(group.iloc[0])
            rows = []
            for _, row in group.iterrows():
                rows.append(self.fix_row(input_row=input_row, row=row))
                row = self.prepare_row(row)
                line = self.row_to_sample(row, sigmas)
                ds_writer.write(line)


def main():
    parser = HfArgumentParser((DatasetBuilderArguments, ))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    sigmas = None
    if args.sigmas_path is not None:
        sigmas = load_json(args.sigmas_path)
    dataset = MDSDataset(
        mds_dataset_paths=args.save_dir, model_name=args.model_name,
        labels=args.labels, origin_df_path=args.origin_df_path,
        shuffle=args.shuffle, modality_key=args.modality_key, sigmas_fields=args.sigmas_fields
    )
    dataset.build(single_file=args.single_file, sigmas=sigmas)


if __name__ == "__main__":
    main()

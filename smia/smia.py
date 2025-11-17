from collections import defaultdict
from tqdm import tqdm
from smia.smia_models import init_model
from transformers import default_data_collator, Trainer, TrainingArguments
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset
from dataclasses import dataclass
from glob import glob
import numpy as np
import torch
import pandas as pd


@dataclass
class ModelArguments:
    model_name: str = "BaseLineModel"
    embedding_size: int = 4096
    projection_size: int = 512
    image_embedding_size: int = 1024
    model_path: str = None
    sigmas_path: str = None
    sigmas_type: str = None


class SMIABatchDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self._size = len(examples)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self.examples[idx]

    @classmethod
    def from_rows(cls, group):
        examples = []
        input_loss = None
        input_embeds = None
        for _, row in group.iterrows():
            if input_loss is None:
                input_loss = row.input_loss
                if isinstance(row.input_embeds, str):
                    row.input_embeds = eval(row.input_embeds)
                input_embeds = np.array(row.input_embeds)
            if isinstance(row.neighbor_embeds, str):
                row.neighbor_embeds = eval(row.neighbor_embeds)
            neighbor_embeds = np.array(row.neighbor_embeds)
            line = {
                "label": row.label,
                "loss_input": input_loss - row.neighbor_loss,
                "embedding_input": input_embeds - neighbor_embeds
            }
            examples.append(line)
        return cls(examples)


def iter_batches(df_path: str):
    save_dir = df_path[:-4]
    files = glob(f"{save_dir}/*.csv")
    idx = 0
    for fn in files:
        df = pd.read_csv(fn)
        for _, group_label in df.groupby("input"):
            for _, group in group_label.groupby("label"):
                ds = SMIABatchDataset.from_rows(group)
                yield idx, ds, group
                idx += 1


class SMIA:
    @staticmethod
    def load_data(path: str):
        return iter_batches(path)

    @staticmethod
    def _get_scores(processor, row):
        with torch.inference_mode():
            scores = processor.predict(row).predictions
            return scores

    def get_scores(self, processor, data):
        predictions = []
        for idx, row, group in tqdm(data, leave=True, desc="predict smia scores..."):
            scores = self._get_scores(processor, row)
            group["predictions"] = list(scores)
            predictions.append(group)
        return predictions

    @staticmethod
    def calc_metrics(scores, labels):
        fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr_list, tpr_list)
        fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
        return auroc, fpr95, tpr05

    def _get_metrics(self, scores, labels):
        results = defaultdict(list)
        for method, scores in scores.items():
            auroc, fpr95, tpr05 = self.calc_metrics(scores, labels)

            results['method'].append(method)
            results['auroc'].append(f"{auroc:.1%}")
            results['fpr95'].append(f"{fpr95:.1%}")
            results['tpr05'].append(f"{tpr05:.1%}")

        return results

    def get_metrics(self, scores, labels):
        # 1: training, 0: non-training
        return self._get_metrics(scores, labels)

    def predict(
            self, model_args: ModelArguments,
            df_path: str,
            training_args: TrainingArguments
    ):
        model = init_model(model_args)
        processor = Trainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator
        )
        data = self.load_data(df_path)
        scores = self.get_scores(processor, data)
        return scores


class SMIAMERA(SMIA):
    @staticmethod
    def load_data(parent_df_path: str):
        save_dir = f"{parent_df_path[:-4]}_ng_parts"
        files = glob(f"{save_dir}/*/*.csv")
        idx = 0
        for df_path in files:
            df = pd.read_csv(df_path)
            for _, group_label in df.groupby("input"):
                for _, group in group_label.groupby("label"):
                    ds = SMIABatchDataset.from_rows(group)
                    yield idx, ds, group
                    idx += 1

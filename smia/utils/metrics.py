from collections import defaultdict
from smia.utils.mds_dataset import get_streaming_ds
from smia.smia import SMIA
from sklearn.metrics import accuracy_score
from transformers import HfArgumentParser, Trainer
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


@dataclass
class MetricArgs:
    model_name: str
    model_name_pred: str
    test_path: str
    save_path: str


def get_sample_scores(group_model):
    cscores = []
    y_true = []
    y_pred = []
    labels = []
    for label, group_label in tqdm(group_model.groupby("label"), total=len(group_model)):
        for _, group in group_label.groupby("hash"):
            scc = np.vstack(group.score)
            scc = torch.nn.functional.softmax(torch.Tensor(scc)).numpy()
            y_true.extend(scc.argmax(-1))
            y_pred.extend([label] * len(group))
            cscores.append(np.abs(scc.max(-1) - 1 + scc.argmax(-1)).mean())
            labels.append(label)
    return cscores, labels, y_true, y_pred


def get_sample_scores_by_groups(df_test: pd.DataFrame):
    new_scores = defaultdict(dict)
    new_labels = defaultdict(dict)
    y_true_n = defaultdict(dict)
    y_pred_n = defaultdict(dict)
    for ds_name, group_ds in df_test.groupby("ds_name"):
        for model_name, group_model in group_ds.groupby("model_name"):
            cscores, y_true, y_pred, labels = get_sample_scores(group_model)
            new_scores[ds_name][model_name] = cscores
            new_labels[ds_name][model_name] = labels
            y_true_n[ds_name][model_name] = y_true
            y_pred_n[ds_name][model_name] = y_pred
    return new_scores, new_labels, y_true_n, y_pred_n


def get_metrics_from_scores(new_scores, new_labels, y_true_n, y_pred_n):
    smia = SMIA()
    ds_metrics = []
    # new_scores, new_labels, y_true_n, y_pred_n = get_sample_scores(df_test)
    for ds_name in new_scores:
        scores = []
        labels = []
        y_trues = []
        y_preds = []
        for model_name in new_scores[ds_name]:
            if "Giga" in model_name:
                pass
            else:
                acc = -1
                scores.extend(new_scores[ds_name][model_name])
                labels.extend(new_labels[ds_name][model_name])
                y_trues.extend(y_true_n[ds_name][model_name])
                y_preds.extend(y_pred_n[ds_name][model_name])
        m = {n: x[0] for n, x in dict(smia.get_metrics({"smia": scores}, labels)).items()}
        acc = accuracy_score(y_trues, y_preds)
        m["per_neighbors_acc"] = acc
        m["acc"] = accuracy_score(labels, np.array(scores) > 0.5)
        m["ds_name"] = ds_name
        ds_metrics.append(m)
    ds_metrics = pd.DataFrame(ds_metrics)
    return ds_metrics


def convert_str(s):
    return np.array(list(map(float, s[1:-1].strip().split())))


def get_metrics_from_df(df):
    cscores, labels, y_true, y_pred = get_sample_scores(df)
    smia = SMIA()
    m = {n: x[0] for n, x in dict(smia.get_metrics({"smia": cscores}, labels)).items()}
    m["acc"] = accuracy_score(labels, np.array(cscores) > 0.5)
    m["per_neighbors_acc"] = accuracy_score(y_true, y_pred)
    m = pd.DataFrame([m])
    return m


def get_df_with_predictions(data: str, trainer: Trainer):
    if isinstance(data, str):
        ds_test = get_streaming_ds(data, shuffle=False)
    else:
        ds_test = data
    with torch.inference_mode():
        raw_preds_test = trainer.predict(ds_test)
    lines = []
    for x, score in zip(ds_test, raw_preds_test.predictions):
        lines.append({
            "ds_name": x["ds_name"],
            # "model_name": x["model_name"],
            "hash": x["hash"],
            "label": x["label"],
            "score": score
        })
    df_test = pd.DataFrame(lines)
    return df_test


def main():
    parser = HfArgumentParser((MetricArgs, ))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    df = pd.read_csv(args.test_path)
    df.score = list(map(convert_str, df.score))
    m = get_metrics_from_df(df)
    m.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()

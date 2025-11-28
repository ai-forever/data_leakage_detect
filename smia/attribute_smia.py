from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from transformers import HfArgumentParser
from glob import glob
import torch
import pandas as pd
import numpy as np
import pickle
import pathlib
import seaborn as sns
from matplotlib import pyplot as plt
from smia.train import ModelArguments
from smia.utils.mds_dataset import get_streaming_ds

from captum.attr import (
    IntegratedGradients,
    NoiseTunnel, 
)

@dataclass
class Args:
    model_dir: str
    mds_dataset_path: str
    dataset_name: str = None
    model_cls: str = "BaseLineModelV2"
    embedding_size: int = 1024
    modality_embedding_size: int = 1024
    separate_modality_embeddings: bool = False
    add_attribution_noise: bool = False
    create_graphs: bool = True

@torch.no_grad
def attribute_dataset(model, dataset, add_noise=False):

    def attribute(attribution, loss_input, emb_inputs, label, modality_emb_inputs=None, **attribution_kwargs):
        if modality_emb_inputs is not None:
            inputs = (loss_input.to(device), emb_inputs.to(device), modality_emb_inputs.to(device))
            baselines = (
                torch.zeros_like(loss_input, 
                                dtype=torch.float32, device=device), 
                torch.zeros_like(emb_inputs, 
                    dtype=torch.float32, device=device),
                torch.zeros_like(modality_emb_inputs, 
                    dtype=torch.float32, device=device)
                    )
        else:
            inputs = (loss_input.to(device), emb_inputs.to(device))
            baselines = (
                torch.zeros_like(loss_input, 
                                dtype=torch.float32, device=device), 
                torch.zeros_like(emb_inputs, 
                    dtype=torch.float32, device=device),
                    )
        return attribution.attribute(
            inputs,
            baselines=baselines,
            target=label,
            return_convergence_delta=False,
            **attribution_kwargs
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    attribution = IntegratedGradients(model)
    attribution_kwargs = {"n_steps": 100, }
    if add_noise:
        attribution = NoiseTunnel(attribution)
        attribution_kwargs["nt_type"] = "smoothgrad"
        attribution_kwargs["nt_samples"] = 250
        attribution_kwargs["stdevs"] = 0.02


    attrs_loss_list, attrs_embs_list, attrs_modality_list, labels = [], [], [], []

    for item in tqdm(dataset):
        loss_input, embedding_input = torch.tensor(
            np.array([item["loss_input"]]), 
            dtype=torch.float32,), torch.tensor(
                np.array([item["embedding_input"]]), 
            dtype=torch.float32,), 
        modality_embedding_input = item.get("image_embedding_input", None)
        if modality_embedding_input is not None:
            modality_embedding_input = torch.tensor(
                    np.array([modality_embedding_input]),
                    dtype=torch.float32,) 

        attrs_loss, *attrs_embs = attribute(attribution, loss_input, embedding_input, label=item["label"].item(), modality_emb_inputs=modality_embedding_input, **attribution_kwargs)
        attrs_embs, *attrs_modality = attrs_embs

        attrs_loss_list.append(attrs_loss.item())
        attrs_embs_list.append(attrs_embs.cpu().numpy())
        attrs_modality_list.append(attrs_modality[0].cpu().numpy()) if 0 != len(attrs_modality) else None
        labels.append(item["label"].item())

    attrs_embs_list = np.vstack(attrs_embs_list)
    attrs_modality_list = np.vstack(attrs_modality_list) if 0 != len(attrs_modality_list) else None
    return attrs_loss_list, attrs_embs_list, attrs_modality_list, labels


def sampled_scatterplot(x, y, **kwargs):
    if len(x) > 2000:
        indices = np.random.choice(len(x), size=2000, replace=False)
        x_sampled = np.array(x)[indices]
        y_sampled = np.array(y)[indices]
        sns.scatterplot(x=x_sampled, y=y_sampled, alpha=0.5, s=5, **kwargs)
    else:
        sns.scatterplot(x=x, y=y, alpha=0.5, s=2, **kwargs)

def kdeplot_with_black_contours(x, y, **kwargs):
    # First: colored filled KDE with transparency
    sns.kdeplot(x=x, y=y, fill=True, alpha=0.95, **kwargs)
    # Second: black contour lines on top (no fill, black color)
    kwargs.pop("color")
    sns.kdeplot(x=x, y=y, fill=False, color='black', linewidths=0.3, linestyles="dotted", **kwargs)


def main():
    parser = HfArgumentParser((Args, ))
    args, = parser.parse_args_into_dataclasses()
    
    model_dir=pathlib.Path(args.model_dir)
    model_dir=model_dir.joinpath(args.dataset_name) if args.dataset_name is not None else None

    model_args = ModelArguments(
        model_name=args.model_cls, 
        model_path=model_dir.as_posix(),
        )

    model_args.image_embedding_size = args.modality_embedding_size
    model_args.embedding_size = args.embedding_size

    model = init_model(model_args)

    ds_test = get_streaming_ds(args.mds_dataset_path, )

    attrs_loss_list, attrs_embs_list, attrs_modality_list, labels_list = attribute_dataset(
        model, 
        ds_test, 
        add_noise=args.add_attribution_noise
    )

    save_name = "attributions.pkl"
    if args.add_attribution_noise:
        save_name = "noise_" + save_name

    with open(model_dir.joinpath(save_name), "wb") as f:
        
        pickle.dump(
            (
            attrs_loss_list,
            attrs_embs_list,
            attrs_modality_list,
            labels_list, 
            ), f
        )
    

    if not args.create_graphs:
        return 

    attrs_cheb_norm = np.linalg.norm(attrs_embs_list, ord=np.inf, axis=1)
    attrs_l2_norm = np.linalg.norm(attrs_embs_list, ord=2, axis=1)
    attrs_modal_cheb_norm = np.linalg.norm(attrs_modality_list, ord=np.inf, axis=1) if args.separate_modality_embeddings else None
    attrs_modal_l2_norm = np.linalg.norm(attrs_modality_list, ord=2, axis=1) if args.separate_modality_embeddings else None

    if args.separate_modality_embeddings:
        data = (attrs_loss_list, attrs_cheb_norm, attrs_l2_norm, attrs_modal_cheb_norm, attrs_modal_l2_norm, labels_list)
        # columns = ["Loss",  r"\left \lVert Emb \right \rVert_\infty", r"\left \lVert Emb \right \rVert_2", r"\left \lVert MEmb \right \rVert_\infty", r"\left \lVert MEmb \right \rVert_2", "label"]
        columns = ["Loss",  "Embs, infnorm", "Embs, l2norm", "MEmbs, infnorm", "MEmbs, l2norm", "label"]
    else:
        data = (attrs_loss_list, attrs_cheb_norm, attrs_l2_norm, labels_list)
        # columns = ["Loss",  r"\left \lVert Emb \right \rVert_\infty", r"\left \lVert Emb \right \rVert_2", "label"]
        columns = ["Loss",  "Embs, infnorm", "Embs, l2norm", "label"]

    df = pd.DataFrame.from_records(data).T
    df.columns=columns
    df["label"] = df.label.astype(int)

    plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    })

    g = sns.PairGrid(df, hue="label", palette="magma", )
    g.map_lower(kdeplot_with_black_contours)
    g.map_diag(sns.histplot, element="step", linewidth=0.5, multiple="layer")
    g.map_upper(sampled_scatterplot)
    g.add_legend()

    g.savefig(model_dir.joinpath("attribution_graph.pdf"), bbox_inches="tight", dpi=300)
    return    

if __name__ == "__main__":
    main()

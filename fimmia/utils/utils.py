import torch
import gc
from pathlib import Path
import random
import numpy as np
import pickle
import json
from omegaconf import OmegaConf


def cast_int(val):
    if isinstance(val, str):
        if val.lower() == "true":
            val = 1
        elif val.lower() == "false":
            val = 0
    return int(val)


def load_json(path):
    with open(path, "r") as file:
        text = json.loads(file.read().strip())
    return text


def save_json(obj, path):
    with open(path, "w") as file:
        json.dump(obj, file)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def update_seed(seed):
    torch.manual_seed(seed)
    random.seed(10)
    np.random.seed(seed)


def empty_cache(*args):
    for x in args:
        del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_files_from_dir(dir_path):
    f = []
    dir_path_obj = Path(dir_path)
    for file_path in dir_path_obj.rglob("*"):
        if file_path.is_file():
            f.append(str(file_path))
    return f


def load_yaml(conf):
    if isinstance(conf, str):
        conf = OmegaConf.load(conf)
    return conf

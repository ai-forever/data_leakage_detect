from smia.smia_models.registry import register_all, rel_dir, get_all_models
from safetensors.torch import load_file
import os


register_all(rel_dir(__file__))


def init_model(model_args):
    model_cls = get_all_models()[model_args.model_name]
    model = model_cls(model_args)
    if model_args.model_path is not None:
        model.load_state_dict(
            load_file(os.path.join(model_args.model_path, "model.safetensors"))
        )
    return model.cuda()


__all__ = ["get_all_models", "init_model"]

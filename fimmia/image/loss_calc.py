from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
import datasets
from trl import SFTConfig, SFTTrainer
from transformers import HfArgumentParser
from fimmia.sft_finetune_image import MODEL_DICT


@dataclass
class Args:
    df_path: str
    model_name: str
    part_size: int = 5000
    label: int = 0
    user_answer: int = 0
    # If 1, ignore modality_neighbors column (if present) and always use the
    # original modality path for all neighbors.
    ignore_modality_neighbors: int = 0


def format_sample(text, image, answer):
    sample = {
        "images": [image],
        "prompt": [{"content": text, "role": "user"}],
        "completion": [{"content": answer, "role": "assistant"}],
    }
    return sample


class LossCalculator:
    def __init__(self, args: Args):
        self.args = args
        self.model = None
        self.model_name = Path(self.args.model_name).name

    def load_model(self):
        model_cls = MODEL_DICT[self.args.model_name]
        self.model = model_cls.from_pretrained(
            self.args.model_name, device_map="auto", dtype=torch.bfloat16
        )

    def get_loss(self, text, image, answer):
        # TODO: rewrite this o my goh
        ds = datasets.Dataset.from_list([format_sample(text, image, answer)])
        trainer = SFTTrainer(
            model=self.model,
            args=SFTConfig(
                prediction_loss_only=True,
                per_device_train_batch_size=1,
                disable_tqdm=True,
                max_length=None,
            ),
            train_dataset=ds,
        )
        loss = trainer.predict(trainer.train_dataset).metrics["test_loss"]
        return loss

    def _neighbor_specs(self, row):
        """Return list of (neighbor_text, modality_path) for this row, in order used for loss computation."""
        neighbors = eval(row["neighbors"])
        modality_key = "image"
        mod_list = None
        if (
            not self.args.ignore_modality_neighbors
            and "modality_neighbors" in row
            and pd.notna(row.get("modality_neighbors"))
        ):
            try:
                mod_list = (
                    eval(row["modality_neighbors"])
                    if isinstance(row["modality_neighbors"], str)
                    else row["modality_neighbors"]
                )
            except Exception:
                pass
        if (
            mod_list is not None
            and isinstance(mod_list, list)
            and len(mod_list) == len(neighbors)
        ):
            return [(n, mod_list[i]) for i, n in enumerate(neighbors)]
        mod_path = row[modality_key]
        return [(n, mod_path) for n in set(neighbors)]

    def prc_df(self):
        save_dir = (
            Path(self.args.df_path).with_suffix("")
            / "loss"
            / self.model_name
            / ("leak" if self.args.label else "no_leak")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        df = pd.read_csv(self.args.df_path)
        num_part = 0
        res = []
        neighbor_specs_per_row = [self._neighbor_specs(row) for _, row in df.iterrows()]
        for (_, row), specs in tqdm(
            zip(df.iterrows(), neighbor_specs_per_row),
            total=len(df),
            desc=f"prc {self.args.df_path}",
        ):
            row.label = self.args.label
            new_row = dict(row)
            new_row.pop("neighbors", None)
            new_row.pop("modality_neighbors", None)
            input_loss = self.get_loss(row.input, row.image, row.answer)
            is_add = True
            for neighbor, mod_path in specs:
                line = deepcopy(new_row)
                line["neighbor"] = neighbor
                line["image"] = mod_path
                if self.args.user_answer:
                    text = row.input
                    answer = neighbor
                else:
                    text = neighbor
                    answer = row.answer
                line["neighbor_loss"] = self.get_loss(text, mod_path, answer)
                if is_add:
                    is_add = False
                    line["input_loss"] = input_loss
                else:
                    line["input_loss"] = None
                lines.append(line)
            if self.args.part_size < len(lines):
                print("Save part:", num_part, "Saved:", len(lines))
                fp = save_dir / f"part_{num_part}.csv"
                pd.DataFrame(lines).to_csv(str(fp), index=False)
                res.append(str(fp))
                num_part += 1
                lines = []
        if len(lines):
            print("Save part:", num_part, "Saved:", len(lines))
            fp = save_dir / f"part_{num_part}.csv"
            pd.DataFrame(lines).to_csv(str(fp), index=False)
            res.append(str(fp))
        return res


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    loss_calc = LossCalculator(args)
    loss_calc.load_model()
    res = loss_calc.prc_df()

    print("Processed:")
    for path in res:
        print(path)


if __name__ == "__main__":
    main()

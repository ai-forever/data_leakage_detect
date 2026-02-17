from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, HfArgumentParser
from fimmia.audio.train_qwen import SFTDataset, collate_fn


@dataclass
class Args:
    df_path: str
    model_name: str
    model_id: str
    part_size: int = 5000
    label: int = 0
    user_answer: int = 0
    # If 1, ignore modality_neighbors column (if present) and always use the
    # original modality path for all neighbors.
    ignore_modality_neighbors: int = 0


class LossCalculator:
    def __init__(self, args: Args):
        self.args: Args = args
        self.model = None
        self.tokenizer = None
        self.model_name = os.path.split(self.args.model_name)[-1]
        self.max_seq_length = 2048
        self.system_prompt = "You are a helpful assistant."

    def load_model(self):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.args.model_name, trust_remote_code=True, dtype=torch.bfloat16
            )
            .eval()
            .cuda()
        )

    def _get_audio_path(self, row):
        audio = row.get("audio")
        if audio is None or (isinstance(audio, float) and pd.isna(audio)):
            audio = row.get("audio_1")
        if audio is None or (isinstance(audio, float) and pd.isna(audio)):
            audio = row.get("audio_2")
        return audio

    def _neighbor_specs(self, row):
        """Return list of (neighbor_text, modality_path) for this row."""
        neighbors = eval(row["neighbors"])
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
        mod_path = self._get_audio_path(row)
        return [(n, mod_path) for n in set(neighbors)]

    def prc_df(self):
        save_dir = os.path.join(
            self.args.df_path[:-4],
            "loss",
            self.model_name,
            "leak" if self.args.label else "no_leak",
        )
        os.makedirs(save_dir, exist_ok=True)
        df = pd.read_csv(self.args.df_path)
        input_ds = []
        neighbor_ds = []
        neighbor_specs_per_row = []
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"prc {self.args.df_path}"
        ):
            specs = self._neighbor_specs(row)
            neighbor_specs_per_row.append(specs)
            input_ds.append(row)
            for neighbor, audio in specs:
                if self.args.user_answer:
                    text = row.input
                    answer = neighbor
                else:
                    text = neighbor
                    answer = row.answer
                new_row = {"input": text, "answer": answer, "audio": audio}
                neighbor_ds.append(new_row)

        test_ds = SFTDataset(data=input_ds, model_id=self.args.model_id)
        test_ds_neighbors = SFTDataset(data=neighbor_ds, model_id=self.args.model_id)

        input_losses = self.get_losses(test_ds)
        neighbor_losses = self.get_losses(test_ds_neighbors)

        num_part = 0
        lines = []
        res = []
        line_idx = 0
        for (_, row), input_loss, specs in tqdm(
            zip(df.iterrows(), input_losses, neighbor_specs_per_row),
            total=len(df),
        ):
            row.label = 1
            new_row = dict(row)
            new_row.pop("neighbors", None)
            for neighbor, audio in specs:
                line = deepcopy(new_row)
                line["neighbor"] = neighbor
                line["audio"] = audio
                line["neighbor_loss"] = neighbor_losses[line_idx]
                line["input_loss"] = input_loss
                line_idx += 1
                lines.append(line)
            if self.args.part_size < len(lines):
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

    def get_losses(self, ds):
        res = []
        data_collator = collate_fn
        with torch.inference_mode():
            for x in tqdm(ds, total=len(ds)):
                inputs = data_collator([x])
                audio_info = inputs.pop("audio_info", None)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(audio_info=audio_info, **inputs)
                loss = float(outputs.loss.item())
                res.append(loss)
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

from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from transformers import Qwen2AudioForConditionalGeneration, HfArgumentParser
from fimmia.video.train_qwen25vl import SFTDataset


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


def create_data_collator(self: SFTDataset):
    def data_collator(features):
        modality_inputs = []
        conversation = list(map(self.format_sample, features))
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        modality_inputs = {f"{self.modality}": self.load_modality_input(conversation)}
        inputs = self.processor(
            text=text,
            **modality_inputs,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        ).to("cuda")
        labels: torch.Tensor = inputs["input_ids"].clone()
        eos_token = getattr(self.processor, f"{self.modality}_eos_token")
        eos_token_id = self.processor.tokenizer(eos_token).input_ids[0]
        for idx in range(len(labels)):
            eos_indecies = torch.where(labels[idx] == eos_token_id)[0]
            if len(eos_indecies) > 0:
                last_eos_idx = eos_indecies[-1]
                labels[idx][: last_eos_idx + 1] = -100
        # print(inputs)
        inputs["labels"] = labels
        return inputs

    return data_collator


class LossCalculator:
    def __init__(self, args: Args):
        self.args: Args = args
        self.model = None
        self.model_name = Path(self.args.model_name).name

    def load_model(self):
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.args.model_name, device_map="auto", dtype=torch.bfloat16
        ).cuda()

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
        save_dir = (
            Path(self.args.df_path).with_suffix("")
            / "loss"
            / self.model_name
            / ("leak" if self.args.label else "no_leak")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
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

        test_ds = SFTDataset(
            data=input_ds, modality="audio", model_id=self.args.model_id
        )
        test_ds_neighbors = SFTDataset(
            data=neighbor_ds, modality="audio", model_id=self.args.model_id
        )

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

    def get_losses(self, ds):
        res = []
        data_collator = create_data_collator(ds)
        with torch.inference_mode():
            for x in tqdm(ds, total=len(ds)):
                inputs = data_collator([x])
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(**inputs)
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

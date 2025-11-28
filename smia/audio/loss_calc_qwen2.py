from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from fimmia.video.train_qwen25vl import *
from transformers import Qwen2AudioForConditionalGeneration
import os
import pandas as pd


@dataclass
class Args:
    df_path: str
    model_name: str
    model_id: str
    part_size: int = 5000
    label: int = 0
    user_answer: int = 0


def create_data_collator(self: SFTDataset):
    def data_collator(features):
        modality_inputs = []
        conversation = list(map(self.format_sample, features))
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        modality_inputs = {f"{self.modality}": self.load_modality_input(conversation)}
        inputs = self.processor(
            text=text, **modality_inputs, return_tensors="pt", padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate
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
        self.model_name = os.path.split(self.args.model_name)[-1]

    def load_model(self):
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.args.model_name, device_map="auto", dtype=torch.bfloat16).cuda()

    def prc_df(self):
        save_dir = os.path.join(
            self.args.df_path[:-4], "loss", self.model_name, "leak" if self.args.label else "no_leak")
        os.makedirs(save_dir, exist_ok=True)
        df = pd.read_csv(self.args.df_path)
        input_ds = []
        neighbor_ds = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"prc {self.args.df_path}"):

            neighbors = eval(row["neighbors"])
            input_ds.append(row)
            for neighbor in set(neighbors):
                if self.args.user_answer:
                    text = row.input
                    answer = neighbor
                else:
                    text = neighbor
                    answer = row.answer
                audio = row["audio"]
                if isinstance(audio, float):
                    audio = row["audio_1"]
                if isinstance(audio, float):
                    audio = row["audio_2"]
                new_row = {"input": text, "answer": answer, "audio": audio}
                neighbor_ds.append(new_row)

        test_ds = SFTDataset(data=input_ds, modality="audio", model_id=self.args.model_id)
        test_ds_neighbors = SFTDataset(data=neighbor_ds, modality="audio", model_id=self.args.model_id)

        input_losses = self.get_losses(test_ds)
        neighbor_losses = self.get_losses(test_ds_neighbors)

        num_part = 0
        lines = []
        res = []
        for (_, row), input_loss in tqdm(zip(df.iterrows(), input_losses), total=len(df), ):
            row.label = 1
            new_row = dict(row)
            neighbors = eval(new_row.pop("neighbors"))
            is_add = True
            for neighbor in set(neighbors):
                line = deepcopy(new_row)
                line["neighbor"] = neighbor
                line["neighbor_loss"] = neighbor_losses[len(lines)]
                line["input_loss"] = input_loss
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
    parser = HfArgumentParser((Args, ))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    loss_calc = LossCalculator(args)
    loss_calc.load_model()
    res = loss_calc.prc_df()

    print("Processed:")
    for path in res:
        print(path)


if __name__ == "__main__":
    main()

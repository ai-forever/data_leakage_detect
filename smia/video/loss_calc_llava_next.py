from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from fimmia.video.train_llava_next import *
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


class LossCalculator:
    def __init__(self, args: Args):
        self.args: Args = args
        self.model = None
        self.model_name = os.path.split(self.args.model_name)[-1]

    def load_model(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.args.model_id, device_map="auto", dtype=torch.bfloat16).cuda()

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
                video = row["video"]
                new_row = {"input": text, "answer": answer, "video": video}
                neighbor_ds.append(new_row)

        test_ds = SFTDataset(data=input_ds, modality="video", model_id=self.args.model_name)
        test_ds_neighbors = SFTDataset(data=neighbor_ds, modality="video", model_id=self.args.model_name)

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

    def get_losses(self, ds: SFTDataset):
        res = []
        data_collator = ds.create_data_collator()
        with torch.inference_mode():
            for x in tqdm(ds, total=len(ds)):
                inputs = data_collator([x])
                new_inputs = {}
                for k, v in inputs.items():
                    if hasattr(k, "cuda"):
                        v = v.cuda()
                    new_inputs[k] = v
                outputs = self.model(**new_inputs)
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

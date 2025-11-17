from tqdm import tqdm
from copy import deepcopy
from smia.sft_finetune_image import *
import os
import pandas as pd


@dataclass
class Args:
    df_path: str
    model_name: str
    part_size: int = 5000
    label: int = 0
    user_answer: int = 0


def format_sample(text, image, answer):
    sample = {
        'images': [image],
        'prompt': [{'content': text,'role': 'user'}],
        'completion': [{'content': answer, 'role': 'assistant'}]
    }
    return sample


class LossCalculator:
    def __init__(self, args: Args):
        self.args = args
        self.model = None
        self.model_name = os.path.split(self.args.model_name)[-1]

    def load_model(self):
        model_cls = MODEL_DICT[self.args.model_name]
        self.model = model_cls.from_pretrained(self.args.model_name, device_map="auto", dtype=torch.bfloat16)

    def get_loss(self, text, image, answer):
        # TODO: rewrite this o my goh
        ds = datasets.Dataset.from_list([format_sample(text, image, answer)])
        trainer = SFTTrainer(
            model=self.model,
            args=SFTConfig(
                prediction_loss_only=True, per_device_train_batch_size=1, disable_tqdm=True, max_length=None),
            train_dataset=ds,
        )
        loss = trainer.predict(trainer.train_dataset).metrics["test_loss"]
        return loss

    def prc_df(self):
        save_dir = os.path.join(
            self.args.df_path[:-4], "loss", self.model_name, "leak" if self.args.label else "no_leak")
        os.makedirs(save_dir, exist_ok=True)
        lines = []
        df = pd.read_csv(self.args.df_path)
        num_part = 0
        res = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"prc {self.args.df_path}"):
            row.label = self.args.label
            new_row = dict(row)
            neighbors = eval(new_row.pop("neighbors"))
            input_loss = self.get_loss(row.input, row.image, row.answer)
            is_add = True
            for neighbor in set(neighbors):
                line = deepcopy(new_row)
                line["neighbor"] = neighbor
                if self.args.user_answer:
                    text = row.input
                    answer = neighbor
                else:
                    text = neighbor
                    answer = row.answer
                line["neighbor_loss"] = self.get_loss(text, row.image, answer)
                if is_add:
                    is_add = False
                    line["input_loss"] = input_loss
                else:
                    line["input_loss"] = None
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

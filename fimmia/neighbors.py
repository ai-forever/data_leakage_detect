from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, HfArgumentParser
from tqdm import tqdm
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import string


@dataclass
class Args:
    dataset_path: str
    model_path: str
    mask_token: str = "<extra_id_0>"
    end_token: str = "<extra_id_1>"
    max_length_ratio: int = 5
    prefix_token: str = "<SC1>"
    n: int = 25
    mask_size: float = 0.1
    max_masks: int = 40
    min_masks: int = 5
    num_return_sequences: int = 1
    run_on_batch: int = 1
    num_parts: int = 5
    max_text_len: int = 3000
    user_answer: int = 0


class NeighborsGenerator:
    def __init__(self, args: Args):
        self.t5_tokenizer = None
        self.t5_mlm = None
        self.args = args
        self._trans = str.maketrans('', '', string.punctuation)

    @staticmethod
    def load_data(dataset_path):
        return pd.read_csv(dataset_path)

    def load_model_and_tokenizer(self):
        if self.t5_tokenizer is None:
            if "FRED" in self.args.model_path:
                self.t5_tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, eos_token='</s>')
            else:
                self.t5_tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        if self.t5_mlm is None:
            t5_config = T5Config.from_pretrained(self.args.model_path)
            self.t5_mlm = T5ForConditionalGeneration.from_pretrained(self.args.model_path, config=t5_config).cuda()
            self.t5_mlm = torch.compile(self.t5_mlm, mode="reduce-overhead", fullgraph=True)

    def fill_mask(self, text, max_length):
        text = self.args.prefix_token + text
        with torch.inference_mode():
            encoded = self.t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
            input_ids = encoded['input_ids'].cuda()
            outputs = self.t5_mlm.generate(
                input_ids=input_ids,
                num_return_sequences=self.args.num_return_sequences,
                max_length=max_length
            )
            res = []
            index = text.index(self.args.mask_token)
            result_prefix = text[:index].strip()
            result_suffix = text[index + len(self.args.mask_token):].strip()
            for output in outputs:
                generated = self.t5_tokenizer.decode(
                    output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if self.args.end_token in generated:
                    end_token_index = generated.index(self.args.end_token)
                    generated = result_prefix + " " + generated[:end_token_index].strip() + " " + result_suffix
                else:
                    generated = result_prefix + " " + generated.strip() + " " + result_suffix
                res.append(
                    generated.replace("</s>", " ").replace("<s>", " ").replace(self.args.prefix_token, "").strip())
        return str(np.random.choice(res))

    def filter_punctuation(self, tokens):
        ids = []
        for idx, tok in enumerate(tokens):
            if tok.translate(self._trans) == tok:
                ids.append(idx)
        return ids

    def substituting_neighbors(self, text, n=10, mask_count=5):

        res = []
        replace_idx = 0
        for _ in range(n):
            current_text = text
            for _ in range(mask_count):
                current_text = current_text.split()
                token_ids = self.filter_punctuation(current_text)
                try:
                    replace_idx = np.random.choice(token_ids)
                    max_length = len(self.t5_tokenizer.encode([current_text[replace_idx]])) + self.args.max_length_ratio
                except TypeError:
                    max_length = len(self.t5_tokenizer.encode(current_text[replace_idx])) + self.args.max_length_ratio
                current_text[replace_idx] = self.args.mask_token
                current_text = " ".join(current_text)
                current_text = self.fill_mask(text=current_text, max_length=max_length)
            res.append(current_text)
        return res

    def deletion_neighbors(self, token_ids, tokens, n=10, mask_count=5):
        res = []
        for _ in range(n):
            new_tokens = deepcopy(tokens)
            deleted_ids = []
            new_token_ids = deepcopy(token_ids)
            for _ in range(mask_count):
                try:
                    idx = np.random.choice(new_token_ids)
                    new_token_ids.remove(idx)
                    shift = len([1 for x in deleted_ids if x <= idx])
                    deleted_ids.append(idx)
                    new_tokens.pop(idx - shift)
                except:
                    pass
            res.append(" ".join(new_tokens))
        return res

    def duplication_neighbors(self, token_ids, tokens: list, n=10, mask_count=5):
        res = []
        for _ in range(n):
            new_tokens = deepcopy(tokens)
            inserted_ids = []
            new_token_ids = deepcopy(token_ids)
            for _ in range(mask_count):
                idx = np.random.choice(new_token_ids)
                new_token_ids.remove(idx)
                shift = len([1 for x in inserted_ids if x < idx])
                inserted_ids.append(idx)
                new_tokens.insert(idx + shift, tokens[idx])
            res.append(" ".join(new_tokens))
        return res

    def swap_neighbors(self, tokens, n=10, mask_count=5):
        res = []
        token_ids = list(range(len(tokens)))
        for _ in range(n):
            new_tokens = deepcopy(tokens)
            new_token_ids = deepcopy(token_ids)
            for _ in range(mask_count):
                idx1 = np.random.choice(new_token_ids)
                idx2 = np.random.choice(new_token_ids)
                new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
            res.append(" ".join(new_tokens))
        return res

    def get_neighbors_for_sample(self, text):
        tokens = text.split()
        token_ids = self.filter_punctuation(tokens)
        size = self.args.n // 4
        mask_count = int(len(tokens) * self.args.mask_size)
        min_masks = max(0, min(self.args.min_masks, len(token_ids) - 1))
        mask_count = min(max(mask_count, min_masks), self.args.max_masks)
        try:
            substituting_neighbors = self.substituting_neighbors(text=text, n=size, mask_count=mask_count)
        except:
            substituting_neighbors = []
        try:
            deletion_neighbors = self.deletion_neighbors(
                token_ids=token_ids, tokens=tokens, n=size, mask_count=mask_count)
        except:
            deletion_neighbors = []
        try:
            duplication_neighbors = self.duplication_neighbors(
                token_ids=token_ids, tokens=tokens, n=size, mask_count=mask_count)
        except:
            duplication_neighbors = []
        try:
            swap_neighbors = self.swap_neighbors(tokens, n=size, mask_count=mask_count)
        except:
            swap_neighbors = []
        neighbors = substituting_neighbors + deletion_neighbors + duplication_neighbors + swap_neighbors
        return neighbors

    def predict(self, dataset_path):
        data = self.load_data(dataset_path)
        self.load_model_and_tokenizer()
        neighbors = []
        new_input = []
        if self.args.user_answer:
            data_iter = data.answer
        else:
            data_iter = data.input

        for text in tqdm(data_iter, total=len(data), leave=True):
            text = text[-self.args.max_text_len:]
            new_input.append(text)
            neighbors.append(self.get_neighbors_for_sample(
                text=text,
            ))
        data["neighbors"] = neighbors
        if self.args.user_answer:
            data["answer"] = new_input
        else:
            data["input"] = new_input

        return data

    @classmethod
    def run_on_batch(cls, args):
        ng = cls(args)
        df = ng.predict(args.dataset_path)
        df.to_csv(args.dataset_path, index=False)


def main():
    parser = HfArgumentParser((Args, ))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    NeighborsGenerator.run_on_batch(args)


if __name__ == "__main__":
    main()

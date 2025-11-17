from torch.utils.data import Dataset
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedModel, AutoModelForCausalLM
from typing import Dict, Any, List, Optional
import pandas as pd
from smia.sft_finetune_image import Args


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_length = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for ids, mask, lbl in zip(input_ids, attention_mask, labels):
        padding_length = max_length - len(ids)

        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)]))
        padded_attention_mask.append(
            torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        )

        # For labels, we use -100 for padding tokens (ignored in CrossEntropyLoss)
        padded_labels.append(
            torch.cat([lbl, torch.full((padding_length,), -100, dtype=torch.long)])
        )

    input_ids_tensor = torch.stack(padded_input_ids)
    attention_mask_tensor = torch.stack(padded_attention_mask)
    labels_tensor = torch.stack(padded_labels)

    audio_info_list = [item["audio_info"] for item in batch]

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
        "audio_info": audio_info_list,
    }


class QwenAudioSFTTrainer(SFTTrainer):
    """Custom SFTTrainer for Qwen-Audio-Chat that handles audio information."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        audio_info = inputs.pop("audio_info", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            audio_info=audio_info,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class SFTDataset(Dataset):
    def __init__(
            self, data, model_id,
            max_length: int = 2048,
            system_prompt: str = "You are a helpful assistant.",

    ):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model_id = model_id
        self.max_length = max_length
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, answer = self.format_sample(self.data.iloc[idx])
        raw_text = self.get_raw_text(query, answer)
        audio_info = self.tokenizer.process_audio(raw_text)
        tokens = self.get_context_tokens(query, answer)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        try:
            prefix_len = len(tokens) - tokens[::-1].index(self.tokenizer.audio_end_id)
        except ValueError:
            prefix_len = 0
        labels = [-100] * prefix_len + tokens[prefix_len:]
        attention_mask = [1] * len(tokens)

        input_ids = torch.tensor(tokens)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_info": audio_info,
            "labels": labels,
        }

    def get_audio_info(self, query: str, answer: str) -> Dict[str, Any]:
        raw_text = self.get_raw_text(query, answer)
        audio_info = self.tokenizer.process_audio(raw_text)
        return audio_info

    def get_raw_text(self, query: str, answer: str) -> str:
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        raw_text = (
            f"{im_start}system\n"
            f"{self.system_prompt}{im_end}\n"
            f"{im_start}user\n"
            f"{query}{im_end}\n{im_start}assistant\n"
            f"{answer}{im_end}"
        )
        return raw_text

    def get_context_tokens(self, query: str, answer: str) -> List[int]:
        im_start_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        nl_tokens = self.tokenizer.encode("\n")

        def tokenize(role: str, content: str) -> List[int]:
            audio_info = self.tokenizer.process_audio(content)
            role_tokens = self.tokenizer.encode(
                role, allowed_special=set(self.tokenizer.AUDIO_ST), audio_info=audio_info
            )
            content_tokens = self.tokenizer.encode(
                content, allowed_special=set(self.tokenizer.AUDIO_ST), audio_info=audio_info
            )
            return role_tokens + nl_tokens + content_tokens

        system_tokens_part = tokenize("system", self.system_prompt)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens + nl_tokens
        query_tokens = im_start_tokens + tokenize("user", query) + im_end_tokens + nl_tokens
        answer_tokens = im_start_tokens + tokenize("assistant", answer) + im_end_tokens
        return system_tokens + query_tokens + answer_tokens

    def format_sample(self, row):
        audio = row["audio"]
        if isinstance(audio, float):
            audio = row["audio_1"]
        if isinstance(audio, float):
            audio = row["audio_2"]
        text = row.input
        text = text.replace("<audio>", " ")
        query = self.tokenizer.from_list_format([{"audio": audio}, {"text": text}])
        answer = row.answer
        return query, answer


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    train_ds = SFTDataset(data=args.train_df_path, model_id=args.model_id)
    test_ds = SFTDataset(data=args.test_df_path, model_id=args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=args.output_dir,  # Directory to save the model
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Steps to accumulate gradients
        # gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim=args.optim,  # Optimizer type
        learning_rate=args.learning_rate,  # Learning rate for training
        lr_scheduler_type=args.lr_scheduler_type,  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=args.logging_steps,  # Steps interval for logging
        # eval_steps=10,  # Steps interval for evaluation
        eval_strategy=args.eval_strategy,  # Strategy for evaluation
        save_strategy=args.save_strategy,  # Strategy for saving the model
        metric_for_best_model=args.metric_for_best_model,  # Metric to evaluate the best model
        greater_is_better=args.greater_is_better,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=False,  # Use TensorFloat-32 precision
        max_grad_norm=args.max_grad_norm,  # Maximum norm for gradient clipping
        warmup_ratio=args.warmup_ratio,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        data_seed=1234,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    # SFTTrainer
    trainer = QwenAudioSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config,
        data_collator=collate_fn,
        processing_class=train_ds.tokenizer
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()

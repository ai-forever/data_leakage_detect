from torch.utils.data import Dataset
import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor, HfArgumentParser
from peft import LoraConfig, get_peft_model
import pandas as pd
from smia.sft_finetune_image import Args
import numpy as np
import av


def read_video_pyav(video_path):
    '''
    Decode the video with PyAV decoder.
    Args:
        video_path: str
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_videos(conversation):
    res = []
    for conv in conversation:
        for message in conv:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "video":
                        res.append(read_video_pyav(ele["video"]))
    return res


class SFTDataset(Dataset):
    def __init__(self, data, modality, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model_id = model_id
        self.data = None
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
        self.data_path = data
        self.modality = modality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def format_sample(self, row):
        if self.modality == "audio":
            audio = row["audio"]
            if isinstance(audio, float):
                audio = row["audio_1"]
            if isinstance(audio, float):
                audio = row["audio_2"]
            modality_input = {"type": self.modality, self.modality: audio}
        else:
            modality_input = {"type": self.modality, self.modality: row[self.modality]}
        text = row["input"]
        text = text.replace("<video>", " ")
        if self.modality == "video":
            modality_input["resized_height"] = 256
            modality_input["resized_width"] = 256
        text = row["input"]
        text = text.replace("<video>", " ")
        sample = [{
            'content': [{"type": "text", "text": text}, modality_input],
            "role": "user"
        }, {
            'content': [{"type": "text", "text": row["answer"]}],
            'role': 'assistant'
        }]
        return sample

    def create_data_collator(self):
        def data_collator(features):
            modality_inputs = []
            conversation = list(map(self.format_sample, features))
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            modality_inputs = {f"{self.modality}s": get_videos(conversation)}
            inputs = self.processor(text=text, **modality_inputs, return_tensors="pt", padding=True).to("cuda")
            labels: torch.Tensor = inputs["input_ids"].clone()
            eos_token_id = getattr(self.processor, f"{self.modality}_token_id")
            for idx in range(len(labels)):
                eos_indecies = torch.where(labels[idx] == eos_token_id)[0]
                if len(eos_indecies) > 0:
                    last_eos_idx = eos_indecies[-1]
                    labels[idx][: last_eos_idx + 1] = -100
            # print(inputs)
            inputs["labels"] = labels
            max_length = 4096
            inputs["labels"] = inputs["labels"][:, -max_length:]
            inputs["input_ids"] = inputs["input_ids"][:, -max_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_length:]
            return inputs
        return data_collator


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    train_ds = SFTDataset(data=args.train_df_path, modality="video", model_id=args.model_id)
    test_ds = SFTDataset(data=args.test_df_path, modality="video", model_id=args.model_id)
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, device_map="auto", dtype=torch.bfloat16)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
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
        gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing for memory efficiency
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
        load_best_model_at_end=False,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=False,  # Use TensorFloat-32 precision
        max_grad_norm=args.max_grad_norm,  # Maximum norm for gradient clipping
        warmup_ratio=args.warmup_ratio,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        gradient_checkpointing_kwargs={"use_reentrant": False},
        data_seed=1234,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=False
    )
    data_collator = train_ds.create_data_collator()
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()

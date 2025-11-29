import datasets
from dataclasses import dataclass
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForVision2Seq, HfArgumentParser, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch


MODEL_DICT = {
    "Qwen/Qwen2.5-VL-3B-Instruct": AutoModelForVision2Seq,
    "Qwen/Qwen2-VL-7B-Instruct": AutoModelForVision2Seq,
    "Qwen/Qwen2.5-VL-7B-Instruct": AutoModelForVision2Seq,
    "llava-hf/llama3-llava-next-8b-hf": AutoModelForVision2Seq,
    "google/gemma-3-4b-it": AutoModelForCausalLM,
    "google/gemma-3-12b-it": AutoModelForCausalLM
}


@dataclass
class Args:
    train_df_path: str
    test_df_path: str
    model_id: str
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "constant"
    # Logging and evaluation
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    # Mixed precision and gradient settings
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    data_seed: int = 1234


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_cls = MODEL_DICT[args.model_id]
    model = model_cls.from_pretrained(args.model_id, device_map="auto", dtype=torch.bfloat16)
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
    )
    train_ds = datasets.load_from_disk(args.train_df_path)
    test_ds = datasets.load_from_disk(args.test_df_path)
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()

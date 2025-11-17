from smia.video.train_qwen25vl import *
from transformers import Qwen2AudioForConditionalGeneration


MODEL_DICT = {
    "Qwen/Qwen2-Audio-7B-Instruct": Qwen2AudioForConditionalGeneration,
}


def main():
    parser = HfArgumentParser((Args,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    train_ds = SFTDataset(data=args.train_df_path, modality="audio", model_id=args.model_id)
    test_ds = SFTDataset(data=args.test_df_path, modality="audio", model_id=args.model_id)
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

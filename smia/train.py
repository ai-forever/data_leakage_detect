import os.path
from dataclasses import dataclass, asdict
from transformers import TrainingArguments, Trainer, HfArgumentParser
from smia.smia_models import init_model
from smia.utils.mds_dataset import get_streaming_ds
from smia.utils.data import create_data_collator
from transformers import default_data_collator
from smia.utils.metrics import get_metrics_from_df, get_df_with_predictions


@dataclass
class DataTrainingArguments:
    train_dataset_path: str
    val_dataset_path: str = None
    test_dataset_path: str = None


@dataclass
class ModelArguments:
    model_name: str = "BaseLineModel"
    embedding_size: int = 4096
    projection_size: int = 512
    image_embedding_size: int = 1024
    model_path: str = None
    sigmas_path: str = None
    sigmas_type: str = None


@dataclass
class DefaultTrainingArguments:
    output_dir: str
    num_train_epochs: int = 10
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch_fused"
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    logging_steps: int = 3000
    save_strategy: str = "epoch"
    learning_rate: float = 2e-5
    max_grad_norm: float = None  # 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    push_to_hub: bool = False
    report_to: str = "tensorboard"
    save_total_limit: int = None
    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "epoch"


def train(
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        data_collator=default_data_collator
):
    train_paths = data_args.train_dataset_path
    if isinstance(train_paths, str):
        train_paths = train_paths.split(",")
    model = init_model(model_args)
    train_ds = get_streaming_ds(
        paths=train_paths,
        shuffle=True
    )
    val_paths = data_args.val_dataset_path
    if isinstance(val_paths, str):
        val_paths = val_paths.split(",")
    val_ds = get_streaming_ds(
        paths=val_paths,
        shuffle=False
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**asdict(training_args)),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )
    try:
        trainer.train()
        save_path = os.path.join(training_args.output_dir, "test.csv")
        save_metrics_path = os.path.join(training_args.output_dir, "test_metrics.csv")
        # save model
        trainer.save_model()
        df_pred = get_df_with_predictions(val_ds, trainer)
        df_pred.to_csv(save_path, index=False)
        m = get_metrics_from_df(df_pred)
        m.to_csv(save_metrics_path, index=False)
    except KeyboardInterrupt:
        print("Stop training...")
    return trainer


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DefaultTrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_collator = create_data_collator(
        model_args.model_name, sigmas_path=model_args.sigmas_path, sigmas_type=model_args.sigmas_type)
    train(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

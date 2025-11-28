from fimmia.train import (
    ModelArguments,
    init_model,
    TrainingArguments,
    DefaultTrainingArguments,
    Trainer
)
from transformers import HfArgumentParser
from dataclasses import asdict, dataclass
from fimmia.utils.data import create_data_collator
from fimmia.utils.metrics import get_metrics_from_df, get_df_with_predictions


@dataclass
class InferenceArgs:
    model_name: str
    model_path: str
    test_path: str
    save_path: str
    save_metrics_path: str
    sigmas_path: str = None
    sigmas_type: str = None


def main():
    parser = HfArgumentParser((InferenceArgs,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_args = ModelArguments(
        model_name=args.model_name,
        model_path=args.model_path
    )
    model = init_model(model_args)
    training_args = TrainingArguments(**asdict(
        DefaultTrainingArguments(output_dir=args.model_path, eval_strategy="no")), remove_unused_columns=False)
    data_collator = create_data_collator(
        model_name=args.model_name, sigmas_path=args.sigmas_path, sigmas_type=args.sigmas_type)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=data_collator
    )
    df_pred = get_df_with_predictions(args.test_path, trainer)
    # save_path = os.path.join(args.save_dir, os.path.split(args.test_path)[-1])
    # os.makedirs(args.save_path, exist_ok=True)
    df_pred.to_csv(args.save_path, index=False)
    m = get_metrics_from_df(df_pred)
    m.to_csv(args.save_metrics_path, index=False)

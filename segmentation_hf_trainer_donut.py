import torch
from transformers import (
    Trainer,
    TrainingArguments,
)
from datetime import datetime
from transformers import DonutProcessor, VisionEncoderDecoderModel
import click
import os
from segmentation_dataset import SegmentationDataset, Y_COORDS_TOKENS, X_COORDS_TOKENS

MISSING_TOKEN = "</Missing>"
MAX_STEPS = int(1e6)


def custom_collate_fn(batch):
    # Stacking pixel_values
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_values_stacked = torch.concatenate(pixel_values)

    # Padding and stacking labels
    labels = [item["labels"] for item in batch]
    labels_stacked = torch.concatenate(labels)
    # To mask the loss
    labels_stacked[labels_stacked == 1] = -100

    return {"pixel_values": pixel_values_stacked, "labels": labels_stacked}


@click.command()
@click.option(
    "--dataset-path",
    default="synth_data/batch_1/",
    help="Path to the dataset directory.",
)
@click.option(
    "--resume-from-checkpoint",
    default=False,
    is_flag=True,
    help="Whether to resume training from a checkpoint.",
)
@click.option(
    "--test-run",
    default=False,
    is_flag=True,
    help="Runs the training script on a very small subset",
)
def train(dataset_path: str, resume_from_checkpoint: bool, test_run: bool):
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    # the tokenizer doesn't natively have a pad token
    processor.tokenizer.add_tokens([MISSING_TOKEN])

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    model.gradient_checkpointing_enable()

    processor.tokenizer.add_tokens(list(Y_COORDS_TOKENS.values()))
    processor.tokenizer.add_tokens(list(X_COORDS_TOKENS.values()))
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    train_dataset = SegmentationDataset(processor, dataset_path, "train", test_run)
    eval_dataset = SegmentationDataset(processor, dataset_path, "val", test_run)

    project = "segmentation"
    base_model_name = "donut"
    run_name = base_model_name + "-" + project

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir="test_run/",
            warmup_steps=1000,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            remove_unused_columns=False,
            max_steps=MAX_STEPS,
            learning_rate=2.5e-5,
            logging_steps=100 if not test_run else 5,
            # bf16=True,
            # TODO: torch.compile leads to OOM for now
            # torch_compile=True,
            # torch_compile_backend='inductor',
            resume_from_checkpoint=resume_from_checkpoint,
            max_grad_norm=1.0,  # This should already be the default
            # optim="adamw_bnb_8bit",
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=5000,  # Save checkpoints every 50 steps
            save_total_limit=2,
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=5000
            if not test_run
            else 10,  # Evaluate and save checkpoints every 50 steps
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            do_eval=True,  # Perform evaluation at the end of training
            # report_to=None,
            report_to="wandb",  # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=custom_collate_fn,
    )
    trainer.train()


if __name__ == "__main__":
    train()

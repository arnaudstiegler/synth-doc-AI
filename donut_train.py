import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import wandb
import os
from datetime import datetime
from typing import Dict, Any
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import DonutProcessor, VisionEncoderDecoderModel
import click

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
import torch._dynamo

torch._dynamo.config.suppress_errors = True

import os
import json
from PIL import Image
from faker import Faker
import random

MISSING_TOKEN = "</Missing>"


class KVDataset:
    VAL_SPLIT_SIZE = 0.05

    def __init__(self, folder_path: str, split: str):
        self.folder_path = folder_path
        self.split = split

        self.docs = self.init_docs(split)
        self.kv_pairs = list(self.get_kv_pairs())
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        

    def init_docs(self, split: str):
        docs = []
        for file in os.listdir(self.folder_path):
            if "kv_pairs" in file:
                kv_pairs = json.load(open(os.path.join(self.folder_path, file)))
                sample_id = file.split("_")[2].replace(".json", "")
                img_path = os.path.join(self.folder_path, f"sample_{sample_id}_aug.png")
                if not os.path.exists(img_path):
                    # Skipping if we're missing the corresponding img (download issue)
                    continue
                docs.append((kv_pairs, img_path))

        print(f'Split={self.split} size: {len(docs)}')

        docs = sorted(docs, key=lambda x: x[1])

        if split == "train":
            train_split_size = int((1 - self.VAL_SPLIT_SIZE) * len(docs))
            return docs[:train_split_size]
        elif split == "val":
            val_split_size = int((self.VAL_SPLIT_SIZE) * len(docs))
            return docs[-val_split_size:]

    def get_kv_pairs(self):
        kv_pairs_list = []
        for kv_pairs, _ in self.docs:
            kv_pairs_list += [k for k, _ in kv_pairs]
        return set(kv_pairs_list)

    def __getitem__(self, i):
        doc_kv, img_path = self.docs[i]
        img = Image.open(img_path).convert('RGB')

        if len(doc_kv) == 0:
            random_key = random.choice(self.kv_pairs)
            text_target = f"{random_key.lower() if random.random() < 0.2 else random_key}: {MISSING_TOKEN}"
        else:
            # TODO: should I change that for eval?
            k, v = random.choice(doc_kv)
            text_target = f"{k.lower() if random.random() < 0.2 else k}: {v}"

        # Breakdown to avoid the warning message
        pixel_values = self.processor(img, return_tensors="pt")
        labels = self.processor.tokenizer(
            text_target, return_tensors="pt", max_length=128, padding="max_length", truncation=True
        )

        return {
            "pixel_values": pixel_values["pixel_values"],
            "labels": labels["input_ids"],
            "image_path": img_path,
        }

    def __len__(self):
        return len(self.docs)


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
def train(dataset_path: str, resume_from_checkpoint: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = KVDataset(dataset_path, "train")
    eval_dataset = KVDataset(dataset_path, "val")

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    # the tokenizer doesn't natively have a pad token
    processor.tokenizer.add_tokens([MISSING_TOKEN])

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.gradient_checkpointing_enable()

    model = accelerator.prepare_model(model)

    project = "synth-donut"
    base_model_name = "donut"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir="test_run/",
            warmup_steps=1000,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            remove_unused_columns=False,
            max_steps=MAX_STEPS,
            learning_rate=2.5e-5,
            logging_steps=100,
            bf16=True,
            resume_from_checkpoint=resume_from_checkpoint,
            max_grad_norm=1.0, # This should already be the default
            optim="adamw_torch",
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=5000,  # Save checkpoints every 50 steps
            save_total_limit=2,
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=5000,  # Evaluate and save checkpoints every 50 steps
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

    # model.config.use_cache = (
    #     False  # silence the warnings. Please re-enable for inference!
    # )
    trainer.train()



if __name__ == "__main__":
    train()

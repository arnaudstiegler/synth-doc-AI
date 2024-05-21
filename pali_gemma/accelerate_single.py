import logging
from functools import partial
import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor
import bitsandbytes as bnb
import torch
import time
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaForConditionalGeneration
import torch
from datasets import load_dataset



accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
device_index = accelerator.process_index


dataset = load_dataset("arnaudstiegler/synthetic_us_passports_easy")
train_dataset = dataset['train']
eval_dataset = dataset['test']



# TODO: to replace
# model_id = "google/paligemma-3b-pt-896"
model_id = 'google/paligemma-3b-pt-224'
processor = AutoProcessor.from_pretrained(model_id)
processor.max_length = 128
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    device_map = {"": device_index})
model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()

def collate_fn(examples):
    texts = []
    for example in examples:
        text_str = ', '.join([f'{k.replace("_", " ")} = {v}' for k, v in example.items() if k != 'image'])
        texts.append(text_str)
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images,
                return_tensors="pt", truncation=True, padding=True, max_length=128,
                tokenize_newline_separately=False)
    labels = tokens["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token] = -100
    tokens["labels"] = labels
    return tokens

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

accelerator.init_trackers(
    project_name="huggingface", init_kwargs={"wandb": {"entity": "arnaud-stiegler"}}
)

train_data = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=1,
    collate_fn=collate_fn,
)
val_data = torch.utils.data.DataLoader(
    eval_dataset,
    shuffle=False,
    batch_size=1,
    collate_fn=collate_fn,
)

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5)
model, optimizer, train_data, val_data = accelerator.prepare(
    model, optimizer, train_data, val_data
)

total_step = 0
for epoch in range(10):
    for batch in train_data:
        model.train()
        optimizer.zero_grad()
        train_start = time.time()

        output = model(**batch)
        loss = output.loss
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.log({"train/loss": loss.item()}, step=total_step)
        accelerator.backward(loss)
        optimizer.step()

        accelerator.log(
            {"train/time_per_step": time.time() - train_start}, step=total_step
        )
        total_step += 1

        if total_step % 10 == 0:
            # Run eval
            model.eval()
            with torch.no_grad():
                loss_avg = []
                eval_step_avg = []
                for eval_batch in val_data:
                    eval_start = time.time()
                    output = model(**eval_batch)
                    loss = output.loss
                    eval_step_avg.append(time.time() - eval_start)
                    loss_avg.append(loss.cpu())

                accelerator.log(
                    {
                        "val/loss": np.mean(loss_avg),
                        "eval/samples_per_second": sum(eval_step_avg)
                        / len(eval_step_avg),
                    },
                    step=total_step,
                )

# Make sure that the wandb tracker finishes correctly
accelerator.end_training()

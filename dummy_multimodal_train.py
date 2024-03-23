import logging
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from torch.utils.data import default_collate
import bitsandbytes as bnb
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model directly
model = AutoModelForCausalLM.from_pretrained(
    "adept/fuyu-8b",
    torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    # attn_implementation="flash_attention_2",
    # code_revision="main",
)
processor = AutoProcessor.from_pretrained(
    "adept/fuyu-8b", padding="max_length", max_length=128
)

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
# config = read_deepspeed_config()
accelerator = Accelerator()

dataset = load_dataset("textvqa")
sample = dataset["train"][0]

# collate = partial(TextVQADataset.collate_fn, processor.tokenizer.pad_token_id)
data = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=1,
    collate_fn=default_collate,
)

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.01)
model, optimizer, data = accelerator.prepare(model, optimizer, data)
model.gradient_checkpointing_enable()

sample = dataset["train"][0]
model.train()
for epoch in range(10):
    for k in range(100):
        optimizer.zero_grad()
        text = sample["question"] + " " + sample["answers"][0]
        maxsize = (512, 512)
        sample["image"].thumbnail(maxsize, PIL.Image.LANCZOS)
        inputs = processor(text=text, images=sample["image"], return_tensors="pt").to(
            "cuda:0"
        )
        # print(inputs['image_patches'][0])
        inputs["labels"] = inputs["input_ids"].clone()
        output = model(**inputs)
        loss = output.loss
        print(f"Loss: {loss.item()}")
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.backward(loss)
        optimizer.step()

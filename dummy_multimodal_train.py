import logging
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import TextVQADataset
from utils import read_deepspeed_config
from transformers import LlavaForConditionalGeneration, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model directly
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
    # code_revision="main",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
config = read_deepspeed_config()
accelerator = Accelerator()

dataset = TextVQADataset(processor, "train")

collate = partial(TextVQADataset.collate_fn, processor.tokenizer.pad_token_id)
data = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=config["train_micro_batch_size_per_gpu"],
    collate_fn=collate,
)

optimizer_cls = (
    AdamW
    if accelerator.state.deepspeed_plugin is None
    or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    else DummyOptim
)
optimizer = optimizer_cls(model.parameters())
model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
    for batch in data:
        optimizer.zero_grad()
        print(batch["input_ids"].shape)
        output = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = output.loss
        print(f"Loss: {loss.item()}")
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.backward(loss)
        optimizer.step()

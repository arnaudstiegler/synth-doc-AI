import logging
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import SquadDataset
from dataset import collate_fn
from utils import read_deepspeed_config

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    code_revision="main",
).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# BEWARE !!!
tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
tokenizer.pad_token = "<PAD>"

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
config = read_deepspeed_config()
accelerator = Accelerator()

dataset = SquadDataset(tokenizer, "train")

collate = partial(collate_fn, tokenizer.pad_token_id)
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
        print(batch['input_ids'].shape)
        output = model(batch["input_ids"], labels=batch["labels"])
        loss = output.loss
        print(f"Loss: {loss.item()}")
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.backward(loss)
        optimizer.step()

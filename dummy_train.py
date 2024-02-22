from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.optim import AdamW
from utils import read_deepspeed_config
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from dataset import SquadDataset
from accelerate.logging import get_logger
import accelerate
from accelerate.utils import DummyOptim
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/phi-2", torch_dtype='auto' if torch.cuda.is_available() else torch.float32,
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2",
#     code_revision='main'
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
config = read_deepspeed_config()
accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

dataset = SquadDataset(tokenizer, "train")


data = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)

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

        output = model(batch["input_ids"], labels=batch["labels"])
        loss = output.loss
        print(f"Loss: {loss.item()}")
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.backward(loss)
        optimizer.step()

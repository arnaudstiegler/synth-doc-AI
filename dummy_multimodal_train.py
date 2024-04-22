import logging
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, BitsAndBytesConfig
from torch.utils.data import default_collate
import bitsandbytes as bnb
import torch
from donut_train import KVDataset, MISSING_TOKEN, custom_collate_fn


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


dataset_path = '/home/ubuntu/synth_data/'
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = KVDataset(dataset_path, "train", True)
eval_dataset = KVDataset(dataset_path, "val", True)

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
# the tokenizer doesn't natively have a pad token
processor.tokenizer.add_tokens([MISSING_TOKEN])

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.gradient_checkpointing_enable()

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

accelerator = Accelerator()
train_data = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=2,
    collate_fn=default_collate,
)
val_data = torch.utils.data.DataLoader(
    eval_dataset,
    shuffle=True,
    batch_size=4,
    collate_fn=default_collate,
)

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-5)
model, optimizer, data = accelerator.prepare(model, optimizer, train_data, val_data)
model.gradient_checkpointing_enable()


model.train()
for epoch in range(10):
    for batch in train_data:
        optimizer.zero_grad()
        
        output = model(**batch)
        loss = output.loss
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.backward(loss)
        optimizer.step()

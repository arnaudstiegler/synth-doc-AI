import logging
from functools import partial
import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, BitsAndBytesConfig
from torch.utils.data import default_collate
import bitsandbytes as bnb
import torch
from donut_train import KVDataset, MISSING_TOKEN, custom_collate_fn


# For now, not using quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )


dataset_path = "/home/ubuntu/synth_data/"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = KVDataset(dataset_path, "train", test_run=True)
eval_dataset = KVDataset(dataset_path, "val", test_run=True)

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


deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
accelerator = Accelerator(
    mixed_precision="bf16", deepspeed_plugin=deepspeed_plugin, log_with="wandb"
)

accelerator.init_trackers(
    project_name="huggingface", init_kwargs={"wandb": {"entity": "arnaud-stiegler"}}
)

train_data = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=2,
    collate_fn=custom_collate_fn,
)
val_data = torch.utils.data.DataLoader(
    eval_dataset,
    shuffle=False,
    batch_size=4,
    collate_fn=custom_collate_fn,
)

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-5)
model, optimizer, train_data, val_data = accelerator.prepare(model, optimizer, train_data, val_data)
model.gradient_checkpointing_enable()


model.train()
total_step = 0
for epoch in range(10):
    for batch in train_data:
        optimizer.zero_grad()

        output = model(**batch)
        loss = output.loss
        logger.info(f"Loss: {loss.item()}", main_process_only=True)
        accelerator.log({"train_loss": loss.item()}, step=total_step)
        accelerator.backward(loss)
        optimizer.step()

        total_step += 1

        if total_step % 10 == 0:
            # Run eval
            model.eval()
            with torch.no_grad():
                loss_avg = []
                for eval_batch in val_data:
                    output = model(**eval_batch)
                    loss = output.loss
                    loss_avg.append(loss.cpu())

                accelerator.log({"val_loss": np.mean(loss_avg)}, step=total_step)

# Make sure that the wandb tracker finishes correctly
accelerator.end_training()

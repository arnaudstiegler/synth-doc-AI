import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaForConditionalGeneration
import torch
import json
from datasets import load_dataset
from transformers import AutoProcessor
import bitsandbytes as bnb
from pali_gemma.utils import collate_fn
from functools import partial

accelerator = Accelerator()
device_index = accelerator.process_index

# the 896 will run on 20GB with batch size 1
model_id = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("arnaudstiegler/synthetic_us_passports_easy")

processor.max_length = 128

collate = partial(collate_fn, processor)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": device_index}
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

data = torch.utils.data.DataLoader(
    dataset["train"], shuffle=True, collate_fn=collate_fn, batch_size=1
)

model, optimizer, data = accelerator.prepare(model, optimizer, data)


for epoch in range(100):
    for batch in data:
        model.train()
        optimizer.zero_grad()

        output = model(**batch)
        loss = output.loss

        # This is an ugly fix
        loss.requires_grad_(True)
        print(loss)

        accelerator.backward(loss)
        optimizer.step()

    # if epoch % 10 == 0:
    #     with torch.no_grad():
    #         model.eval()
    #         out = model.generate(**batch, max_new_tokens=128)
    #         print(f'epoch {epoch}', processor.batch_decode(out, skip_special_tokens=True))

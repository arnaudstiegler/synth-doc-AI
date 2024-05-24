import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaForConditionalGeneration
import torch
from datasets import load_dataset
from transformers import AutoProcessor
import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments
from pali_gemma.utils import collate_fn
from functools import partial


model_id = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(model_id)
collate = partial(collate_fn, processor)

dataset = load_dataset("arnaudstiegler/synthetic_us_passports_easy")

processor.max_length = 128


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
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

args = TrainingArguments(
    output_dir="/home/ubuntu/out/",
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=2,
    learning_rate=1e-4,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=1,
    optim="adamw_bnb_8bit",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=False,
    save_total_limit=1,
    bf16=True,
    # report_to=["tensorboard"],
    dataloader_pin_memory=False,
    # FSDP arguments
    # fsdp='full_shard',
    # Torch compile fails for now
    # torch_compile=True,
    # torch_compile_backend='inductor'

)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collate,
    args=args,
)
print(trainer.is_model_parallel)

trainer.train()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.optim import AdamW

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

dataset = load_dataset("squad")

train_dataset = dataset["train"]
optimizer = AdamW(model.parameters())


for sample in train_dataset:
    inputs = tokenizer(sample["question"], return_tensors="pt")
    answer = tokenizer(
        sample["question"] + sample["answers"]["text"][0], return_tensors="pt"
    )
    labels = torch.concat(
        [
            torch.tensor([-100] * inputs["input_ids"].shape[1]).unsqueeze(0),
            answer["input_ids"],
        ],
        dim=-1,
    )

    output = model(input_ids=inputs["input_ids"], labels=labels)
    import ipdb

    ipdb.set_trace()

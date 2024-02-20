from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.optim import AdamW


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

dataset = load_dataset("squad")

train_dataset = dataset["train"]
optimizer = AdamW(model.parameters())

sample = train_dataset[0]
inputs_tokens = tokenizer(sample["question"], return_tensors="pt").to(device)
answer_tokens = tokenizer(sample["answers"]["text"][0], return_tensors="pt").to(device)

# TODO: overfitting on a single sample for now
for _ in range(100):
    optimizer.zero_grad()
    input_ids = torch.concat(
        [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
    )
    labels = torch.concat(
        [
            torch.tensor([-100] * inputs_tokens["input_ids"].shape[1]).unsqueeze(0),
            answer_tokens["input_ids"],
        ],
        dim=-1,
    ).to(device)

    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss
    print(loss.item())
    loss.backward()
    optimizer.step()

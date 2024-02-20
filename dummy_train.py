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


# TODO: overfitting on a single sample for now
for sample in [train_dataset[0]]:
    optimizer.zero_grad()
    inputs_tokens = tokenizer(sample["question"], return_tensors="pt")
    answer_tokens = tokenizer(sample["answers"]["text"][0], return_tensors="pt")

    input_ids = torch.concat(
        [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
    )
    labels = torch.concat(
        [
            torch.tensor([-100] * inputs_tokens["input_ids"].shape[1]).unsqueeze(0),
            answer_tokens["input_ids"],
        ],
        dim=-1,
    )

    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss
    print(loss)
    loss.backward()
    optimizer.step()

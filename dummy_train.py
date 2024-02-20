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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/phi-2", torch_dtype='auto' if torch.cuda.is_available() else torch.float32,
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2",
#     code_revision='main'
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

config = read_deepspeed_config()
accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float16,
                                       attn_implementation="flash_attention_2").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

dataset = load_dataset("squad")

train_dataset = dataset["train"]

# with accelerator.main_process_first():
#     # partial_collate_func = partial(collator, processor)
#     # For Donut, we truncate to max length and can use the default collate
#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=config["train_micro_batch_size_per_gpu"],
#         # collate_fn=partial_collate_func
#     )
#
# # TODO: overfitting on a single sample for now
# with torch.cuda.amp.autocast():
#     for _ in range(100):
#         optimizer.zero_grad()
#         input_ids = torch.concat(
#             [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
#         )
#         labels = torch.concat(
#             [
#                 torch.tensor([-100] * inputs_tokens["input_ids"].shape[1], device=device).unsqueeze(
#                     0),
#                 answer_tokens["input_ids"],
#             ],
#             dim=-1,
#         ).to(device)
#
#         output = model(input_ids=input_ids, labels=labels)
#         loss = output.loss
#         print(loss.item())
#         loss.backward()
#         optimizer.step()



accelerator = Accelerator()

model = torch.nn.Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters())

data = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()

inputs_tokens = tokenizer(train_dataset[0]['question'])
answer_tokens = tokenizer(train_dataset[0]['answers'][0])

for epoch in range(10):
    for _ in data:

        input_ids = torch.concat(
        [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
        )
        labels = torch.concat(
        [
        torch.tensor([-100] * inputs_tokens["input_ids"].shape[1], device=device).unsqueeze(
            0),
        answer_tokens["input_ids"],
        ],
        dim=-1,
        ).to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.cross_entropy(output, targets)

        loss.backward()
        accelerator.backward(loss)
        optimizer.step()
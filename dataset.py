from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from typing import Dict, List
import torch.nn.functional as F


MAX_LENGTH = 100


def truncate_or_pad_tensor_right(tensor, max_width, pad_value=0):
    current_width = tensor.size(1)
    padding_needed = max_width - current_width

    if padding_needed > 0:
        # Pad only on the right side
        padding = (
            0,
            padding_needed,
        )  # For a 2D tensor, the padding tuple is (left, right)
        padded_tensor = F.pad(tensor, padding, "constant", value=pad_value)
    elif current_width > max_width:
        # Truncate the tensor from the right
        padded_tensor = tensor[:, :max_width]
    else:
        # No padding needed, return the original tensor
        padded_tensor = tensor

    return padded_tensor


def collate_fn(samples: List[Dict[str, torch.Tensor]]):
    input_ids = torch.concat(
        [
            truncate_or_pad_tensor_right(elem["input_ids"].unsqueeze(0), 100, -1)
            for elem in samples
        ]
    )
    labels = torch.concat(
        [
            truncate_or_pad_tensor_right(elem["labels"].unsqueeze(0), MAX_LENGTH, -1)
            for elem in samples
        ]
    )
    return {"input_ids": input_ids, "labels": labels}


class SquadDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, split: str):
        # Upfront the cost of loading the entire dataset (not that big anyway)
        self.samples = list(load_dataset("squad")[split])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        inputs_tokens = self.tokenizer(sample["question"], return_tensors="pt")
        answer_tokens = self.tokenizer(
            sample["answers"]["text"][0], return_tensors="pt"
        )

        # Concatenate the question and the answer
        input_ids = torch.concat(
            [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
        )
        # Mask out the tokens from the question
        labels = torch.concat(
            [
                torch.tensor([-100] * inputs_tokens["input_ids"].shape[1]).unsqueeze(0),
                answer_tokens["input_ids"],
            ],
            dim=-1,
        )
        return {"input_ids": input_ids.squeeze(0), "labels": labels.squeeze(0)}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    dataset = SquadDataset(tokenizer, "train")
    data = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=2, collate_fn=collate_fn
    )
    sample = next(iter(data))
    import ipdb

    ipdb.set_trace()

from datasets import load_dataset
from torch.utils.data import Dataset, default_collate
from transformers import AutoTokenizer, AutoProcessor
import torch
from typing import Dict, List
import torch.nn.functional as F


MAX_LENGTH = 100


def truncate_or_pad_tensor_right(tensor, max_width, pad_value):
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


class SquadDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, split: str):
        # Upfront the cost of loading the entire dataset (not that big anyway)
        self.samples = list(load_dataset("squad")[split])
        self.tokenizer = tokenizer

    @staticmethod
    def collate_fn(pad_token_id: int, samples: List[Dict[str, torch.Tensor]]):
        input_ids = torch.concat(
            [
                truncate_or_pad_tensor_right(
                    elem["input_ids"].unsqueeze(0), MAX_LENGTH, pad_token_id
                )
                for elem in samples
            ]
        )
        labels = torch.concat(
            [
                truncate_or_pad_tensor_right(
                    elem["labels"].unsqueeze(0), MAX_LENGTH, -100
                )
                for elem in samples
            ]
        )
        return {"input_ids": input_ids, "labels": labels}

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


class TextVQADataset(Dataset):
    def __init__(self, processor: AutoProcessor, split: str):
        # Upfront the cost of loading the entire dataset (not that big anyway)
        self.samples = load_dataset("textvqa")[split]
        self.processor = processor

    @staticmethod
    def collate_fn(pad_token_id: int, samples: List[Dict[str, torch.Tensor]]):
        import ipdb

        ipdb.set_trace()
        input_ids = torch.concat(
            [
                truncate_or_pad_tensor_right(
                    elem["input_ids"].unsqueeze(0), MAX_LENGTH, pad_token_id
                )
                for elem in samples
            ]
        )
        labels = torch.concat(
            [
                truncate_or_pad_tensor_right(
                    elem["labels"].unsqueeze(0), MAX_LENGTH, -100
                )
                for elem in samples
            ]
        )
        pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        inputs = self.processor(
            text=sample["question"], images=sample["image"], return_tensors="pt"
        )
        question_tokens = self.processor.tokenizer(
            sample["question"], return_tensors="pt", add_special_tokens=False
        )
        answer_tokens = self.processor.tokenizer(
            sample["answers"][0], return_tensors="pt", add_special_tokens=False
        )
        # Concatenate the question and the answer
        input_ids = torch.concat(
            [
                inputs["input_ids"],
                answer_tokens["input_ids"],
                torch.tensor([self.processor.tokenizer.eos_token_id]).unsqueeze(0),
            ],
            dim=-1,
        )
        # Mask out the tokens from the question
        labels = torch.concat(
            [
                torch.tensor([-100] * inputs["input_ids"].shape[1]).unsqueeze(0),
                answer_tokens["input_ids"],
                torch.tensor([self.processor.tokenizer.eos_token_id]).unsqueeze(0),
            ],
            dim=-1,
        )
        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "question_tensor": inputs["input_ids"],
        }


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = TextVQADataset(processor, "validation")
    from functools import partial

    collate = partial(TextVQADataset.collate_fn, processor.tokenizer.pad_token_id)
    data = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=2, collate_fn=collate
    )
    sample = next(iter(data))
    import ipdb

    ipdb.set_trace()

import json

from transformers import AutoProcessor


def collate_fn(processor: AutoProcessor, examples):
    texts = ["Process " for _ in examples]
    labels = [
        json.dumps({k: v for k, v in example.items() if k != "image"})
        for example in examples
    ]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
        tokenize_newline_separately=False,
    )
    return tokens

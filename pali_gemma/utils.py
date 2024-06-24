import json

from transformers import AutoProcessor
from typing import Optional
import re
import json


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


def extract_and_parse_json(input_string: str) -> Optional[str]:
    pattern = r"\{(.*)\}"
    match = re.search(pattern, input_string, re.S)
    if match:
        try:
            json_content = "{" + match.group(1) + "}"
            parsed_json = json.loads(json_content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None
    return None

import torch
from PIL import ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
from datasets import load_dataset
import re
import json
from pali_gemma.utils import extract_and_parse_json
from typing import Tuple, List

PALIGEMMA_IMAGE_SIZE = 896
PALIGEMMA_PATCH_SIZE = 14


def generate_covering_bbox(
    attentions: List[List[torch.Tensor]],
    patch_size: int = PALIGEMMA_PATCH_SIZE,
    img_size: int = PALIGEMMA_IMAGE_SIZE,
) -> Tuple[int, int, int, int]:
    # This grid gives the mapping between patch index and patch coordinates
    grid = {}
    for k in range((img_size // patch_size) * (img_size // patch_size)):
        col_idx = k % (img_size // patch_size)
        row_idx = k // (img_size // patch_size)
        # In format x1,y1,x2,y2
        grid[k] = [
            col_idx * patch_size,
            row_idx * patch_size,
            (col_idx + 1) * patch_size,
            (row_idx + 1) * patch_size,
        ]

    top = []
    # We start from 1 because we're skipping the original self-attention across all tokens
    # TODO: maybe we could bring it back
    for idx in range(1, len(attentions)):
        # Stack across layers
        img_attention = torch.stack([x for x in attentions[idx]], axis=-1)

        # Average the attentions across layers
        avg_img_attention = torch.mean(img_attention, axis=-1)
        # Average the attention across heads
        avg_img_attention = torch.mean(avg_img_attention, axis=1)

        # Only take the attention scores corresponding to the image
        img_attentions = avg_img_attention[:, :, : img_tokens[1][-1]]

        # Only look at absolute value of the attention
        att_score = torch.abs(img_attentions)
        # Take top-k image patches
        patch_indices = torch.topk(att_score, k=5, dim=-1).indices

        # Given a patch index, and a grid, retrieve the location of the different image patches on the image
        top_bbox = [
            tuple(grid[patch_idx.item()]) for patch_idx in patch_indices.flatten()
        ]
        top += top_bbox

    # Get coordinates of the bbox that covers all top-k image patches (should be the reason of interest)
    min_x, min_y, max_x, max_y = PALIGEMMA_IMAGE_SIZE, PALIGEMMA_IMAGE_SIZE, 0, 0
    for box in set(top):
        min_x, min_y, max_x, max_y = (
            min(min_x, box[0]),
            min(min_y, box[1]),
            max(max_x, box[2]),
            max(max_y, box[3]),
        )
    return min_x, min_y, max_x, max_y


def crop_image(normalized_bbox: List[int], image: Image):
    width, height = image.size
    min_x, min_y, max_x, max_y = normalized_bbox
    factor_width = width / PALIGEMMA_IMAGE_SIZE
    factor_height = height / PALIGEMMA_IMAGE_SIZE
    covering_bbox = [
        factor_width * min_x,
        factor_height * min_y,
        factor_width * max_x,
        factor_height * max_y,
    ]
    return image.crop(covering_bbox)


PALIGEMMA_IMAGE_TOKEN_ID = 257152

dataset = load_dataset("arnaudstiegler/v2_synthetic_us_passports_easy")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
dataset = load_dataset("arnaudstiegler/v2_synthetic_us_passports_easy")
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "google/paligemma-3b-pt-896"
adapter_model = "arnaudstiegler/paligemma-3b-pt-896-us-passports-lora-adapters"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_model, quantization_config=bnb_config
)
model = PeftModel.from_pretrained(model, adapter_model).to(device)
processor = AutoProcessor.from_pretrained(base_model)

model = model.eval().to("cuda")

regular_preds = []
zoomin_preds = []
for sample in tqdm(dataset["test"]):
    image = sample["image"].convert("RGB")
    inputs = processor(text="Process ", images=image, return_tensors="pt")
    inputs = {
        "input_ids": inputs["input_ids"].to("cuda"),
        "attention_mask": inputs["attention_mask"].to("cuda"),
        "pixel_values": inputs["pixel_values"].to("cuda"),
    }
    out = model.generate(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=164,
    )
    regular_preds.append(out.sequences.detach().cpu())

    img_tokens = torch.where(inputs["input_ids"] == PALIGEMMA_IMAGE_TOKEN_ID)

    min_x, min_y, max_x, max_y = generate_covering_bbox(out.attentions)
    image = crop_image([min_x, min_y, max_x, max_y], image)

    # Re-run with the zoomin
    inputs = processor(text="Process ", images=image, return_tensors="pt")
    inputs = {
        "input_ids": inputs["input_ids"].to("cuda"),
        "attention_mask": inputs["attention_mask"].to("cuda"),
        "pixel_values": inputs["pixel_values"].to("cuda"),
    }
    out = model.generate(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=164,
    )
    zoomin_preds.append(out.sequences.detach().cpu())

reg_acc = []
zoomin_acc = []
for sample, reg_pred, zoomin_pred in zip(dataset["test"], regular_preds, zoomin_preds):
    reg_pred = extract_and_parse_json(reg_pred[0])
    zoomin_pred = extract_and_parse_json(zoomin_pred[0])
    gt = {k: v for k, v in sample.items() if k not in ["image", "document_id"]}
    for k in gt.keys():
        reg_acc.append(reg_pred[k] == gt[k])
        zoomin_acc.append(zoomin_pred[k] == gt[k])

print(sum(reg_acc) / len(reg_acc))
print(sum(zoomin_acc) / len(zoomin_acc))

"""
reg: 0.6348039215686274
zoomin: 0.6850490196078431
"""

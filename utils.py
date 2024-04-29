import numpy as np
import json
import os
import re


def get_num_trainable_params(model) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def read_json(path: str):
    with open(path) as f:
        file = json.load(f)
    return file


def read_deepspeed_config(load_wandb_config: bool = True):
    config = read_json("configs/deepspeed_config.json")

    # The base config will overwrite some of the "auto" config from deepspeed
    config.update(read_json("configs/base_config.json"))

    # Reading the wandb config as well
    if os.path.exists("configs/wandb_config.json") and load_wandb_config:
        wandb_config = read_json("configs/wandb_config.json")
        config["wandb"] = wandb_config

    return config


def extract_coordinates(text):
    # Regular expression to find coordinates
    x_coords = [int(coord) for coord in re.findall(r"<x_coords_(\d+)>", text)]
    y_coords = [int(coord) for coord in re.findall(r"<y_coords_(\d+)>", text)]
    return {"x": x_coords, "y": y_coords}


def extract_box_coordinates(text):
    # Regular expression to find the bounding box coordinates
    pattern = r"<box>\((\d+),(\d+),(\d+),(\d+)\)</box>"
    match = re.search(pattern, text)
    if match:
        # Convert all groups of digits into integers
        return {"box": list(map(int, match.groups()))}
    return {"box": []}

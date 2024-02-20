import requests
from PIL import Image
from transformers import AutoProcessor, Pix2StructVisionModel
import torch
from utils import get_num_trainable_params

image_processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-large")
model = Pix2StructVisionModel.from_pretrained("google/pix2struct-docvqa-large")

print(get_num_trainable_params(model))

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# This is [BS, 2048, 768]
print(last_hidden_states.shape)

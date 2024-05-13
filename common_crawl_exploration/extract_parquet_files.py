from transformers import pipeline
import os
from PIL import Image
import pandas as pd
import requests
checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

parquet_files_location = '/home/ubuntu/cc_parquet/'

for file in os.listdir(parquet_files_location):
    if file.endswith(('parquet')):
        df = pd.read_parquet(os.path.join(parquet_files_location, file))
        for _, row in df.iterrows():
            image_url = row['url']
            try:
                image = Image.open(requests.get(image_url))
                predictions = detector(image, candidate_labels=["textual image", "visual image"])
                print(predictions)
            except:
                continue

        
import praw
from io import BytesIO
from PIL import Image
import requests
import os
import json
from tqdm import tqdm

reddit = praw.Reddit(
)

dest_folder = "test_run/"


def query_and_save(url: str, img_index: int):
    img_data = requests.get(url)
    image_bytes = BytesIO(img_data.content)
    image = Image.open(image_bytes)
    image.save(os.path.join(dest_folder, "images", f"image_{img_index}.png"))


if os.path.exists(os.path.join(dest_folder, "metadata.json")):
    meta = json.load(open(os.path.join(dest_folder, "metadata.json")))
else:
    meta = {}

img_index = len(meta)
for submission in tqdm(reddit.subreddit("Handwriting").new(limit=10000)):
    if not submission.url.endswith(("png", "jpeg", "jpg")) and hasattr(
        submission, "media_metadata"
    ):
        for val in submission.media_metadata.values():
            url = val["s"]["u"]
            if url in meta:
                print("passing the url, already done")
                break
            img_name = f"img_{img_index}.png"
            meta[url] = img_name
            query_and_save(url, img_index)
            img_index += 1
    elif submission.url.endswith(("png", "jpeg", "jpg")):
        url = submission.url
        if url in meta:
            print("passing the url, already done")
            continue
        img_name = f"img_{img_index}.png"
        meta[url] = img_name
        query_and_save(url, img_index)
        img_index += 1
    else:
        print(f"cannot process {submission.url}")

    json.dump(meta, open(os.path.join(dest_folder, "metadata.json"), "w"))

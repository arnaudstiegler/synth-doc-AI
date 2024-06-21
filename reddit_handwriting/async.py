import os
import json
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
import asyncpraw
from tqdm.asyncio import tqdm

dest_folder = "test_run/"


async def query_and_save(url: str, img_index: int, session):
    async with session.get(url) as response:
        img_data = await response.read()
        image_bytes = BytesIO(img_data)
        image = Image.open(image_bytes)
        image.save(os.path.join(dest_folder, "images", f"image_{img_index}.png"))


async def process_submission(submission, meta, img_index, session):
    tasks = []
    if not submission.url.endswith(("png", "jpeg", "jpg")) and hasattr(
        submission, "media_metadata"
    ):
        for val in submission.media_metadata.values():
            url = val["s"]["u"]
            if url in meta:
                print("passing the url, already done")
                continue
            img_name = f"img_{img_index}.png"
            meta[url] = img_name
            tasks.append(query_and_save(url, img_index, session))
            img_index += 1
    elif submission.url.endswith(("png", "jpeg", "jpg")):
        url = submission.url
        if url in meta:
            print("passing the url, already done")
            return img_index
        img_name = f"img_{img_index}.png"
        meta[url] = img_name
        tasks.append(query_and_save(url, img_index, session))
        img_index += 1
    else:
        print(f"cannot process {submission.url}")

    if tasks:
        await asyncio.gather(*tasks)

    return img_index


async def main():
    if os.path.exists(os.path.join(dest_folder, "metadata.json")):
        meta = json.load(open(os.path.join(dest_folder, "metadata.json")))
    else:
        meta = {}

    if not os.path.exists(os.path.join(dest_folder, "images")):
        os.makedirs(os.path.join(dest_folder, "images"))

    img_index = len(meta)
    reddit = asyncpraw.Reddit(
        client_id="",
        client_secret="",
        user_agent="",
    )

    async with aiohttp.ClientSession() as session:
        tasks = []
        subreddit = await reddit.subreddit("Handwriting")
        async for submission in subreddit.new():
            tasks.append(process_submission(submission, meta, img_index, session))

        results = await tqdm.gather(*tasks)
        img_index = max(results, default=img_index)

    with open(os.path.join(dest_folder, "metadata.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    asyncio.run(main())

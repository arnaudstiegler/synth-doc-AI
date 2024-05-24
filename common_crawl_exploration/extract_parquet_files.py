import asyncio
import csv
import os
from io import BytesIO

import aiohttp
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

parquet_files_location = "/home/ubuntu/parquet_files/"
output_directory = "/home/ubuntu/synth-doc-AI/common_crawl_exploration/test/"
checkpoint = "openai/clip-vit-large-patch14"
metadata_file = "/home/ubuntu/synth-doc-AI/common_crawl_exploration/metadata.csv"


async def fetch_image(
    session, url, row_uid, detector, semaphore, writer, lock, csvfile
):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                # Read the image stream
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                if image.width * image.height >= 500 * 500:
                    predictions = detector(
                        image, candidate_labels=["document scan", "image"]
                    )
                    if predictions[0]["label"] == "document scan":
                        image_path = os.path.join(output_directory, f"{row_uid}.png")
                        image.save(image_path)

                        async with lock:
                            try:
                                writer.writerow([row_uid, url, image_path, predictions])
                                csvfile.flush()  # Ensure the file is flushed after each write
                                print(
                                    f"Written to CSV: {row_uid}, {url}, {image_path}, {predictions}"
                                )
                            except Exception as csv_error:
                                print(
                                    f"CSV Write Error: {csv_error} for row {row_uid}, {url}"
                                )
        except Exception as e:
            print(f"Skipped: {e}")


async def process_images(file, semaphore, writer, lock, csvfile):
    df = pd.read_parquet(os.path.join(parquet_files_location, file))
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_image(
                session,
                row["url"],
                row["uid"],
                detector,
                semaphore,
                writer,
                lock,
                csvfile,
            )
            for _, row in df.iterrows()
        ]
        await asyncio.gather(*tasks)


async def main():
    files = [
        file for file in os.listdir(parquet_files_location) if file.endswith("parquet")
    ]
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    lock = asyncio.Lock()

    with open(metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["uid", "url", "image_path", "predictions"])

        for file in tqdm(files):
            await process_images(file, semaphore, writer, lock, csvfile)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import csv

parquet_files_location = '/home/ubuntu/parquet_files/'
output_directory = '/home/ubuntu/synth-doc-AI/common_crawl_exploration/test/'
checkpoint = "openai/clip-vit-large-patch14"
metadata_file = '/home/ubuntu/synth-doc-AI/common_crawl_exploration/metadata.csv'

async def fetch_image(session, url, row_uid, detector, semaphore, writer, lock):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                # Read the image stream
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                if image.width * image.height >= 500 * 500:
                    predictions = detector(image, candidate_labels=["document scan", "image"])
                    if predictions[0]['label'] == 'document scan':
                        image_path = os.path.join(output_directory, f'{row_uid}.png')
                        image.save(image_path)

                        async with lock:
                            try:
                                writer.writerow([row_uid, url, image_path, predictions])
                                print(f"Written to CSV: {row_uid}, {url}, {image_path}, {predictions}")
                            except Exception as csv_error:
                                print(f"CSV Write Error: {csv_error} for row {row_uid}, {url}")
        except Exception as e:
            pass
            # print(f'Skipped: {e}')
            # async with lock:
            #     try:
            #         writer.writerow([row_uid, url, None, str(e)])
            #         print(f"Error written to CSV: {row_uid}, {url}, {str(e)}")
            #     except Exception as csv_error:
            #         print(f"CSV Write Error: {csv_error} for row {row_uid}, {url} with error {str(e)}")

async def process_images(file, semaphore, writer, lock):
    df = pd.read_parquet(os.path.join(parquet_files_location, file))
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, row['url'], row['uid'], detector, semaphore, writer, lock) for _, row in df.iterrows()]
        await asyncio.gather(*tasks)

async def main():
    files = [file for file in os.listdir(parquet_files_location) if file.endswith('parquet')]
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    lock = asyncio.Lock()

    with open(metadata_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['uid', 'url', 'image_path', 'predictions'])

        for file in tqdm(files):
            await process_images(file, semaphore, writer, lock)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

parquet_files_location = '/home/ubuntu/parquet_files/'
output_directory = '/home/ubuntu/synth-doc-AI/common_crawl_exploration/test/'
checkpoint = "openai/clip-vit-large-patch14"

async def fetch_image(session, url, row_uid, detector, semaphore):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                # Read the image stream
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                if image.width * image.height >= 500*500:
                    predictions = detector(image, candidate_labels=["document scan", "image"])
                    if predictions[0]['label'] == 'document scan':
                        image.save(os.path.join(output_directory, f'{row_uid}.png'))
        except Exception as e:
            print(f'Skipped: {e}')

async def process_images(file, semaphore):
    df = pd.read_parquet(os.path.join(parquet_files_location, file))
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, row['url'], row['uid'], detector, semaphore) for _, row in df.iterrows()]
        await asyncio.gather(*tasks)

async def main():
    files = [file for file in os.listdir(parquet_files_location) if file.endswith('parquet')]
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    for file in tqdm(files):
        await process_images(file, semaphore)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

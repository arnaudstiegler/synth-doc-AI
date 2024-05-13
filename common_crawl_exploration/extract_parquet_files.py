import os
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm

parquet_files_location = '/home/ubuntu/parquet_files/'

async def fetch_image(session, url, row_id):
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            # Read the image stream
            image_data = await response.read()
            image = Image.open(BytesIO(image_data))
            predictions = detector(image, candidate_labels=["text", "no_text"])
            print(predictions)
            image.save(f'/home/ubuntu/synth-doc-AI/common_crawl_exploration/test/image_{row_id}.png')
    except Exception as e:
        print(f'Skipped: {e}')

async def process_images(file):
    df = pd.read_parquet(os.path.join(parquet_files_location, file))
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, row['url'], idx) for idx, row in df.iterrows()]
        await asyncio.gather(*tasks)

async def main():
    files = [file for file in os.listdir(parquet_files_location) if file.endswith('parquet')]
    for file in tqdm(files):
        await process_images(file)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

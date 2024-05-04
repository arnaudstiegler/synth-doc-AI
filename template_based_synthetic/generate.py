import cv2
from faker import Faker
import numpy as np
import json
import os
from typing import List
from template_based_synthetic.utils import custom_metatype_fill, format_date
from typing import Any, Dict
from PIL import Image, ImageDraw, ImageFont, ImageOps
from html_based_synthetic.augraphy_pipeline import AUG_PIPE
from augraphy import default_augraphy_pipeline
import random
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import random
from multiprocessing import Pool
import click


template_info = json.load(open("template_based_synthetic/assets/metadata.json"))

background_image_folder_path = "template_based_synthetic/assets/background_images/"
background_images = [
    os.path.join(background_image_folder_path, file)
    for file in os.listdir(background_image_folder_path)
]

templates = [
    "template_based_synthetic/assets/622897914_cleanup.jpg",
    "template_based_synthetic/assets/template_1_cleanup.jpeg",
]

pipeline = default_augraphy_pipeline()


def create_grey_background(width, height):
    """
    Create a background image with a random grey shade.

    Args:
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    Image: A Pillow Image object with a random grey shade.
    """
    # Generate a random grey shade
    grey_value = random.randint(0, 255)
    grey_color = (grey_value, grey_value, grey_value)

    # Create a new image with the random grey color
    image = Image.new("RGBA", (width, height), grey_color)
    return image


def paste_on_random_background(image: Image, background_images: list):
    if random.random() > 0.0:
        # Select a random background image
        if random.random() > 0.7:
            width = random.randint(1000, 2500)
            height = random.randint(1000, 2500)
            background_image = create_grey_background(width, height)
        else:
            background_image_path = random.choice(background_images)
            background_image = Image.open(background_image_path).convert("RGBA")

        new_width = int(
            random.uniform(background_image.width / 3, (3 / 4) * background_image.width)
        )
        aspect_ratio = image.width / image.height
        new_height = min(int(new_width / aspect_ratio), background_image.height)
        image_resized = image.resize((new_width, new_height))

        # Random position
        x_offset = random.randint(0, background_image.width - image_resized.width)
        y_offset = random.randint(0, background_image.height - image_resized.height)

        # Random rotation without cropping
        angle = random.uniform(-180, 180)
        rotated_image = image_resized.rotate(angle, expand=True)

        # Ensure the background is large enough to fit the resized and rotated image
        background_image = ImageOps.fit(
            background_image,
            (
                max(background_image.width, rotated_image.width + x_offset),
                max(background_image.height, rotated_image.height + y_offset),
            ),
            method=0,
            bleed=0.0,
            centering=(0.5, 0.5),
        )

        background_image.paste(rotated_image, (x_offset, y_offset), rotated_image)
        return background_image
    else:
        return image


def paste_faker_data(image_path: str, template_metadata: Dict[str, Any]):
    """
    Takes an image and a list of tuples (bbox, faker metatype) and pastes faker generated values into the bounding boxes on the image.

    Args:
    image_path (str): Path to the image file.
    bbox_faker_list (list of tuples): List containing tuples of bounding box coordinates and faker metatype.

    Returns:
    np.array: Image array with text pasted in specified bounding boxes.
    """
    image = Image.open(image_path).convert("RGBA")
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    fake = Faker()

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    metadata = {}
    for field_name, field in template_metadata["fields"].items():
        bbox = field["bbox"]
        metatype = field["metatype"]

        if metatype["source"] == "faker":
            if hasattr(fake, metatype["value"]):
                fake_data = getattr(fake, metatype["value"])()
            else:
                raise AttributeError(f"Faker has no attribute '{metatype['value']}'.")

            # TODO: integrate this into the custom metatype logic
            if metatype["value"] == "date":
                fake_data = format_date(fake_data)

        elif metatype["source"] == "custom":
            fake_data = custom_metatype_fill(metatype["value"])
        else:
            raise ValueError(f'Source {field["source"]} is not supported.')

        metadata[field_name] = fake_data
        text_bbox = font.getbbox(fake_data)
        height_diff = np.abs(bbox["height"] - (text_bbox[3] - text_bbox[1]))
        draw.text(
            (bbox["x"], bbox["y"] + height_diff // 2),
            fake_data,
            font=font,
            fill="black",
        )

    return image, metadata


def process_image(sample_idx: int):
    template_path = random.choice(templates)
    template_metadata = template_info[os.path.basename(template_path)]
    result_image, sample_metadata = paste_faker_data(template_path, template_metadata)
    image = paste_on_random_background(result_image, background_images)
    image.save(f"test_run/sample_{sample_idx}.png")
    json.dump(sample_metadata, open(f"test_run/sample_{sample_idx}_metadata.json", "w"))
    return sample_metadata

def augment_image(sample_idx: str) -> None:
    augmented_image = pipeline(np.array(Image.open(f"test_run/sample_{sample_idx}.png").convert("RGB")))
    Image.fromarray(augmented_image).save(f"test_run/aug_sample_{sample_idx}.png")


@click.command()
@click.option("--num_samples", required=True, type=int)
@click.option("--run_augraphy", is_flag=True, show_default=True, default=False)
def run_generation(num_samples: int, run_augraphy: bool):
    metadata = []
    for sample_idx in range(num_samples):
        sample_metadata = process_image(sample_idx)
        metadata.append(sample_metadata)
    json.dump(metadata, open("test_run/metadata.json", "w"))

    if run_augraphy:
        num_processes = os.cpu_count() - 1
        with Pool(processes=num_processes) as pool:
            _ = list(
                tqdm(pool.imap(process_image, range(num_samples)), total=num_samples)
            )


if __name__ == "__main__":
    run_generation()

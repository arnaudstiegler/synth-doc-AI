import cv2
from faker import Faker
import numpy as np
import json
import os
from template_based_synthetic.utils import custom_metatype_fill
from typing import Any, Dict


# Get the correct bounding box
#


def paste_on_random_background(image, background_images):
    """
    Randomly selects a background image and pastes the given image onto it based on a 50% chance.

    Args:
    image (np.array): The image to be pasted onto a background.
    background_images (list of str): List of paths to background images.

    Returns:
    np.array: The resulting image after potentially pasting onto a background.
    """
    if np.random.rand() > 0.5:
        # Select a random background image
        background_image_path = np.random.choice(background_images)
        background_image = cv2.imread(background_image_path)
        if background_image is None:
            raise FileNotFoundError(
                f"Background image at {background_image_path} not found."
            )

        # Resize background to match the image dimensions if necessary
        if background_image.shape[:2] != image.shape[:2]:
            background_image = cv2.resize(
                background_image, (image.shape[1], image.shape[0])
            )

        # Blend the images
        result_image = cv2.addWeighted(background_image, 0.5, image, 0.5, 0)
        return result_image
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
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Initialize Faker
    fake = Faker()

    # Define font for cv2.putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 0)  # Black color in BGR
    thickness = 2

    # Loop through each bbox and faker metatype
    for field in template_metadata["fields"].values():
        bbox = field["bbox"]
        metatype = field["metatype"]

        if metatype["source"] == "faker":
            if hasattr(fake, metatype["value"]):
                fake_data = getattr(fake, metatype["value"])()
            else:
                raise AttributeError(f"Faker has no attribute '{metatype['value']}'.")
        elif metatype["source"] == "custom":
            fake_data = custom_metatype_fill(metatype["value"])
        else:
            raise ValueError(f'Source {field["source"]} is not supported.')

        # Put text on the image
        cv2.putText(
            image,
            fake_data,
            (bbox["x"], bbox["y"] + bbox["height"]),
            font,
            font_scale,
            color,
            thickness,
        )

    return image


# Example usage:
metadata = json.load(open("template_based_synthetic/assets/metadata.json"))
image_path = "template_based_synthetic/assets/622897914_cleanup.jpg"
template_metadata = metadata[os.path.basename(image_path)]
result_image = paste_faker_data(image_path, template_metadata)
cv2.imwrite("output.jpg", result_image)

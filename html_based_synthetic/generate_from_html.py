import numpy as np
from PIL import Image
from augraphy import *
from faker import Faker
from jinja2 import Environment, FileSystemLoader
import random
from html_based_synthetic.utils import get_random_metatype
import os
import json
from io import BytesIO
import base64
from html_based_synthetic.augraphy_pipeline import AUG_PIPE
from weasyprint import HTML, CSS
from typing import Optional
from html_based_synthetic.style_utils import generate_css
from tqdm import tqdm
from multiprocessing import Pool
import time
import click
from pdf2image import convert_from_path
from weasyprint.fonts import FontConfiguration
from PyPDF2 import PdfReader

# Enable WeasyPrint logging
# import logging
# from weasyprint.logger import LOGGER
# LOGGER.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

NUM_SAMPLES = 50000

# Set the width and height of the output image
DOCUMENT_WIDTH = 2480
DOCUMENT_HEIGHT = 3508
# Default PDF DPI is 72
SCALE_FACTOR = 200 / 72

SEPARATORS = [":", ";", "=", "->", "=>", " "]

pipeline = AUG_PIPE
fake = Faker()


def get_random_kv_pairs():
    with open("synth_data_gen/templates/llm_content/key_value.json") as f:
        kv_pairs = json.load(f)
        return random.sample(list(kv_pairs.items()), random.randint(1, 20))


def generate_random_kv_pairs(fake: Faker):
    out_list = []
    for k, v in get_random_kv_pairs():
        formated_key = [subword.capitalize() for subword in k.split("_")]
        out_list.append((" ".join(formated_key), str(getattr(fake, v)())))

    return out_list


def generate_random_kv_pairs_v2(fake: Faker, num_pairs: Optional[int] = None):
    out_list = []

    num_pairs = random.randint(1, 10) if not num_pairs else num_pairs
    for _ in range(num_pairs):
        metatype = get_random_metatype()
        if isinstance(metatype, list):
            metatype = random.choice(metatype)
        out_list.append((fake.word(), str(getattr(fake, metatype)())))

    return out_list


def generate_faker_image() -> str:
    image = Image.open(BytesIO(fake.image()))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def generate_augmented_png(out_dir: str, i: int):
    # Path to your PDF file
    pdf_path = os.path.join(out_dir, f"sample_{i}.pdf")

    # Convert PDF to a list of image objects
    images = convert_from_path(pdf_path, dpi=200)

    # Only keep the first page
    images[0].save(
        os.path.join(out_dir, f"sample_{i}.png"),
        "PNG",
    )

    img = Image.open(os.path.join(out_dir, f"sample_{i}.png")).convert("RGB")
    augmented = pipeline(np.array(img))
    Image.fromarray(augmented).save(os.path.join(out_dir, f"sample_{i}_aug.png"))
    # # To save some space
    os.remove(os.path.join(out_dir, f"sample_{i}.png"))
    os.remove(os.path.join(out_dir, f"sample_{i}.pdf"))


def get_words_with_bboxes(page):
    """
    Return a dictionary containing all words and their corresponding bounding boxes on the page.

    :param page: A PyMuPDF page object
    :return: A dictionary with words as keys and lists of bounding box tuples as values
    """
    words_dict = {}

    # Extract words and their bounding boxes
    word_list = page.get_text("words")  # List of words with their bounding boxes

    # Iterate through the list of words
    for word_info in word_list:
        word, bbox = word_info[4], word_info[:4]  # Extract word and its bbox
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            x1 * SCALE_FACTOR,
            y1 * SCALE_FACTOR,
            x2 * SCALE_FACTOR,
            y2 * SCALE_FACTOR,
        ]
        # Append the bbox to the list of bboxes for the word in the dictionary
        if word in words_dict:
            words_dict[word].append(scaled_bbox)
        else:
            words_dict[word] = [scaled_bbox]

    return words_dict


def generate_image(args):
    start = time.time()
    out_dir, image_index, env, component_env, template, use_augraphy, task = args

    template = env.get_template("random_macros.html")
    metadata = []
    macros = component_env.list_templates()

    components_to_add = []
    # We are capped at 12 from 'random_macros.html'
    MAX_MACRO = 12
    for _ in range(MAX_MACRO):
        component_mapping = {
            "utils_macro.html": {},
            "table.html": {
                "charges": [
                    {"amount": random.randint(0, 10000), "description": fake.word()}
                    for _ in range(random.randint(1, 5))
                ]
            },
            "multi_columns_kv.html": {
                "kv_pairs": generate_random_kv_pairs(fake),
                "separator": random.choice(SEPARATORS),
                "num_columns": random.randint(1, 5),
            },
            "footer.html": {"text": fake.text()},
            "paragraph.html": {"text": fake.text()},
            "header.html": {"text": fake.text()},
            "list.html": {
                "kv_pairs": generate_random_kv_pairs(fake),
                "separator": random.choice(SEPARATORS),
            },
            "image_badge.html": {"logo": generate_faker_image()},
            "timeline.html": {
                "kv_pairs": generate_random_kv_pairs_v2(fake),
                "separator": random.choice(SEPARATORS),
            },
            "structured_grid.html": {"kv_pairs": generate_random_kv_pairs_v2(fake)},
            "structured_box.html": {"kv_pairs": generate_random_kv_pairs_v2(fake, 1)},
            "two_line_list.html": {"kv_pairs": generate_random_kv_pairs_v2(fake)},
        }
        comp = random.choice(macros)
        current_comp = component_env.get_template(comp)
        data = component_mapping[comp]
        metadata.append(data)
        components_to_add.append(current_comp.render(**data))

    template_data = {f"macro{i}": comp for i, comp in enumerate(components_to_add)}

    # Render the template with data
    output = template.render(**template_data)
    css = generate_css()

    font_config = FontConfiguration()
    html = HTML(string=output)
    css = CSS(string=css, font_config=font_config)

    html.write_pdf(
        os.path.join(out_dir, f"sample_{image_index}.pdf"),
        stylesheets=[css],
        font_config=font_config,
    )

    pdf_file_path = os.path.join(out_dir, f"sample_{image_index}.pdf")

    # Postproc the kv pairs
    if task == "kv_pair":
        with open(pdf_file_path, "rb") as file:
            pdf = PdfReader(file)
            first_page_text = pdf.pages[0].extract_text()
        kv_pairs = []
        has_kv_pairs = False
        for elem in metadata:
            if "kv_pairs" in elem.keys():
                has_kv_pairs = True
                for k, v in elem["kv_pairs"]:
                    if (
                        " ".join(k.split()) in first_page_text
                        and " ".join(v.split()) in first_page_text
                    ):
                        kv_pairs.append((k, v))

        json.dump(
            kv_pairs,
            open(os.path.join(out_dir, f"kv_pairs_sample_{image_index}.json"), "w"),
        )
    elif task == "segmentation":
        # For the segmentation task
        import fitz

        pdf_document = fitz.open(pdf_file_path)

        # Select the page to analyze
        page_number = 0  # Adjust as needed for the page you're interested in
        page = pdf_document.load_page(page_number)
        page_words = get_words_with_bboxes(page)
        json.dump(
            page_words,
            open(os.path.join(out_dir, f"segmentation_sample_{image_index}.json"), "w"),
        )
    else:
        raise ValueError(f"Task {task} is not supported")

    if use_augraphy:
        # This could be run outside of the PDF generation script
        generate_augmented_png(out_dir, image_index)

    # return
    return True


@click.command()
@click.option("--out_dir", default="synth_data_gen/samples/", type=str)
@click.option(
    "--task",
    required=True,
    type=click.Choice(["segmentation", "kv_pair"]),
    help="Choose the task to perform.",
)
def generate_documents(out_dir: str, task: str) -> None:
    # TODO: this needs to be reworked
    template_folder = "synth_data_gen/templates/"
    templates = [file for file in os.listdir(template_folder) if file.endswith(".html")]
    template_env = Environment(loader=FileSystemLoader(template_folder))
    component_env = Environment(
        loader=FileSystemLoader("synth_data_gen/html_components")
    )

    args_list = [
        (out_dir, i, template_env, component_env, templates[1], True, task)
        for i in range(NUM_SAMPLES)
    ]

    with Pool(processes=os.cpu_count() // 2) as pool:
        out = list(tqdm(pool.imap(generate_image, args_list), total=len(args_list)))

    print(f"Proportion of docs having kv pairs: {sum(out) / len(out)}")


if __name__ == "__main__":
    generate_documents()

import numpy as np
from PIL import Image
from augraphy import *
from faker import Faker
from jinja2 import Environment, FileSystemLoader, Template
import random
from synth_data_gen.utils import read_file, get_random_metatype
import os
import json
from io import BytesIO
import base64
from augraphy import *
from weasyprint import HTML, CSS
import logging
from weasyprint.logger import LOGGER
from synth_data_gen.style_utils import generate_css
from tqdm import tqdm
from multiprocessing import Pool
import time
import click


# Enable WeasyPrint logging
# LOGGER.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

# NUM_SAMPLES = int(5e5)
NUM_SAMPLES = 20

# Set the width and height of the output image
DOCUMENT_WIDTH = 2480
DOCUMENT_HEIGHT = 3508

SEPARATORS = [':', ',', ';', '=', '->', '=>']

pipeline = default_augraphy_pipeline()
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


def generate_random_kv_pairs_v2(fake: Faker):
    out_list = []

    for _ in range(random.randint(1, 10)):
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
    from pdf2image import convert_from_path

    # Path to your PDF file
    pdf_path = os.path.join(out_dir, f"sample_{i}.pdf")

    # Convert PDF to a list of image objects
    images = convert_from_path(pdf_path)

    # Only keep the first page
    images[0].save(
        os.path.join(out_dir, f"sample_{i}.png"),
        "PNG",
    )

    img = Image.open(os.path.join(out_dir, f"sample_{i}.png")).convert("RGB")
    augmented = pipeline(np.array(img))
    Image.fromarray(augmented).save(os.path.join(out_dir, f"sample_{i}_aug.png"))
    # To save some space
    os.remove(os.path.join(out_dir, f"sample_{i}.png"))


def generate_image(args):
    start = time.time()
    out_dir, image_index, env, component_env, template, use_augraphy = args

    template = env.get_template("random_macros.html")
    metadata = []
    macros = component_env.list_templates()

    components_to_add = []
    # We are capped at 8 from 'random_macros.html'
    for _ in range(8):
        component_mapping = {
            "utils_macro.html": {},
            "table.html": {
                "charges": [
                    {"amount": random.randint(0, 10000), "description": fake.word()}
                    for _ in range(random.randint(1, 15))
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
            "list.html": {"kv_pairs": generate_random_kv_pairs(fake), "separator": random.choice(SEPARATORS)},
            "image_badge.html": {"logo": generate_faker_image()},
            "timeline.html": {"kv_pairs": generate_random_kv_pairs_v2(fake), "separator": random.choice(SEPARATORS)},
            "structured_grid.html": {"kv_pairs": generate_random_kv_pairs_v2(fake)},
            # TODO: this is ugly
            "structured_box.html": {"key": generate_random_kv_pairs_v2(fake)[0][0], "value": generate_random_kv_pairs_v2(fake)[0][1]}
        }
        comp = random.choice(macros)
        current_comp = component_env.get_template(comp)
        data = component_mapping[comp]
        metadata.append(data)
        components_to_add.append(current_comp.render(**data))

    template_data = {f"macro{i}": comp for i, comp in enumerate(components_to_add)}
    json.dump(metadata, open(os.path.join(out_dir, f"sample_{image_index}.json"), "w"))
    # Render the template with data
    output = template.render(**template_data)

    html = HTML(string=output)
    css = CSS(string=generate_css())
    html.write_pdf(
        os.path.join(out_dir, f"sample_{image_index}.pdf"),
        stylesheets=[css],
    )

    if use_augraphy:
        # This could be run outside of the PDF generation script
        generate_augmented_png(out_dir, image_index)

    return time.time() - start


@click.command()
@click.option("--out_dir", default="synth_data_gen/samples/", type=str)
def generate_documents(out_dir: str) -> None:
    # TODO: this needs to be reworked
    template_folder = "synth_data_gen/templates/"
    templates = [file for file in os.listdir(template_folder) if file.endswith(".html")]
    template_env = Environment(loader=FileSystemLoader(template_folder))
    component_env = Environment(
        loader=FileSystemLoader("synth_data_gen/html_components")
    )

    args_list = [
        (out_dir, i, template_env, component_env, templates[1], True)
        for i in range(NUM_SAMPLES)
    ]

    with Pool(processes=os.cpu_count() // 2) as pool:
        list(tqdm(pool.imap(generate_image, args_list), total=len(args_list)))


if __name__ == "__main__":
    generate_documents()

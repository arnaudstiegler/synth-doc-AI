import numpy as np
from PIL import Image
from augraphy import *
from faker import Faker
from jinja2 import Environment, FileSystemLoader, Template
import random
from synth_data_gen.utils import read_file
import os
import json
from io import BytesIO
import base64
from augraphy import *
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import logging
from weasyprint.logger import LOGGER

# Enable WeasyPrint logging
# LOGGER.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)


# Set the width and height of the output image
DOCUMENT_WIDTH = 2480
DOCUMENT_HEIGHT = 3508

pipeline = default_augraphy_pipeline()
fake = Faker()


def get_random_kv_pairs():
    with open(
        "/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/llm_content/key_value.json"
    ) as f:
        kv_pairs = json.load(f)
        return random.sample(list(kv_pairs.items()), random.randint(1, 20))


def generate_random_kv_pairs(fake: Faker):
    out_list = []
    for k, v in get_random_kv_pairs():
        formated_key = [subword.capitalize() for subword in k.split("_")]
        out_list.append((" ".join(formated_key), getattr(fake, v)()))

    return out_list


def generate_faker_image() -> str:
    image = Image.open(BytesIO(fake.image()))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_augmented_png(i: int):
    from pdf2image import convert_from_path
    # Path to your PDF file
    pdf_path = f"/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.pdf"

    # Convert PDF to a list of image objects
    images = convert_from_path(pdf_path)

    # Only keep the first page
    images[0].save(
        f"/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png",
        "PNG",
    )

    img = Image.open(
        f"/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png"
    ).convert("RGB")
    augmented = pipeline(np.array(img))
    Image.fromarray(augmented).save(
        f"/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}_aug.png"
    )


def generate_image(image_index: int, env: Environment, component_env: Environment, template: Template, use_augraphy=False):
    # chosen = random.choice(templates)
    # print(parse_jinja_variables(os.path.join(template_folder, chosen)))
    # template = env.get_template(chosen)
    template = env.get_template("random_macros.html")

    macros = component_env.list_templates()
    random_macros = random.sample(macros, 2)

    component_mapping = {
        'utils_macro.html': {},
        'table_macro.html': {'charges': [{'amount': 1, 'description': 'test'}]},
        'multi_columns_kv.html': {'kv_pairs': generate_random_kv_pairs(fake), 'num_columns': 1},
        'footer.html': {'text': fake.text()},
        'paragraph.html': {'text': fake.text()},
        'header.html': {'text': fake.text()}
        }

    components_to_add = []
    for comp in random_macros:
        current_comp = component_env.get_template(comp)
        # Here we need to populat ethe values that need to be populated
        # For now, ugly mapping:
        data = component_mapping[comp]
        components_to_add.append(current_comp.render(**data))
        
    template_data = {f'macro{i}': comp for i, comp in enumerate(components_to_add)}

    # kv_pairs = generate_random_kv_pairs(fake)

    # num_items = random.randint(0, 10)
    # charges = [
    #     {
    #         "description": f"Item {i}",
    #         "amount": float(f"{random.randint(0, 1000000)/100:.2f}"),
    #     }
    #     for i in range(num_items)
    # ]

    # terms = read_file(
    #     "/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/text_samples/terms.txt"
    # )
    terms = fake.text()

    # Read CSS content
    style_folder_path = "synth_data_gen/templates/static/"
    style_files = os.listdir(style_folder_path)

    # chosen_style_file = random.choice(style_files)
    # chosen_style_file = "style.css"
    # print(chosen_style_file)
    # with open(os.path.join(style_folder_path, chosen_style_file), "r") as css_file:
    #     css_content = css_file.read()

    # macro_template


    # Render the template with data
    output = template.render(
        **template_data
    )

    font_config = FontConfiguration()
    html = HTML(string=output)
    css = CSS("synth_data_gen/templates/static/style.css", font_config=font_config)
    html.write_pdf(
        f"/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{image_index}.pdf",
        stylesheets=[css],
        font_config=font_config,
    )

    if use_augraphy:
        generate_augmented_png(image_index)


def generate_documents():
    template_folder = "synth_data_gen/templates/"
    templates = [file for file in os.listdir(template_folder) if file.endswith(".html")]
    template_env = Environment(loader=FileSystemLoader(template_folder))
    component_env = Environment(loader=FileSystemLoader('synth_data_gen/html_components'))

    for i in range(5):
        generate_image(image_index=i, env=template_env, component_env=component_env, template=templates[1], use_augraphy=False)


if __name__ == '__main__':
    generate_documents()

import imgkit
import numpy as np
from PIL import Image
import io
from augraphy import *
import cv2
import matplotlib.pyplot as plt
from faker import Faker
from jinja2 import Environment, FileSystemLoader, Template
import random
from synth_data_gen.utils import read_file, parse_jinja_variables
import os
import pdfkit
import json
from io import BytesIO
import base64
from augraphy import *



def get_random_kv_pairs():
    with open('/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/llm_content/key_value.json') as f:
        kv_pairs = json.load(f)
        return random.sample(list(kv_pairs.items()), random.randint(1, 20))
    
def generate_random_kv_pairs(fake: Faker):
    out_list = []
    for k, v in get_random_kv_pairs():
        formated_key = [subword.capitalize() for subword in k.split('_')]
        out_list.append((' '.join(formated_key), getattr(fake, v)()))
    
    return out_list


pipeline = default_augraphy_pipeline()
fake = Faker()

template_folder = 'synth_data_gen/templates/'
templates = [file for file in os.listdir(template_folder) if file.endswith('.html')]
env = Environment(loader=FileSystemLoader(template_folder))

# Set the width and height of the output image
width = 2480
height = 3508


for i in range(5):
    # TODO: vary the template
    # NB: the env already has the template folder
    chosen = random.choice(templates)
    print(parse_jinja_variables(os.path.join(template_folder, chosen)))
    # template = env.get_template(chosen)
    template = env.get_template('invoice.html')


    kv_pairs = generate_random_kv_pairs(fake)

    num_items = random.randint(0,10)
    charges = [
        {"description": f'Item {i}', "amount": float(f'{random.randint(0, 1000000)/100:.2f}')} for i in range(num_items)
    ]

    terms = read_file('/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/text_samples/terms.txt')

    # Read CSS content
    style_folder_path = 'synth_data_gen/templates/static/'
    style_files = os.listdir(style_folder_path)

    chosen_style_file = random.choice(style_files)
    print(chosen_style_file)
    with open(os.path.join(style_folder_path, chosen_style_file), 'r') as css_file:
        css_content = css_file.read()

    image = Image.open(BytesIO(fake.image()))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # Render the template with data
    output = template.render(css=css_content, charges=charges, terms=terms, kv_pairs=kv_pairs, logo=img_str)    

    # Convert the HTML template to an image
    
    pdfkit.from_string(output, f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.pdf')

    # TODO: this call is buggy
    # imgkit.from_file(
    #     f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.pdf', 
    #     f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png'
    #     )

    use_augraphy=True
    if use_augraphy:
        # TODO: seems like the PDF formatting break down here
        img = imgkit.from_string(output, None, options={'format': 'png', 'width': width, 'height': height})

        img = Image.open(f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png').convert('RGB')
        augmented = pipeline(np.array(img))
        Image.fromarray(augmented).save(f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png')

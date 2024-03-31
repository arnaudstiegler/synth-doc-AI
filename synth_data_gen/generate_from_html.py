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


pipeline = default_augraphy_pipeline()
fake = Faker()

            # "first_name": self.fake.first_name(),
            # "last_name": self.fake.last_name(),
            # "date_of_birth": self.fake.date_of_birth().strftime('%Y-%m-%d'),
            # "social_security_number": self.fake.ssn(),
            # "address": self.fake.street_address(),
            # "city": self.fake.city(),
            # "state": self.fake.state(),
            # "zip_code": self.fake.zipcode(),
            # "email": self.fake.email(),
            # "phone_number": self.fake.phone_number()

# Set the path to your HTML file
# html_file_path = '/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/test.html'

template_folder = 'synth_data_gen/templates/'
templates = [file for file in os.listdir(template_folder) if file.endswith('.html')]
env = Environment(loader=FileSystemLoader(template_folder))



# # Read the template from the file
# with open(html_file_path, 'r') as file:
#     template = Template(file.read())



# Set the width and height of the output image
width = 2480
height = 3508


for i in range(5):
    # TODO: vary the template
    # NB: the env already has the template folder
    chosen = random.choice(templates)
    print(parse_jinja_variables(os.path.join(template_folder, chosen)))
    # template = env.get_template(chosen)
    template = env.get_template('test.html')

    data = {
    'Customer Name': str(fake.first_name() + ' ' + fake.last_name()),
    'Account Number': str(fake.numerify(text='INV-#####')),
    'Bill Date': str(fake.date()),
    'Service Start': '2022-01-01',
    'Service End': '2022-01-31',
    'Important Messages': str(' '.join(fake.words()))
    }

    kv_pairs = [(k, v) for k, v in data.items()]

    num_items = random.randint(0,10)
    charges = [
        {"description": f'Item {i}', "amount": 10} for i in range(num_items)
    ]

    terms = read_file('/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/text_samples/terms.txt')

    # Read CSS content
    style_folder_path = 'synth_data_gen/templates/static/'
    style_files = os.listdir(style_folder_path)

    chosen_style_file = random.choice(style_files)
    print(chosen_style_file)
    with open(os.path.join(style_folder_path, chosen_style_file), 'r') as css_file:
        css_content = css_file.read()
    # Render the template with data
    output = template.render(css=css_content, charges=charges, terms=terms, kv_pairs=kv_pairs)    

    # Convert the HTML template to an image
    # img = imgkit.from_string(output, None, options={'format': 'png', 'width': width, 'height': height})
    pdfkit.from_string(output, f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.pdf')

    # Convert the image to a PIL image
    # pil_img = Image.open(io.BytesIO(img))

    # Convert the PIL image to a NumPy array
    # np_img = np.array(pil_img)
    # pil_img.save(f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png')

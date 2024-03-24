import imgkit
import numpy as np
from PIL import Image
import io
from jinja2 import Template
from augraphy import *
import cv2
import matplotlib.pyplot as plt
from faker import Faker
from jinja2 import Environment, FileSystemLoader


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

template_dir = '/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/'
env = Environment(loader=FileSystemLoader(template_dir))



# # Read the template from the file
# with open(html_file_path, 'r') as file:
#     template = Template(file.read())



# Set the width and height of the output image
width = 2480
height = 3508



for i in range(5):
    # np_augmented = pipeline(np_img)
    # pil_augmented = Image.fromarray(np_augmented)
    template = env.get_template('test_variation.html')  # Replace 'test.html' with your actual template name
    data = {
    'customerName': fake.first_name() + ' ' + fake.last_name(),
    'accountNumber': fake.numerify(text='INV-#####'),
    'billDate': fake.date(),
    'serviceStart': '2022-01-01',
    'serviceEnd': '2022-01-31',
    # TODO should update that logic
    'gasDeliveryCharge': str(fake.random_number(digits=3)) + '.00',
    'gasSupplyCharge': str(fake.random_number(digits=3)) + '.00',
    'totalCurrentCharges': str(fake.random_number(digits=3)) + '.00',
    'importantMessages': fake.words()
    }
    charges = [
        {"description": fake.name(), "amount": fake.random_number(digits=5), "class": "gasDeliveryCharge", "metatype": "Number"},
        {"description": fake.name(), "amount": fake.random_number(digits=5), "class": "gasSupplyCharge", "metatype": "Number"},
        # Add more charges as needed
    ]

    # Render the template with data
    output = template.render(charges=charges)    

    # Convert the HTML template to an image
    img = imgkit.from_string(output, None, options={'format': 'png', 'width': width, 'height': height})

    # Convert the image to a PIL image
    pil_img = Image.open(io.BytesIO(img))

    # Convert the PIL image to a NumPy array
    np_img = np.array(pil_img)
    pil_img.save(f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png')

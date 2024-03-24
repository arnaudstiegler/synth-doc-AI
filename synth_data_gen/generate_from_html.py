import imgkit
import numpy as np
from PIL import Image
import io
from jinja2 import Template
from augraphy import *
import cv2
import matplotlib.pyplot as plt

pipeline = default_augraphy_pipeline()


# Set the path to your HTML file
html_file_path = '/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/test.html'

# Your data
data = {
    'customerName': 'John Doe',
    'accountNumber': '123456',
    'billDate': '2022-01-01',
    'serviceStart': '2022-01-01',
    'serviceEnd': '2022-01-31',
    'gasDeliveryCharge': '100.00',
    'gasSupplyCharge': '200.00',
    'totalCurrentCharges': '300.00',
    'importantMessages': 'Please pay your bill on time.'
}

# Read the template from the file
with open(html_file_path, 'r') as file:
    template = Template(file.read())

# Render the template with the data
output = template.render(data)

# Set the width and height of the output image
width = 2480
height = 3508

# Convert the HTML template to an image
img = imgkit.from_string(output, None, options={'format': 'png', 'width': width, 'height': height})

# Convert the image to a PIL image
pil_img = Image.open(io.BytesIO(img))

# Convert the PIL image to a NumPy array
np_img = np.array(pil_img)

for i in range(5):
    np_augmented = pipeline(np_img)
    pil_augmented = Image.fromarray(np_augmented)
    pil_augmented.save(f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_{i}.png')

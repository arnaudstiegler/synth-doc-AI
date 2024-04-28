import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
import torch


ground_truth = {
    "signature_doc.png": [
        {"field": "Matter no", "correct": "70104"},
        {"field": "Invoice #", "correct": "172882"},
    ],
    "email.png": [
        {"field": "To", "correct": "gibillis@"},
        {"field": "From", "correct": "matt.adent@"},
    ],
    "invoice.png": [
        {"field": "Invoice date and time", "correct": "12:48"},
        {"field": "Payment Terms", "correct": "30 days"},
        {"field": "Account Name", "correct": "XXXXX"},
    ],
    "passport_card.jpg": [
        {"field": "Date of Birth", "correct": "14 DEC 1998"},
        {"field": "Surname", "correct": "LOPEZ"},
    ],
    "drivers_license.jpg": [
        {"field": "ID", "correct": "123 456 789"},
        {"field": "DOB", "correct": "08/31/1982"},
    ],
    "in_distrib_1.png": [
        {"field": "Cvv", "correct": "1993-06-02"},
        {"field": "positive", "correct": "Estates"},
    ],
    "in_distrib_2.png": [
        {"field": "score", "correct": "6308487"},
        {"field": "Last Maintenance", "correct": "2024-04-07"},
    ]
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = '/Users/arnaudstiegler/Desktop/synth_data_run_v3/'
# model_path = "/home/ubuntu/synth_data_run_v3/"
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
processor.tokenizer.padding_side = "left"

for file in ground_truth.keys():
    img_path = "test_samples/"
    img = Image.open(os.path.join(img_path, file)).convert("RGB")
    plt.figure(figsize=(20, 20))
    plt.imshow(img)

    for elem in ground_truth[file]:
        # Breakdown to avoid the warning message
        pixel_values = processor(img, return_tensors="pt")
        labels = processor.tokenizer(
            elem["field"] + ":", return_tensors="pt", padding=True
        )
        pred = model.generate(
            pixel_values["pixel_values"].to(device),
            decoder_input_ids=labels["input_ids"][:, :-1].to(device),
            num_beams=1,
        )
        print(elem, processor.batch_decode(pred))

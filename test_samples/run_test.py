import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt


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
        {"field": "DOB", "correct": "08/31/2022"},
    ],
}

# model_path = '/Users/arnaudstiegler/Desktop/arnaud_gpu_synth/'
model_path = "/home/ubuntu/synth_data_run_v0/"
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained(model_path).to("cuda")
processor.tokenizer.padding_side = "left"

for file in ground_truth.keys():
    img_path = "/home/ubuntu/kv_samples/"
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
            pixel_values["pixel_values"].to("cuda"),
            decoder_input_ids=labels["input_ids"][:, :-1].to("cuda"),
            num_beams=1,
        )
        print(elem, processor.batch_decode(pred))
import os
import json
from PIL import Image

class ImageAnnotator:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.img_index = 0
        self.annotations = []
        self.load_progress()

    def annotate_images(self):
        while self.img_index < len(self.image_files):
            image_file = self.image_files[self.img_index]
            image_path = os.path.join(self.image_folder, image_file)
            self.display_image(image_path)
            
            question = input("Enter your question: ")
            answer = input("Enter your answer: ")
            pii_sensitive = input("Is this image PII sensitive? (yes/no): ").lower() == 'yes'
            
            annotation = {
                "image": image_file,
                "question": question,
                "answer": answer,
                "pii_sensitive": pii_sensitive
            }
            self.annotations.append(annotation)
            self.save_progress()
            
            next_action = input("Do you want to annotate the next image? (yes/no): ").lower()
            if next_action == 'no':
                break

            self.img_index += 1
        
        print("Annotation process completed.")

    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.show()
        except Exception as e:
            print(f"Unable to open image {image_path}. Error: {e}")

    def save_progress(self):
        data = {
            "index": self.img_index,
            "annotations": self.annotations
        }
        with open("progress.json", "w") as f:
            json.dump(data, f, indent=4)

    def load_progress(self):
        if os.path.exists("progress.json"):
            with open("progress.json", "r") as f:
                data = json.load(f)
                self.img_index = data.get("index", 0)
                self.annotations = data.get("annotations", [])

if __name__ == "__main__":
    image_folder = input("Enter the folder path containing the images: ")
    annotator = ImageAnnotator(image_folder)
    annotator.annotate_images()

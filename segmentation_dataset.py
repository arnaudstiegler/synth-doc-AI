from transformers import DonutProcessor
import os
import json
import random
from PIL import Image

Y_COORDS_TOKENS = {i: f"<y_coords_{i}>" for i in range(0, 1001)}
X_COORDS_TOKENS = {i: f"<x_coords_{i}>" for i in range(0, 1001)}


class SegmentationDataset:
    VAL_SPLIT_SIZE = 0.05

    def __init__(
        self, processor: DonutProcessor, folder_path: str, split: str, test_run: bool
    ):
        self.folder_path = folder_path
        self.split = split
        self.test_run = test_run

        self.docs = self.init_docs(split)
        self.processor = processor

    def init_docs(self, split: str):
        docs = []
        for file in os.listdir(self.folder_path):
            if "segmentation" in file:
                bboxes = json.load(open(os.path.join(self.folder_path, file)))
                sample_id = file.split("_")[-1].replace(".json", "")
                img_path = os.path.join(self.folder_path, f"sample_{sample_id}.png")
                if not os.path.exists(img_path):
                    # Skipping if we're missing the corresponding img (download issue)
                    print(f"Skipping sample {sample_id}: could not find image")
                    continue
                docs.append((bboxes, img_path))

        docs = sorted(docs, key=lambda x: x[1])

        if split == "train":
            train_split_size = int((1 - self.VAL_SPLIT_SIZE) * len(docs))
            split_docs = docs[:train_split_size]
        elif split == "val":
            val_split_size = int((self.VAL_SPLIT_SIZE) * len(docs))
            split_docs = docs[-val_split_size:]
        else:
            raise ValueError(
                f"Split should be either train or val but received {split}"
            )

        print(f"Split={self.split} size: {len(split_docs)}")
        return split_docs

    def __getitem__(self, i):
        bboxes, img_path = self.docs[i]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        word_to_segment = random.choice(list(bboxes.keys()))

        text_target = word_to_segment + " "
        for bbox in bboxes[word_to_segment]:
            x1, y1, x2, y2 = bbox
            text_target += (
                x_coords_tokens[int((x1 / width) * 1000)]
                + y_coords_tokens[int((y1 / height) * 1000)]
                + x_coords_tokens[int((x2 / width) * 1000)]
                + y_coords_tokens[int((y2 / height) * 1000)]
            )

        # Breakdown to avoid the warning message
        pixel_values = self.processor(img, return_tensors="pt")
        labels = self.processor.tokenizer(
            text_target,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        return {
            "pixel_values": pixel_values["pixel_values"],
            "labels": labels["input_ids"],
            "image_path": img_path,
            "image_size": img.size,
        }

    def __len__(self):
        if self.test_run:
            # For debugging purposes
            return 10
        return len(self.docs)


if __name__ == "__main__":
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.tokenizer.add_tokens(list(Y_COORDS_TOKENS.values()))
    processor.tokenizer.add_tokens(list(X_COORDS_TOKENS.values()))
    dataset = SegmentationDataset(
        processor,
        "/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples",
        "train",
        True,
    )
    for i in range(len(dataset)):
        for k in range(100):
            x = dataset[i]

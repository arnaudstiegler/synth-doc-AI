from PIL import Image
from transformers import DonutProcessor


import json
import os
import random


class KVDataset:
    VAL_SPLIT_SIZE = 0.05

    def __init__(self, folder_path: str, split: str, test_run: bool):
        self.folder_path = folder_path
        self.split = split
        self.test_run = test_run

        self.docs = self.init_docs(split)
        self.kv_pairs = list(self.get_kv_pairs())
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    def init_docs(self, split: str):
        docs = []
        missing_kv_pairs = []
        for file in os.listdir(self.folder_path):
            if "kv_pairs" in file:
                kv_pairs = json.load(open(os.path.join(self.folder_path, file)))
                sample_id = file.split("_")[-1].replace(".json", "")
                img_path = os.path.join(self.folder_path, f"sample_{sample_id}_aug.png")
                if not os.path.exists(img_path):
                    # Skipping if we're missing the corresponding img (download issue)
                    print(f"Skipping sample {sample_id}: could not find image")
                    continue
                elif len(kv_pairs) == 0:
                    print(f"Skipping sample {sample_id}: no kv_pair")
                    missing_kv_pairs.append(1)
                    continue
                missing_kv_pairs.append[0]
                docs.append((kv_pairs, img_path))

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

    def get_kv_pairs(self):
        kv_pairs_list = []
        for kv_pairs, _ in self.docs:
            kv_pairs_list += [k for k, _ in kv_pairs]
        return set(kv_pairs_list)

    def __getitem__(self, i):
        doc_kv, img_path = self.docs[i]
        img = Image.open(img_path).convert("RGB")

        if len(doc_kv) == 0:
            raise ValueError('Should always have kv_pairs')
            # random_key = random.choice(self.kv_pairs)
            # text_target = f"{random_key.lower() if random.random() < 0.2 else random_key}: {MISSING_TOKEN}"
        else:
            # TODO: should I change that for eval?
            k, v = random.choice(doc_kv)
            text_target = f"{k.lower() if random.random() < 0.2 else k}: {v}"

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
        }

    def __len__(self):
        if self.test_run:
            # For debugging purposes
            return 10
        return len(self.docs)
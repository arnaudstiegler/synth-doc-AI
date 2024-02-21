from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class SquadDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, split: str):
        # Upfront the cost of loading the entire dataset (not that big anyway)
        self.samples = list(load_dataset('squad')[split])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        inputs_tokens = self.tokenizer(sample['question'], return_tensors='pt')
        answer_tokens = self.tokenizer(sample['answers']['text'][0], return_tensors='pt')

        # Concatenate the question and the answer
        input_ids = torch.concat(
            [inputs_tokens["input_ids"], answer_tokens["input_ids"]], dim=-1
        )
        # Mask out the tokens from the question
        labels = torch.concat(
            [
                torch.tensor([-100] * inputs_tokens["input_ids"].shape[1]).unsqueeze(
                    0),
                answer_tokens["input_ids"],
            ],
            dim=-1,
        )
        return {'input_ids': input_ids, 'labels': labels}


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    dataset = SquadDataset(tokenizer, 'train')
    import ipdb; ipdb.set_trace()
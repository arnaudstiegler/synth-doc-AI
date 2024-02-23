import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = "cuda" if torch.cuda.is_available() else 'cpu'

# TODO: should we force it to return a json dict?
SYS_PROMPT = "You are an AI agent used for automation. Do not act like a chatbot. Return a json file as the output"


class Predictor:
    def generate_answer(self, prompt: str):
        raise NotImplementedError


class MistralOpenOrcaPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Open-Orca/Mistral-7B-OpenOrca",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Open-Orca/Mistral-7B-OpenOrca")

    def generate_answer(self, prompt: str):
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + SYS_PROMPT + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        generation_config = GenerationConfig(
            max_length=256, temperature=1.1, top_p=0.95, repetition_penalty=1.0,
            do_sample=True, use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id,
        )

        inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(
            device)
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        return self.post_process_output(outputs)

    def post_process_output(self, outputs):
        start_index = torch.where(outputs == torch.tensor(
            self.tokenizer.encode('assistant', add_special_tokens=False)).to(device))[1]
        return self.tokenizer.decode(outputs[0, start_index + 1:], skip_special_tokens=True).strip()


class MistralInstructPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1")

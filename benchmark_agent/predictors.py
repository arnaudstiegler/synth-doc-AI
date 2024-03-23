from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
# None means we use regular attention
ATTN_TO_USE = "flash_attention_2" if torch.cuda.is_available() else None

# TODO: should we force it to return a json dict?
# TODO: unify generation config
SYS_PROMPT = (
    "You are an AI agent used for automation. Do not act like a chatbot. Execute the task and"
    "follow the instructions for the formatting of the output"
)


@dataclass
class Sample:
    id: int
    type: str
    task_definition: str
    task_input: str
    json_expected: bool
    task_output_format: str
    expected_output: str


class Predictor:
    @staticmethod
    def format_sample_into_prompt(sample: Sample) -> str:
        return (
            sample.task_definition
            + " "
            + sample.task_output_format
            + "\n Input: \n"
            + sample.task_input
        )

    def generate_answer(self, sample: Sample) -> str:
        raise NotImplementedError

    def post_process_output(self, outputs: torch.Tensor) -> str:
        raise NotImplementedError


class MistralOpenOrcaPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Open-Orca/Mistral-7B-OpenOrca",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")

    def generate_answer(self, sample: Sample):
        prompt = self.format_sample_into_prompt(sample)
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + SYS_PROMPT + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        generation_config = GenerationConfig(
            max_new_tokens=256,
            # temperature=1.1,
            # top_p=0.95,
            # repetition_penalty=1.0,
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        inputs = self.tokenizer(
            input_text, return_tensors="pt", return_attention_mask=True
        ).to(device)
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        return self.post_process_output(outputs)

    def post_process_output(self, outputs: torch.Tensor) -> str:
        # Assuming the output will not contain "assistant" more than once
        return (
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            .split("assistant")[1]
            .strip()
        )


class MistralInstructPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )

    def generate_answer(self, sample: Sample):
        prompt = self.format_sample_into_prompt(sample)

        # BEWARE: Mistral instruct doesn't accept system prompt
        messages = [
            {"role": "user", "content": prompt},
        ]
        encodeds = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        generated_ids = self.model.generate(
            encodeds, max_new_tokens=1000, do_sample=True, use_cache=True
        )
        answer = self.post_process_output(generated_ids)
        return answer

    def post_process_output(self, outputs: torch.Tensor) -> str:
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(decoded, decoded[0].split("[/INST]")[1].strip())
        return decoded[0].split("[/INST]")[1].strip()

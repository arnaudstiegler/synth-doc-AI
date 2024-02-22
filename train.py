from datetime import datetime
from functools import partial

import click
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from dataset import SquadDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    Trainer,
    get_scheduler,
)
from transformers.utils.logging import set_verbosity_error
from dataset import collate_fn
from utils import read_deepspeed_config

logger = get_logger(__name__, log_level="INFO")
# Required to suppress the warning during .generate
set_verbosity_error()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        flattened_patches = inputs.pop("flattened_patches")
        attention_mask = inputs.pop("attention_mask")

        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss if not return_outputs else outputs


def pad_tensors_to_left(tensors, pad_value):
    # Find the maximum size along each dimension
    max_sizes = [
        max([tensor.size(dim) for tensor in tensors]) for dim in range(tensors[0].dim())
    ]

    # Pad the tensors on the left side
    padded_tensors = []
    for tensor in tensors:
        pad_config = [
            (max_size - tensor.size(dim), 0) for dim, max_size in enumerate(max_sizes)
        ]
        pad_config = [
            val for pad_tuple in pad_config for val in pad_tuple
        ]  # Flatten the pad_config list
        padded_tensors.append(F.pad(tensor, pad_config, value=pad_value))
    return torch.stack(padded_tensors, dim=0)


def evaluate_model(accelerator, model, val_dataloader, config, curr_step):
    val_loss = MeanMetric(nan_strategy="error").to(model.device)
    similarity = MeanMetric(nan_strategy="error").to(model.device)

    prediction_samples_list = []

    counter = 0

    with torch.no_grad():
        for batch in tqdm(
            val_dataloader, disable=not accelerator.is_local_main_process
        ):
            output = model(batch["input_ids"], labels=batch["labels"])
            loss = output.loss
            # decoder_prompts = [input_id[: end_idx + 1] for input_id, end_idx in zip(batch['input_ids'], batch['answer_start_position'])]
            # decoder_prompts = pad_tensors_to_left(decoder_prompts, pad_value=processor.tokenizer.pad_token_id)
            # predictions = model.generate(batch['pixel_values'], decoder_input_ids=decoder_prompts, max_length=128, early_stopping=True, pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,use_cache=True,num_beams=1,bad_words_ids=[[processor.tokenizer.unk_token_id]], synced_gpus=True)
            # predicted_answers = processor.tokenizer.batch_decode(predictions[:, (torch.max(batch['answer_start_position']).item()+1):], skip_special_tokens=True)
            # # TODO: we can just log pred and answers and compute the metric later
            # similarities = accelerator.pad_across_processes(torch.tensor([compute_levenstein_similarity(pred, answer) for pred, answer in zip(predicted_answers, batch['answer'])], device=model.device))

            gathered_val = accelerator.gather_for_metrics(
                {
                    "loss": loss.detach(),
                    # "mean_similarity": similarities,
                }
            )

            val_loss.update(gathered_val["loss"])
            # similarity.update(gathered_val['mean_similarity'])

            # for question, answer, prediction in zip(batch['question'], batch['answer'], predicted_answers):
            #     prediction_samples_list.append((question, answer, prediction))

            # # TODO: remove
            # counter += 1

            # if counter > 10:
            #     break

    # TODO: create a dedicated class to wrap this
    # columns = ['question', 'answer', 'prediction']
    # table =  wandb.Table(data=prediction_samples_list, columns=columns)

    log_val = {
        "val loss": val_loss.compute(),
        # 'mean_similarity': similarity.compute(),
        # 'examples': table
    }
    if config["wandb"]:
        accelerator.log({**log_val}, step=curr_step)

    return


def training_loop_accelerate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    val_dataset,
    run_name: str,
    no_log: bool,
):
    config = read_deepspeed_config()

    if no_log:
        config["wandb"] = None

    if config.get("wandb"):
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name=config["wandb"]["project"],
            config=config,
            init_kwargs={"wandb": {"entity": config["wandb"]["entity"]}},
        )
    else:
        accelerator = Accelerator()

    if config['enable_peft']:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    with accelerator.main_process_first():
        partial_collate_func = partial(collate_fn, tokenizer.pad_token_id)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config["train_micro_batch_size_per_gpu"],
            collate_fn=partial_collate_func,
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config["val_micro_batch_size_per_gpu"],
            collate_fn=partial_collate_func,
        )

    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )
    else:
        gradient_accumulation_steps = 1

    # decay to min_lr instead of 0
    accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")

    total_num_steps = (len(train_dataloader) / gradient_accumulation_steps) * config[
        "num_epochs"
    ]
    # instead of decaying to zero, decay to ratio of min_lr / lr
    total_num_steps += int(total_num_steps) + config["warmup_steps"]
    accelerator.print(f"Total training steps: {total_num_steps}")

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=config["warmup_steps"] * accelerator.num_processes,
            num_training_steps=total_num_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer,
            total_num_steps=config["warmup_steps"],
            warmup_num_steps=config["warmup_steps"],
        )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    accelerator.register_for_checkpointing(scheduler)

    for epoch in range(config["num_epochs"]):
        train_loss = MeanMetric(nan_strategy="error").to(model.device)

        for step, batch in tqdm(
            enumerate(train_dataloader), disable=not accelerator.is_local_main_process
        ):
            with accelerator.accumulate(model):
                model.train()
                output = model(batch["input_ids"], labels=batch["labels"])
                loss = output.loss

                # gather loss before backprop in case of gradient accumulation
                loss_values = accelerator.gather_for_metrics(
                    {"loss": loss.detach().float()}
                )
                train_loss.update(loss_values["loss"])

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_train = {"train_loss": train_loss.compute()}

                curr_step = step + epoch * len(train_dataloader)

                if step > 0 and step % 100 == 0:
                    if config["wandb"]:
                        accelerator.log(
                            {"lr": scheduler.get_last_lr()[0]}, step=curr_step
                        )
                        accelerator.log({**log_train}, step=curr_step)

                if (step + 1) % config["eval_every"] == 0:
                    model.eval()
                    evaluate_model(
                        accelerator,
                        model,
                        val_dataloader,
                        config,
                        curr_step,
                    )
                    model.train()

                train_loss.reset()

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            f"{config['output_dir']}/{run_name}/epoch_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )


def train(run_name: str, no_log: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    # BEWARE !!!
    tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
    tokenizer.pad_token = "<PAD>"

    train_dataset = SquadDataset(tokenizer, "train")
    val_dataset = SquadDataset(tokenizer, "train")

    training_loop_accelerate(
        model, tokenizer, train_dataset, val_dataset, run_name, no_log
    )


@click.command()
@click.option("--run_name", default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt"))
@click.option("--no_log", is_flag=True, default=False)
def main(run_name: str, no_log: bool):
    train(run_name, no_log)


if __name__ == "__main__":
    main()

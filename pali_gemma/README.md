# Command

python -m pali_gemma.hf_trainer

torchrun --nproc_per_node 4 pali_gemma/hf_trainer.py


# Training PaliGemma

- using 448 version: resolution is too small for the model to read the document. Learns correctly the answer format, but unable to actually parse the image
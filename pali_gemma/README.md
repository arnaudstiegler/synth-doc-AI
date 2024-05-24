# Command

python -m pali_gemma.hf_trainer

torchrun \
    --nproc_per_node 4 pali_gemma/hf_trainer.py \
    --bf16 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --overwrite_output_dir \
    --predict_with_generate

        --model_name_or_path google-t5/t5-small \
            --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
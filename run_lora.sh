#!/bin/bash
python3 finetune_lora.py \
	--dataset_path train_data_token \
	--lora_rank 8 \
	--per_device_train_batch_size 128 \
	--gradient_accumulation_steps 1 \
	--save_steps=50 \
	--max_steps=1 \
	--save_total_limit 2 \
	--learning_rate 1e-4 \
	--remove_unused_columns false \
	--fp16 \
	--logging_steps 1 \
	--output_dir model-lora

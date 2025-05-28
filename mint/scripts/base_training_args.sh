#!/bin/bash

ID=$RANDOM
PORT=$((12000 + RANDOM % 20000))
export header="torchrun --nproc_per_node 1 --nnodes 1 --master_port $PORT \
-m mint.train.train"

export base_training_args="--do_train True \
--max_seq_length 1024 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 100 \
--save_strategy no \
--num_train_epochs 4 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--optim adamw_torch \
--percentage 1.0 \
--save_strategy epoch \
--save_total_limit 1 \
--report_to tensorboard \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32"


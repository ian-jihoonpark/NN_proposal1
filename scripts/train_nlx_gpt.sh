#!/bin/bash

CKPT_DIR="checkpoint"

CUDA_VISIBLE_DEVICES=0 python Trainer.py \
--seed 32 \
--mode train \
--model_path "NLX_GPT" \
--project_name "Train1" \
--cached_dir cached \
--experiment_name experimet1 \
--max_epochs 30 \
--ngpu 1 \
--warmup_ratio 0.6 \
--checkpoints_dir ${CKPT_DIR} \
--weight_decay 0.0 \
--nle_anno_path "datasets/VQA-X/annotated/" \
--nle_image_dir "datasets/image" \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 1e-5 \
--gradient_accumulation_steps 1 \
--val_check_interval 0.5 \
--max_seq_len 40 \
--n_train_workers 8 \
--n_valid_workers 4 \
--img_size 224 \
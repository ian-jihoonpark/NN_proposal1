#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python Trainer.py \
--mode test \
--model_path "NLX_github" \
--output_dir outputs \
--cached_dir /media/storage/checkpoints/NLX_GPT/cached \
--load_ckpt_path nlx-gpt/nle_model_11 \
--test_batch_size 1 \
--ngpu 1 \
--img_size 224 \
--nle_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated" \
--nle_image_dir "/media/storage/datasets/image" \
--vqax_test_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated/vqax_test.json" \
--AEmode AE \
--top_k 0 \
--top_p 0.9 \
# #!/bin/bash
# CUDA_VISIBLE_DEVICES=1 python Trainer.py \
# --mode test \
# --model_path "NLX_GPT" \
# --output_dir outputs \
# --cached_dir /media/storage/checkpoints/NLX_GPT/cached \
# --load_ckpt_path /media/storage/checkpoints/NLX_GPT/NLX_GPT_AE/epoch=13-NLX_GPT_val_loss=0.727.ckpt
# --test_batch_size 1 \
# --ngpu 1 \
# --img_size 224 \
# --nle_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated" \
# --nle_image_dir "/media/storage/datasets/image" \
# --vqax_test_anno_path "/media/storage/datasets/NLE_annotation/VQA-X/annotated/vqaX_test.json" \
# --AEmode AE \
# --top_k 0 \
# --top_p 0.9 \
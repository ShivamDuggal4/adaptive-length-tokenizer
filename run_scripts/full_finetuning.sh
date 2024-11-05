## Additonal run tools:
## CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

## Single node run comamnd
torchrun --nproc_per_node=8 \
    --master_port=12345 main_full_finetuning.py \
    --batch_size 12 \
    --model alit_small \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.e-4 --weight_decay 0.05 \
    --base_tokenizer vqgan \
    --quantize_latent \
    --factorize_latent \
    --output_dir ./output_dir/full_finetuning/alit_small_vqgan_quantized_latents/ \
    --finetune ./output_dir/latent_distillation_pretrain/alit_small_vqgan_quantized_latents/checkpoint-last.pth \
    --data_path $TRAIN_DATA_DIR
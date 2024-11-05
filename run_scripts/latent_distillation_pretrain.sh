## Additional run tools:
## WANDB__SERVICE_WAIT=300 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_P2P_DISABLE=1 TORCH_NCCL_DESYNC_DEBUG=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1

## Single node run comamnd
torchrun --nproc_per_node=8 \
    --master_port=12345 main_pretrain.py \
    --batch_size 32 \
    --num_workers 10 \
    --model alit_base \
    --base_tokenizer vqgan \
    --quantize_latent \
    --factorize_latent \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --output_dir ./output_dir/latent_distillation_pretrain/alit_small_vqgan_quantized_latents/ \
    --data_path $TRAIN_DATA_DIR
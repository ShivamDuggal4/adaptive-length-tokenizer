python evaluate_rfid.py \
    --model alit_small \
    --base_tokenizer vqgan \
    --quantize_latent \
    --output_dir ./output_dir/full_finetuning/alit_small_vqgan_quantize_latents/ \
    --ckpt adaptive_tokenizers/pretrained_models/imagenet100/alit_small_vqgan_quantized_latents.pth \
    --data_path $TRAIN_DATA_DIR
from omegaconf import OmegaConf
from modules.base_tokenizers import VQGANWrapper, LDMVAEWrapper

# ALIT Latent-Distillation Encoder / Decoder configurations.
width      = {"small": 512, "base": 768, "semi_large": 1024}
num_layers = {"small": 8,   "base": 12,  "semi_large": 16}
num_heads  = {"small": 8,   "base": 12,  "semi_large": 16}

base_tokenizers = {
    "vqgan": VQGANWrapper,
    "vae": LDMVAEWrapper,
}

adaptive_tokenizers = {
    "vqgan": __import__('adaptive_tokenizers.adaptive_vqgan', fromlist=['AdaptiveLengthImageTokenizer']).AdaptiveLengthImageTokenizer,
    "vae": __import__('adaptive_tokenizers.adaptive_vae', fromlist=['AdaptiveLengthImageTokenizer']).AdaptiveLengthImageTokenizer,
}

adaptive_tokenizer_args = {
    "vqgan": OmegaConf.load('adaptive_tokenizers/configs/adaptive_vqgan.yaml'),
    "vae": OmegaConf.load('adaptive_tokenizers/configs/adaptive_vae.yaml')
}

def alit_tiny(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = adaptive_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["small"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["small"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **adaptive_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def alit_small(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = adaptive_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["small"], encoder_num_heads=num_heads["small"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["small"], decoder_num_heads=num_heads["small"],
        **adaptive_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def alit_base(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = adaptive_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["base"], encoder_num_layers=num_layers["base"], encoder_num_heads=num_heads["base"],
        decoder_width=width["base"], decoder_num_layers=num_layers["base"], decoder_num_heads=num_heads["base"],
        **adaptive_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer

def alit_semilarge(base_tokenizer_args, **kwargs):
    base_tokenizer = base_tokenizers[base_tokenizer_args["id"]](**base_tokenizer_args)
    tokenizer = adaptive_tokenizers[base_tokenizer_args["id"]](
        base_tokenizer=base_tokenizer,
        encoder_width=width["semi_large"], encoder_num_layers=num_layers["semi_large"], encoder_num_heads=num_heads["semi_large"],
        decoder_width=width["semi_large"], decoder_num_layers=num_layers["semi_large"], decoder_num_heads=num_heads["semi_large"],
        **adaptive_tokenizer_args[base_tokenizer_args["id"]], **kwargs
    )
    return tokenizer
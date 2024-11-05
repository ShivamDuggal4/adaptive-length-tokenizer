import torch
import torch.nn as nn
from omegaconf import OmegaConf
from base_tokenizers.taming.models.vqgan import VQModel
from base_tokenizers.vae_ldm import AutoencoderKL

class VQGANWrapper(nn.Module):
    def __init__(self, id="vqgan", embed_dim=256, pretrained_ckpt_path="base_tokenizers/pretrained_models/vqgan.ckpt", is_requires_grad=False):
        super().__init__()

        self.is_requires_grad = is_requires_grad
        self.embed_dim = embed_dim
        config = OmegaConf.load('base_tokenizers/configs/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                            n_embed=config.params.n_embed,
                            embed_dim=config.params.embed_dim,
                            ckpt_path=pretrained_ckpt_path)
        self.codebook_emb_dim = config.params.embed_dim
        self.codebook_size = config.params.n_embed

        # For stage 1, we do not finetune image encoder-decoder.
        for param in self.vqgan.parameters():
            param.requires_grad = self.is_requires_grad


    def forward_vqgan(self, imgs):
        if not self.is_requires_grad:
            with torch.no_grad():
                z, z_q, _, token_tuple = self.vqgan.encode(imgs)
        else:
            z, z_q, _, token_tuple = self.vqgan.encode(imgs)
        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        gt_indices = token_indices.clone().detach().long()
        return z, z_q, token_indices, gt_indices

    def get_img_tokens(self, imgs):
        vqgan_tokens, vqgan_tokens_quantized, token_indices, gt_indices = self.forward_vqgan(imgs)
        return vqgan_tokens, gt_indices
    

class LDMVAEWrapper(nn.Module):
    def __init__(self, id="vae", embed_dim=16, pretrained_ckpt_path="base_tokenizers/pretrained_models/vae.ckpt", is_requires_grad=False):
        super().__init__()

        self.is_requires_grad = is_requires_grad
        self.embed_dim = embed_dim
        self.vae = AutoencoderKL(
            embed_dim=embed_dim, 
            ch_mult=(1, 1, 2, 2, 4), ckpt_path=pretrained_ckpt_path).cuda()
        
        # For stage 1, we do not finetune image encoder-decoder.
        for param in self.vae.parameters():
            param.requires_grad = is_requires_grad
        
    def forward_vae(self, imgs):
        if not self.is_requires_grad:
            with torch.no_grad():
                posterior = self.vae.encode(imgs)
                x = posterior.sample().mul_(0.2325)
        else:
            posterior = self.vae.encode(imgs)
            x = posterior.sample().mul_(0.2325)
        return x

    def get_img_tokens(self, imgs):
        vae_tokens = self.forward_vae(imgs)
        return vae_tokens


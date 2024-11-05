import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.base import _expand_token
from modules.latent_distillation_modules import Encoder, Decoder

class AdaptiveLengthImageTokenizer(nn.Module):
    def __init__(self, 
            base_tokenizer,
            encoder_width, encoder_num_layers, encoder_num_heads,
            decoder_width, decoder_num_layers, decoder_num_heads,
            quantize_latent=True, factorize_latent=True, vq_codebook_size=4096, vq_token_dim=12, vq_commitment_cost=0.25, vq_use_l2_norm = True,
            num_init_latent_tokens=32, img_size=256, patch_size=16, max_rollout_iters=8,
            dynamic_halting=True, dynamic_halting_threshold=0.025, 
            train_stage="latent_distillation_pretrain"
        ):
        
        super().__init__()
        
        self.train_stage = train_stage
        self.quantize_latent = quantize_latent
        if quantize_latent is True: factorize_latent=True
        self.factorize_latent = factorize_latent
        self.dynamic_halting = dynamic_halting
        self.dynamic_halting_threshold = dynamic_halting_threshold
        self.max_rollout_iters = max_rollout_iters
        grid_size = img_size // patch_size
        scale = encoder_width ** -0.5

        self.encoder_ln_pre = nn.LayerNorm(encoder_width)
        self.encoder_ln_post = nn.LayerNorm(encoder_width)
        self.encoder_ln_recursive = nn.LayerNorm(encoder_width)
        self.pre_quantizer_mlp = nn.Linear(encoder_width, vq_token_dim, bias=True)
        self.encoder = Encoder(encoder_width, encoder_num_layers, encoder_num_heads)
        self.decoder = Decoder(decoder_width, decoder_num_layers, decoder_num_heads, factorize_latent=self.factorize_latent, factorized_latent_dim=vq_token_dim, output_dim=base_tokenizer.embed_dim)

        self.encoder_positional_embedding = nn.Parameter(scale * torch.randn(grid_size ** 2 + 1, encoder_width))
        self.encoder_class_embedding = nn.Parameter(scale * torch.randn(1, encoder_width))
        self.encoder_mask_token = nn.Parameter(scale * torch.randn(1, 1, encoder_width))

        self.decoder_positional_embedding = nn.Parameter(scale * torch.randn(grid_size ** 2 + 1, decoder_width))
        self.decoder_class_embedding = nn.Parameter(scale * torch.randn(1, decoder_width))
        self.decoder_mask_token  = nn.Parameter(scale * torch.randn(1, 1, decoder_width))
        
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_init_latent_tokens, encoder_width))
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(num_init_latent_tokens, encoder_width))
        self.timestep_embedding = nn.Parameter(scale * torch.randn(self.max_rollout_iters, num_init_latent_tokens, encoder_width))
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=encoder_width-base_tokenizer.embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True)

        self.apply(self._init_weights)
        
        if self.quantize_latent:
            from modules.vector_quantizer import VectorQuantizer
            # Intialization for Quantizer is done inside VectorQuantizer
            self.quantize = VectorQuantizer(
                codebook_size=vq_codebook_size,
                token_size=vq_token_dim,
                commitment_cost=vq_commitment_cost,
                use_l2_norm=vq_use_l2_norm)
        
        self.base_tokenizer = base_tokenizer

        # TODO: Different loss weights per iteration might not be very critical
        self.lambda_loss_weight = [2.5, 2.0, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0]
        
        if self.train_stage=="full_finetuning":
            # TODO: Ablate the requirement of different discriminators for different recurrent rollout iterations.
            # Intuition is at different rollout iteration .....
            from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
            self.gan_losses = nn.ModuleList([VQLPIPSWithDiscriminator(
                disc_conditional= False, disc_in_channels= 3, 
                disc_start= 0, disc_weight= 0.2, codebook_weight= 1.0, # perceptual_weight=0.0
            ) for _ in range(self.max_rollout_iters)])
        
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def preprocess_encoder(self, img_tokens):
        x = img_tokens
        x = torch.cat([_expand_token(self.encoder_class_embedding.to(x.get_device()), x.shape[0]), x], dim=1)
        x = x + self.encoder_positional_embedding
            
        latent_tokens = self.latent_tokens + self.latent_token_positional_embedding
        latent_tokens = latent_tokens[None].repeat(x.shape[0], 1, 1)
        return x, latent_tokens
    
    def preprocess_decoder(self, img_tokens):
        mask_tokens = self.decoder_mask_token.repeat(img_tokens.shape[0], img_tokens.shape[1], 1).to(img_tokens.dtype)
        mask_tokens = torch.cat([_expand_token(self.decoder_class_embedding, mask_tokens.shape[0]).to(mask_tokens.get_device()), mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.decoder_positional_embedding.to(mask_tokens.dtype)
        return mask_tokens
    
    def perform_dynamic_halting(self, x, decoded_code, gt_code, num_img_tokens):
        # To save compute, we skip performing dynamic halting at the last iteration, since its a waste.
        if iter!=self.max_rollout_iters-1: 
            drop_mask = torch.zeros_like(decoded_code[:,:,0])
            drop_mask[torch.abs(decoded_code-gt_code).mean(dim=-1)<self.dynamic_halting_threshold][:] = 1.

            img_tokens_minus_class = x[:,1:num_img_tokens]
            encoder_mask_token = self.encoder_positional_embedding[None,1:].repeat(x.shape[0],1,1) + self.encoder_mask_token.repeat(x.shape[0], img_tokens_minus_class.shape[1], 1)
            img_tokens_minus_class = encoder_mask_token * drop_mask[...,None] + (1-drop_mask[...,None]) * img_tokens_minus_class
            x = torch.cat((x[:,:1], img_tokens_minus_class, x[:,num_img_tokens:]), dim=1)
        return x
        
    def reconstruct_images(self, decoded_code):
        decoded_code = decoded_code.reshape(decoded_code.shape[0], 16, 16, decoded_code.shape[-1]).permute([0,3,1,2])
        return self.base_tokenizer.vae.decode(decoded_code)

    def forward(self, imgs, sample_grad_iters=-1, reconstruction_iters=[], gan_optimizer_idx=None, gan_loss_weight=None):
        # sample_grad_iters==-1: evaluate loss at all roll out iterations (default setting). 
        # Otherwise, we randomly evaluate loss at sample_grad_iters number of iterations.
        # reconstruction_iters==[] – reconstruct back images at different iterations (default setting).
        # reconstruction_iters=="grad" – reconstruct back images at all gradient iters.
        # reconstruction_iters=="all" – reconstruct back images at all iters

        # Generating image tokens, vqgan codebook index (gt_indices).
        # Initializing masked_2D_tokens and init_latent_tokens
        vae_tokens = self.base_tokenizer.get_img_tokens(imgs)
        img_tokens = self.patch_embed(imgs)
        img_tokens = torch.cat((vae_tokens, img_tokens), dim=1)
        vae_tokens = vae_tokens.reshape(vae_tokens.shape[0], vae_tokens.shape[1], -1).permute([0,2,1])
        img_tokens = img_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1])
        img_tokens = F.normalize(img_tokens, dim=-1)
        gt_code = vae_tokens

        masked_2d_tokens = self.preprocess_decoder(img_tokens)
        img_tokens, init_latent_tokens = self.preprocess_encoder(img_tokens) 
        num_img_tokens = img_tokens.shape[1]
        x = torch.cat([img_tokens, init_latent_tokens + self.timestep_embedding[0]], dim=1)
        x = self.encoder_ln_pre(x)
        
        # Sampling rollout iterations at which gradient should be computed.
        if isinstance(sample_grad_iters, list):
            # In full_finetuning stage we compute loss at only one iteration which comes from engines/full_finetuning.py
            grad_iters = sample_grad_iters
        else:
            grad_iters = np.arange(self.max_rollout_iters)
            if self.training and sample_grad_iters!=-1:
                np.random.shuffle(grad_iters)
                grad_iters = grad_iters[:sample_grad_iters]
            grad_iters = grad_iters.tolist()
        if reconstruction_iters=="grad": reconstruction_iters=grad_iters
        elif reconstruction_iters=="all": reconstruction_iters = np.arange(self.max_rollout_iters).tolist()

        all_logs = []
        total_loss = 0
        for iter in range(self.max_rollout_iters):
            # image_tokens, initialized_latent_tokens -> processed image_tokens, learned latent_tokens
            x = self.encoder(x)
            latent_tokens = x[:, img_tokens.shape[1]:]
            
            # Latent quantization and decoding is only required either for image reconstruction at test time or for computing reconstruction loss at train time.  
            # To save compute at train time, one could randomly sample different iterations at which gradient should be computed.
            if not self.training or iter in grad_iters or iter in reconstruction_iters:
                latent_tokens = self.encoder_ln_post(latent_tokens)
                latent_tokens_factorized = self.pre_quantizer_mlp(latent_tokens)
                
                if self.factorize_latent: latent_tokens_factorized = self.pre_quantizer_mlp(latent_tokens)
                else: latent_tokens_factorized = latent_tokens # No factorization performed.
                
                if self.quantize_latent:
                    latent_tokens_quantized, quant_result_dict = self.quantize(latent_tokens_factorized, is_quantize=True)
                    decoded_code = self.decoder(latent_tokens_quantized, masked_2d_tokens)
                else:
                    decoded_code = self.decoder(latent_tokens_factorized, masked_2d_tokens)
                
                if self.dynamic_halting: 
                    x = self.perform_dynamic_halting(x, decoded_code, gt_code, num_img_tokens)
                
                iter_logs_dict = {}
                if self.training and iter in grad_iters:
                    if self.train_stage == "latent_distillation_pretrain":
                        iter_code_loss = self.forward_loss(gt_code, decoded_code)
                        total_loss = total_loss + self.lambda_loss_weight[iter] * iter_code_loss
                        iter_logs_dict.update({
                            "code_loss_{}".format(iter): iter_code_loss.item()
                        })
                    elif self.train_stage == "full_finetuning":
                        reconstructed_imgs = self.reconstruct_images(decoded_code)
                        total_loss, iter_logs_dict = self.forward_gan_losses(imgs, reconstructed_imgs, gan_optimizer_idx, iter_idx=iter, discriminator_loss_weight=gan_loss_weight)
                        iter_logs_dict.update({
                            "reconstructed_imgs_{}".format(iter): reconstructed_imgs,
                        })

                    if self.quantize_latent: 
                        total_loss = total_loss + 1. * quant_result_dict['quantizer_loss']
                        iter_logs_dict.update({
                            "quantization_loss_{}".format(iter): quant_result_dict['quantizer_loss'].item(),
                        })

                if iter in reconstruction_iters and "reconstructed_imgs_{}".format(iter) not in iter_logs_dict:
                    reconstructed_imgs = self.reconstruct_images(decoded_code)
                    iter_logs_dict.update({
                        "reconstructed_imgs_{}".format(iter): reconstructed_imgs,
                    })  

                all_logs.append(iter_logs_dict)

            # TODO Ablation -- timestep_embedding is not a critical component, can be avoided. Would enable infinite rollout at test time.
            if iter!=self.max_rollout_iters-1:
                x = torch.cat((x, init_latent_tokens + self.timestep_embedding[iter+1]), dim=1)
                x = self.encoder_ln_recursive(x)
        
        if not self.training: return all_logs
        return total_loss, all_logs
    
    def forward_loss(self, gt_code, decoded_code):
        code_loss = (gt_code - decoded_code)**2
        return code_loss.mean()

    def get_last_layer(self):
        return self.base_tokenizer.vae.decoder.conv_out.weight

    def forward_gan_losses(self, imgs, reconstructed_imgs, optimizer_idx, iter_idx, discriminator_loss_weight):
        assert(optimizer_idx is not None)
        if discriminator_loss_weight==0:
            global_step=-torch.inf
            self.gan_losses[iter_idx].discriminator_weight = 0.2
        else:
            global_step=torch.inf
            self.gan_losses[iter_idx].discriminator_weight = discriminator_loss_weight
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")

            iter_log_dict_ae = {}
            for key in log_dict_ae.keys():
                iter_log_dict_ae["{}_{}".format(key, iter_idx)] = log_dict_ae[key]
            return aeloss, iter_log_dict_ae

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")
            
            iter_log_dict_disc = {}
            for key in log_dict_disc.keys():
                iter_log_dict_disc["{}_{}".format(key, iter_idx)] = log_dict_disc[key]
            
            return discloss, iter_log_dict_disc




    
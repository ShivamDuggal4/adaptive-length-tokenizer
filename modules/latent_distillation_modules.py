import torch
import torch.nn as nn
from modules.base import ResidualAttentionBlock

class Encoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio)
            )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)
        return x  
    

class Decoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, factorize_latent, factorized_latent_dim, output_dim, max_latent_tokens=256, mlp_ratio=4.0, vis_attn_weights=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.factorize_latent = factorize_latent
        if factorize_latent: self.decoder_embed = nn.Linear(factorized_latent_dim, width, bias=True)
        scale = width ** -0.5

        self.vis_attn_weights = vis_attn_weights
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio, vis_attn_weights=vis_attn_weights))
        self.ln_post = nn.LayerNorm(width)

        self.ffn = nn.Sequential(
            nn.Linear(width, 2*width, bias=True), nn.Tanh(),
            nn.Linear(2*width, output_dim)
        )

        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(max_latent_tokens, width))


    def forward(self, latent_1D_tokens, masked_2D_tokens):
        if self.factorize_latent: latent_1D_tokens = self.decoder_embed(latent_1D_tokens)
        latent_1D_tokens = latent_1D_tokens + self.latent_token_positional_embedding[:latent_1D_tokens.shape[1]]
        x = torch.cat([masked_2D_tokens, latent_1D_tokens], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        all_attn_weights = []
        for i in range(self.num_layers):
            if self.vis_attn_weights:
                x, x_weights = self.transformer[i](x)
                all_attn_weights.append(x_weights.cpu().detach().numpy())
            else: x = self.transformer[i](x)
        x = x.permute(1, 0, 2)

        reconstructed_2D_tokens = x[:, 1:masked_2D_tokens.shape[1]]
        reconstructed_2D_tokens = self.ln_post(reconstructed_2D_tokens)
        reconstructed_2D_tokens = self.ffn(reconstructed_2D_tokens)

        if self.vis_attn_weights: return reconstructed_2D_tokens, all_attn_weights
        return reconstructed_2D_tokens
import torch
import torch.nn as nn
from collections import OrderedDict

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, vis_attn_weights=False, mlp_ratio = 4.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        self.vis_attn_weights = vis_attn_weights
        
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(self, x: torch.Tensor):
        if not self.vis_attn_weights:
            return self.attn(x, x, x, need_weights=False)[0]
        return self.attn(x, x, x, need_weights=True)

    def forward(self, x: torch.Tensor):
        if self.vis_attn_weights: 
            attn_output, attn_weights = self.attention(self.ln_1(x))
        else: attn_output = self.attention(self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0: 
            x = x + self.mlp(self.ln_2(x))
        
        if self.vis_attn_weights: return x, attn_weights
        return x




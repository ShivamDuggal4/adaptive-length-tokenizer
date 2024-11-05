import torch
import torch.nn as nn
from collections import OrderedDict

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio = 4.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            # self.ln_2_latent = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            # self.mlp_latent = nn.Sequential(OrderedDict([
            #     ("c_fc", nn.Linear(d_model, mlp_width)),
            #     ("gelu", act_layer()),
            #     ("c_proj", nn.Linear(mlp_width, d_model))
            # ]))

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        attn_output = self.attention(self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0: 
            # print(x.shape, "x shape check")
            # x_latent = x[257:]
            # x_img = x[:257]
            # assert(x_latent.shape[0]%32==0)
            # assert(x_latent.shape[0]!=0)
            # x = x + torch.cat((self.mlp(self.ln_2(x_img)), self.mlp_latent(self.ln_2_latent(x_latent))), dim=0)
            x = x + self.mlp(self.ln_2(x))
        return x




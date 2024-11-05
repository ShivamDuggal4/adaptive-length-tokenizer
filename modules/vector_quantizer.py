import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc as misc
from torch.cuda.amp import autocast

class VectorQuantizer(nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 ema_update: bool = False
                 ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm
        self.codebook_size = codebook_size

        self.ema_update = ema_update
        if self.ema_update:
            print('ema mode is on...')
            self.decay = 0.99
            self.eps = 1e-5
            self.cluster_size = torch.nn.Parameter(torch.zeros(codebook_size), requires_grad = False)
            self.embed_avg = torch.nn.Parameter(self.embedding.weight.clone(), requires_grad = False)
            self.embedding.weight.requires_grad = False

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)
    
    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized) 

    @autocast(enabled=False)
    def forward(self, z, is_quantize = True):
        z = z.float().contiguous()
        z_flattened = z.reshape(z.shape[0] * z.shape[1], z.shape[2])

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.training and self.ema_update:
            z_onehot = F.one_hot(min_encoding_indices, self.codebook_size).type(z.dtype)
            z_onehot_sum = z_onehot.sum(0)            
            z_sum = z_onehot.transpose(0,1) @ z_flattened
            misc.all_reduce_mean(z_onehot_sum.contiguous())
            misc.all_reduce_mean(z_sum.contiguous())
            
            self.cluster_size_ema_update(z_onehot_sum)
            self.embed_avg_ema_update(z_sum)
            self.weight_update(self.codebook_size)

        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
            z = torch.nn.functional.normalize(z, dim=-1)

        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        loss = commitment_loss 
        if not self.ema_update:
            loss = loss + codebook_loss

        z_quantized = z + (z_quantized - z).detach()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z.shape[0], z.shape[1])
        )
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return z_quantized
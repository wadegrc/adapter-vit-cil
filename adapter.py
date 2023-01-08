import torch.nn as nn
import torch
import copy
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, embedding_key='mean'):
        super(Adapter, self).__init__()
        self.embed_dim = c_in
        self.embedding_key = embedding_key
        self.adapters = nn.ModuleList()
        for i in range(10):
            self.adapters.append(nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        ))
        self.adapter_key = nn.Parameter(torch.zeros((10, c_in)))
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, adapter_mask=None,cls_features=None):
        out = dict()
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'mean_cls':
            x_embed_mean = 2*x_embed[:, 0] + torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        adapter_key_norm = self.l2_normalize(self.adapter_key, dim=1)#size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)#B,C

        similarity = torch.matmul(x_embed_norm, adapter_key_norm.t())#B,size
        id_counts = None
        if adapter_mask is None:
            _, idx = torch.topk(similarity, k=1, dim=1)
            adapter_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            _, major_idx = torch.topk(id_counts, k=1)
            major_adapter_id = adapter_id[major_idx]
            idx = major_adapter_id.expand(x_embed.shape[0], -1)
        else:
            idx = adapter_mask
        bias = copy.deepcopy(x_embed)
        bias.to(x_embed.device)
        for i in range(x_embed.shape[0]):
            for j in idx[i]:
                self.adapters[j].to(x_embed.device)
                
                bias[i] = bias[i]+self.adapters[j](x_embed[i])
            bias[i] = bias[i] / len(idx[i])

        out['adapter_idx']=idx

        adapter_norm = adapter_key_norm[idx]
        x_embed_norm = x_embed_norm.unsqueeze(1)
        adapter_key_norm = adapter_key_norm.expand(x_embed.shape[0], 10, -1)
        sim = adapter_norm*x_embed_norm #B,topk,C
        key_sim = adapter_norm*adapter_key_norm[:,:idx[0][0]]
        unsim = torch.sum(key_sim) / x_embed.shape[0]
        reduce_sim = torch.sum(sim) / x_embed.shape[0]
        out['reduce_sim'] = reduce_sim
        out['unsim'] = 0

        out['bias']=bias
        return out


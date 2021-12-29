import torch
import torch.nn as nn


class DvD(nn.Module):
    def forward(self, embeddings: torch.Tensor):
        # embeddings.shape = (population_size, sample_size * action_dim)
        mod = torch.norm(embeddings, p=2, dim=-1).detach()
        # embedding normalization
        embeddings = (embeddings.t() / mod).t()
        population_size = embeddings.shape[0]
        left = embeddings.unsqueeze(0).expand(population_size, *embeddings.shape)
        right = embeddings.unsqueeze(1).expand(population_size, *embeddings.shape)
        '''
        dot product kernel according to Parker-Holder
        x1 x2 x3     x1 x1 x1   x1.*x1
        x1 x2 x3 .*  x2 x2 x2 = x1.*x2 x2.*x2 
        x1 x2 x3     x3 x3 x3   x1.*x3 x2.*x3 x3.*x3
        '''
        similarity = 0.5 * ((left * right).sum(-1) + 1)
        det_k = torch.det(similarity)
        return det_k

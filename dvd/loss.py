import torch
import torch.nn as nn


class LogDet(nn.Module):
    def __init__(self, beta=0.99):
        super(LogDet, self).__init__()
        self.beta = beta

    def l2rbf(self, m_actions):
        actions = torch.stack(m_actions)
        x1 = actions.unsqueeze(0).repeat_interleave(actions.shape[0], 0)
        x2 = actions.unsqueeze(1).repeat_interleave(actions.shape[0], 1)
        d2 = torch.square(x1 - x2)
        l2 = torch.var(actions, dim=0).detach() + 1e-8
        return (d2 / (2 * l2)).mean(-1)

    def forward(self, embeddings):
        d = self.l2rbf(embeddings)
        K = (-d).exp()
        K_ = self.beta * K + (1 - self.beta) * torch.eye(len(embeddings), device=K.device)
        L = torch.linalg.cholesky(K_)
        log_det = 2 * torch.log(torch.diag(L)).sum()
        return log_det

    # def forward(self, embeddings: torch.Tensor):
    #     # embeddings.shape = (population_size, sample_size * action_dim)
    #     mod = torch.norm(embeddings, p=2, dim=-1).detach()
    #     # embedding normalization
    #     embeddings = (embeddings.t() / mod).t()
    #     population_size = embeddings.shape[0]
    #     left = embeddings.unsqueeze(0).expand(population_size, *embeddings.shape)
    #     right = embeddings.unsqueeze(1).expand(population_size, *embeddings.shape)
    #     '''
    #     dot product kernel according to Parker-Holder
    #     x1 x2 x3     x1 x1 x1   x1.*x1
    #     x1 x2 x3 .*  x2 x2 x2 = x1.*x2 x2.*x2
    #     x1 x2 x3     x3 x3 x3   x1.*x3 x2.*x3 x3.*x3
    #     '''
    #     k = 0.5 * ((left * right).sum(-1) + 1)
    #     k = self.beta * k + (1 - self.beta) * torch.eye(population_size, device=k.device)
    #     L = torch.linalg.cholesky(K_)
    #     logdet = 2 * torch.log(torch.diag(L)).sum()
    #     return logdet

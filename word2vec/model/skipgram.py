import numpy as np
import torch
import torch.nn as nn

from torch import LongTensor
from torch import FloatTensor


class SkipGram(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=100, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.wi = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=padding_idx)
        self.wo = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=padding_idx)
        r = -0.5 / self.embed_dim
        self.wi.weight.data.uniform_(-r, r)
        self.wo.weight.data.uniform_(-r, r)
        self.wi.weight.requires_grad = True
        self.wo.weight.requires_grad = True
    
    def forward(self, center, contexts, neg_samples):

        neg_samples = neg_samples.long()
        wi = self.wi(center).unsqueeze(2)
        wo = self.wo(contexts)
        nvectors = self.wo(neg_samples).neg()
        oloss = torch.bmm(wo, wi).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, wi).squeeze().sigmoid().log().mean(1)
        return -(oloss + nloss).mean()

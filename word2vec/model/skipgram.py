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
        self.ivectors = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=padding_idx)
        r = -0.5 / self.embed_dim
        self.ivectors.weight.data.uniform_(-r, r)
        self.ovectors.weight.data.uniform_(-r, r)
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
    
    def forward(self, center, contexts, neg_samples):

        neg_samples = neg_samples.long()
        ivectors = self.ivectors(center).unsqueeze(2)
        ovectors = self.ovectors(contexts)
        nvectors = self.ovectors(neg_samples).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().mean(1)
        return -(oloss + nloss).mean()

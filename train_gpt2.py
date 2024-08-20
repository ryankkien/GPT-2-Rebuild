from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
@dataclass
class GPTConfig: #params
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #weights of token embedding table
            wpe = nn.Embedding(config.block_size, config,n_embd), #weights of position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #n_layer = number of layers
            ln_f = nn.LayerNorm(config.n_embd) #final layer
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #768 x 50257






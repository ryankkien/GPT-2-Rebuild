from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        #query key value
        #queries and keys multiply to get attention scores
        #attention scores are scaled by 1/sqrt(k.size(-1))
        #masking is applied to attention scores
        #softmax is applied to attention scores
        #values are multiplied by attention scores
        #output is re-assembled from head outputs
        #output is projected to n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) nh becomes a batch dimension so that future operations can be broadcasted over it
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) for parallel self attention
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1))) #attention scores
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))#masking only attends to future tokens
        att = F.softmax(att, dim=-1)
        y = att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all head outputs side by side concatenating heads
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU() #gaussian error linear unit (similar to relu but smoother) #gelu approximation used in original
        #picked over relu because removes the dead zone of 0, causing better adaptation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) #happens to every token individually
        return x #clean residual pathway is optimal, which doesnt happen due to norms

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






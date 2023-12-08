import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch import nn, einsum, broadcast_tensors
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from einops import rearrange, repeat
from einops import rearrange, repeat, pack, unpack


import os
import copy
import glob
import shutil
from math import pi, log
import math
import time
from dataclasses import dataclass
from typing import Optional, Union
import logging
from tqdm import tqdm

from transformers import GPT2Tokenizer

from einops import rearrange, repeat
from einops import rearrange, repeat, pack, unpack

import os
import copy
import glob
import shutil
from math import pi, log
import math
import time
from dataclasses import dataclass
from typing import Optional, Union
import logging
from tqdm import tqdm

from gpt2 import Conv1D,_get_clones,FeedForward

#helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def rotate_half(x):
    '''The initial step of our roformer includes use of In order to generalize our results in 2D to any xi ∈ R
  d where d is even, we divide the d-dimension space into d/2
  sub-spaces and combine them in the merit of the linearity of the inner product, turning f{q,k} into
  The above was excerpt from paper which involves splitting into d/2'''
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    '''a function for applying the rotatory embeddings frst getting the rotation dimension and sequence length
    getting the end index by adding the start index and rotation dimension as mentioned above,
    the t left, t and t right with the before token segment, during token segment and after token segment
    Applies the rotational embedding to the central portion of t.
    The rotation involves a combination of cosine and sine operations using the specified frequencies and scaling factor.  '''
    rot_dim, seq_len = freqs.shape[-1], t.shape[seq_dim]
    freqs = freqs[-seq_len:].to(t)
    end_index = start_index + rot_dim
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    '''Learning rotations by frequency handling by scaling out the rotations  this rearrangement helps in combining the rottations into one
    now repeating the rotations by replicating the rotations and then applying the rotatory embeddings'''
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

#rotatory embeddings
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
    ):
        '''This is a constructor class for our rotatory embeddings
        theta: the angle for rotation
        max_freq:the max frequency for rotation
        num_freq:the number of times frequencies need to be iterated
        interpolate factor: A factor used to control the value of positional embedding if it is low or high
        theta_rescale_factor:As the theta decays with the learning so we need to rescale it for decaying
        '''
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))


        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(freqs)


        # default sequence dimension

        self.default_seq_dim = -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos
        self.register_buffer('scale', None)


        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        '''
        The function to get the seq positonal embedding using torch.arange which uses [end-start]/start dividing by interpolation factor to control its value
         '''
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, freq_seq_len = None):
        '''A function to operate the rotation over queries and keys'''
        seq_dim = default(seq_dim, self.default_seq_dim)


        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]#getting device, data type and sequence length

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.forward(lambda: self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset), cache_key = f'freqs:{seq_len}|offset:{offset}')

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim = seq_dim)#applying the final operations over t value

    def forward(self, t, cache_key = None):
        '''The forward function for porpagting our t value'''
        should_cache = exists(cache_key)

        if should_cache and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)# converting the frequency into its transpose
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache:
            self.cache[cache_key] = freqs

        return freqs

class Attention_rope(nn.Module):
    def __init__(self, d_model=768, n_head=12, n_ctx=1024, d_head=64, bias=True, scale=False):
        '''Constructor funtion
        Params:
        d_model:The dimension that needs to be feed into our model
        n_head:The number of heads for attention
        n_ctx:a parameters for buffer registry for bias
        d_head:the dimension head output
        bias:A bool for including or not including bias
        scale: Whether to scale the attention scores by the square root of the dimension of the queries(use sqrt(dk) or not) "
        '''
        super().__init__()
        self.n_head  = n_head
        self.d_model = d_model
        self.c_attn  = Conv1D(d_model, d_model*3)
        self.scale   = scale
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.dropout = nn.Dropout(0.1)
        self.c_proj  = Conv1D(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(dim = 32)#intializing the rotatory embedding with dimension 32

    def split_heads(self, x):
        """
        spliting inyo given number of heads and then returning
        return shape [`batch`, `head`, `sequence`, `features`]
        """
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1)//self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        """The main attention function
        That claculates the attention using our dot product formula"""
        scores  = torch.matmul(q, k.transpose(-2, -1))# dot multiplication between q and k transpose
        if self.scale: scores = scores/math.sqrt(v.size(-1))# scaling it by dividing by sqrt(dk)
        nd, ns  = scores.size(-2), scores.size(-1)
        if attn_mask is not None: scores = scores + attn_mask# adding scores with attention mask values
        scores  = self.softmax(scores)# adding softmax values
        scores  = self.dropout(scores) #dropout of 0.1 as mentioned
        outputs = torch.matmul(scores, v) # now the final matrix multiplication between score and V
        return outputs

    def merge_heads(self, x):
        # merging the attention heads into one
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x):
        '''The feed forward function that calculates the attention, split the heads, make attention, merge heads and project out the output '''
        x        = self.c_attn(x) #new `x` shape - `[1,3,2304]`
        q, k, v  = x.split(self.d_model, dim=2)
        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        #applying the rotatory embeddings over query and key
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        out      = self._attn(q, k, v)
        out      = self.merge_heads(out)
        out      = self.c_proj(out)
        return out

class TransformerBlock_rope(nn.Module):
    def __init__(self, d_model=768, n_head=12, dropout=0.1):
        super().__init__()
        self.attn        = Attention_rope(d_model=768, n_head=12, d_head=64, n_ctx=1024, bias=True, scale=False)
        self.feedforward = FeedForward(dropout=0.1, d_model=768, nx=768*4)
        self.ln_1        = LayerNorm(d_model)
        self.ln_2        = LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x

class GPT2_rope(nn.Module):
    def __init__(self, nlayers=12, n_ctx=1024, d_model=768, vcb_sz=50257):
        super(GPT2_rope, self).__init__()
        self.nlayers = nlayers
        block        = TransformerBlock_rope(d_model=768, n_head=12, dropout=0.1)
        self.h       = _get_clones(block, 12)
        self.wte     = nn.Embedding(vcb_sz, d_model)
        self.wpe     = nn.Embedding(n_ctx, d_model)
        self.drop    = nn.Dropout(0.1)
        self.ln_f    = LayerNorm(d_model)
        self.out     = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        '''Initialization of weights'''
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''If the Linear, Embedding and Conv1D then nomrally initializing with mean and S.D'''
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                '''Data Bias zero'''
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, src, labels=None, pos_ids=None):
        '''Adding the positional embedding, dropping, then adding inputs, logits and outputs which are being used for loss function and then adding outputs and loss'''
        if pos_ids is None: pos_ids = torch.arange(0, src.size(-1)).unsqueeze(0)
        inp = self.drop((self.wte(src)+self.wpe(pos_ids)))
        for i in range(self.nlayers): inp = self.h[i](inp)
        inp     = self.ln_f(inp)
        logits  = self.out(inp)
        outputs = (logits,) + (inp,)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            return outputs
        return logits

#an implementation of GPT2 with rotatory and sliding window attention
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

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    maybe_sparsify,
    register_attention,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern,
)


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
from gpt2_rope2 import RotaryEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#helper functions
def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    '''Function for padding over tensor for how many times which is multiple here
    if the seqlen is not a muliple of multiple so we need to pad the remaining for which we caluclate the remainder and then pad the remainder
    Params:
    tensor:The tensor that needs to be pad
    multiple:The multiple upto which padding is happenining
    dim: the dimension accross for padding
    value:what should the padded values'''
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen#calculating the remaninder after the multiple padding has happened
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    '''This is a function for padding our x with the sliding window attention mechanism first getting the shapes and dimensions
    then padding the x from backward to forward which is of range of window 2n+1
    now we iteratively pad with different combination of windows by loop over the forward+backward+1
    and finally concatenating all the different tensors that are formed resulting in our final attention'''
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

#in attention block
class Attention_rope_slide(nn.Module):
    def __init__(self, d_model=768, n_head=12,window_size=3, n_ctx=1024, d_head=64, bias=True, scale=False,look_forward=1,look_backward=1):
        '''An implementation of a sliding window attention, as proposed in Longformer I am also combing the rotationaol embeddings with it for
        checking out the results
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
        self.proj_out = nn.Linear(n_head * d_head, d_model)
        self.scale   = scale
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.dropout = nn.Dropout(0.1)
        self.c_proj  = Conv1D(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(dim = 32)#intializing the rotatory embedding with dimension 32
        self.window_size = window_size
        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

        self.attention_mask: Optional[torch.Tensor] = None#attention mask to store the values of the slided attention window
        self.requires_same_k_q_dimensions = True

        self.look_backward=look_backward
        self.look_forward=look_forward

        self.causal=False
        self.force_sparsity=False
        self.shared_qk=False

        self.attn_mask=None
        self.TOKEN_SELF_ATTN_VALUE = -5e4

    def _get_local_mask(self, shape: torch.Size) -> AttentionMask:
      self.window_size = min(self.window_size * 2 + 1, shape[1]) if self.causal else min(self.window_size, shape[1])
      mask = local_1d_pattern(shape[1], window_size)

      if self.causal:
          mask &= causal_1d_pattern(shape[1])

      mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)

      # Convert mask to tensor and set its dtype to float32
      mask_tensor = mask.to(torch.float32)

      return AttentionMask(mask_tensor)

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


    def forward(self, x,mask = None,input_mask = None,attn_bias = None,window_size = None):
        '''The feed forward function that calculates the attention, split the heads, make attention, merge heads and project out the output
        Applies convolutional attention to the input tensor.
Splits the query, key, and value tensors into heads.
Applies rotary embeddings to the query and key.
Dynamically sets the window size if provided.
Asserts that the sequence length is divisible by the window size.
Applies the sliding window attention mechanism.
Computes attention, applies masks, and performs aggregation.
Returns the final output tensor.'''
        mask = default(mask, input_mask)

        x        = self.c_attn(x) #new `x` shape - `[1,3,2304]`
        q, k, v  = x.split(self.d_model, dim=2)

        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        #applying the rotatory embeddings over query and key
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        shape, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        if window_size is not None:
          self.window_size = window_size  # Set the window size dynamically

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size


        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, self.TOKEN_SELF_ATTN_VALUE)
            del self_mask


        sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]



            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        # out = self.proj_out(out)
        out, *_ = unpack(out, packed_shape, '* n d')
        out=rearrange(out, 'b n s d -> b s (n d)')
        return out

class TransformerBlock_rope_slide(nn.Module):
    def __init__(self, d_model=768, n_head=12, dropout=0.1,window_size=2):
        self.window_size=window_size
        super().__init__()
        self.attn        = Attention_rope_slide(d_model=768,window_size=window_size, n_head=12, d_head=64, n_ctx=1024, bias=True, scale=False)
        self.feedforward = FeedForward(dropout=0.1, d_model=768, nx=768*4)
        self.ln_1        = LayerNorm(d_model)
        self.ln_2        = LayerNorm(d_model)
        self.window_size=window_size

    def forward(self, x):
        x = x + self.attn(self.ln_1(x),window_size=self.window_size)
        x = x + self.feedforward(self.ln_2(x))
        return x
    
class GPT2_rope_slide(nn.Module):
    def __init__(self, nlayers=12, n_ctx=1024, d_model=768, vcb_sz=50257):
        super(GPT2_rope_slide, self).__init__()
        self.nlayers = nlayers
        block        = TransformerBlock_rope_slide(window_size=window_size,d_model=768, n_head=12, dropout=0.1)
        self.h       = _get_clones(block, 12)
        self.wte     = nn.Embedding(vcb_sz, d_model)
        self.wpe     = nn.Embedding(n_ctx, d_model)
        self.drop    = nn.Dropout(0.1)
        self.ln_f    = LayerNorm(d_model)
        self.out     = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def set_window_size(self, window_size):
        self.window_size = window_size

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



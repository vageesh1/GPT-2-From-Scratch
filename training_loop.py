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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


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

from gpt2_rope_slide_multiquery import GPT2_rope_slide_group
from gpt2 import Conv1D,_get_clones,FeedForward
from gpt2_rope2 import RotaryEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from tqdm import tqdm_notebook, trange
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

class TrainerConfig:
    '''Class for seeting the Training for training configurations'''
    # optimization parameters
    max_epochs = 10
    batch_size = 8
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config#getting the model and training configurations
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    loss = model(x, y)  # The forward method returns the mean of the loss directly
                    logits = model(x)
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    self.optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # repeat progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info(f"test loss: {test_loss}")
                return test_loss

        best_loss = float("inf")
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            # supports early stopping based on the test loss, or just save always is no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss if self.test_dataset is not None else float("inf")
                self.save_checkpoint()

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size:d} characters, {vocab_size:d} unique.")

        self.stoi = { ch: i for i, ch in enumerate(chars) }
        self.itos = { i: ch for i, ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx+self.block_size+1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

trainable_model = GPT2_rope_slide_group()#initialsing our model for training
trainable_model=trainable_model.to(device)

#inferencing the trained model 
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("inf")
    return out

def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b, t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = 25
    model.eval()
    for k in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilites
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

#DDP training loop 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Trainer_DDP:
    def __init__(self, model, train_dataset, test_dataset, config,rank,world_size):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.rank=rank
        self.world_size=world_size

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config#getting the model and training configurations
        setup(self.rank, self.world_size)
        model = model().to(self.rank)#moving the model to device
        model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)#wrapping the model with DDP
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        def run_epoch(self,split):
            print("running")
            is_train = split == "train"
            model.train(is_train)
            train_sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=self.rank, shuffle=False, drop_last=False)#making a distributed sampler for data parallel
            data = self.train_dataset if is_train else self.test_dataset

            self.loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                sampler=train_sampler # adding this as sampler fr distributed data
            )

            losses = []
            pbar = (
                tqdm(enumerate(self.loader), total=len(self.loader))
                if is_train
                else enumerate(self.loader)
            )
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    loss = model(x, y)  # The forward method returns the mean of the loss directly
                    logits = model(x)
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    self.optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # repeat progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, lr {lr:e}"
                    )
                    cleanup()#cleaning up all the the done files on all GPUs

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info(f"test loss: {test_loss}")
                return test_loss

        best_loss = float("inf")
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            # if we are using DistributedSampler, we have to tell it which epoch this is
            self.loader.sampler.set_epoch(epoch)
            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            # supports early stopping based on the test loss, or just save always is no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss if self.test_dataset is not None else float("inf")
                self.save_checkpoint()


def init_process(rank,model,train_dataset,test_dataset,config, world_size):
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    # create model, datasets, etc.
    trainer = Trainer_DDP(model, train_dataset, test_dataset, config, rank, world_size)
    trainer.train()


#FSDP training
class Trainer_FSDP:
    def __init__(self, model, train_dataset, test_dataset, config, rank, world_size):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Use available GPUs if present
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def setup_ddp(self):
        # Distributed Data Parallel setup
        setup(self.rank, self.world_size)
        self.model = self.model().to(self.rank)
        self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

    def initialize_optimizer(self):
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def run_epoch(self, split):
        print("running")
        is_train = split == "train"
        model.train(is_train)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=self.rank, shuffle=False, drop_last=False)
        data = self.train_dataset if is_train else self.test_dataset

        self.loader = DataLoader(
            data,
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=train_sampler
        )
        # Set up auto wrap policy for our model
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(rank)

        # Starting and ending for our cuda event denoting when to allocate and free the memory
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        # Setting up FSDP on our model
        model = FSDP(self.model)

        losses = []
        pbar = tqdm(enumerate(self.loader), total=len(self.loader)) if is_train else enumerate(self.loader)
        for it, (x, y) in pbar:
            # Place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward the model
            with torch.set_grad_enabled(is_train):
                loss = model(x, y)
                logits = model(x)
                losses.append(loss.item())

            if is_train:
                # Backprop and update the parameters
                model.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # Decay the learning rate based on our progress
                if self.config.lr_decay:
                    self.tokens += (y >= 0).sum()
                    if self.tokens < self.config.warmup_tokens:
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        progress = float(self.tokens - self.config.warmup_tokens) / float(
                            max(1, self.config.final_tokens - self.config.warmup_tokens)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = self.config.learning_rate

                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, lr {lr:e}")
                cleanup()

        if not is_train:
            test_loss = float(np.mean(losses))
            logger.info(f"test loss: {test_loss}")
            return test_loss

    def train(self):
        self.setup_ddp()
        self.initialize_optimizer()
        best_loss = float("inf")
        self.tokens = 0  # Counter used for learning rate decay
        for epoch in range(self.config.max_epochs):
            # If we are using DistributedSampler, we have to tell it which epoch this is
            self.loader.sampler.set_epoch(epoch)
            self.run_epoch("train")
            if self.test_dataset is not None:
                test_loss = self.run_epoch("test")

            # Supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss if self.test_dataset is not None else float("inf")
                self.save_checkpoint()

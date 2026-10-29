"""Shared history/spatial/cross attention and per-sequence KV caches.

Caches belong to a sequence, never to state_dict. Incremental attention contains
only past/current whole boards, so is_causal=False (all 90 current squares visible).
"""
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint



@dataclass
class KVCache:
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    valid: list = field(default_factory=list)

    def append(self, k, v, valid, limit):
        self.keys.append(k)
        self.values.append(v)
        self.valid.append(valid)
        while len(self.keys) > limit:
            self.keys.pop(0)
            self.values.pop(0)
            self.valid.pop(0)

    def detach_(self):
        self.keys = [k.detach() for k in self.keys]
        self.values = [v.detach() for v in self.values]
        return self


@dataclass
class CoreCache:
    history: list
    cross: KVCache = field(default_factory=KVCache)


@dataclass
class SequenceCache:
    cores: list
    batch_size: int
    time: int = 0

    def detach_(self):
        """Keep sliding memory values and absolute time, truncate only autograd."""
        for core in self.cores:
            for kv in [*core.history, core.cross]: kv.detach_()
        return self


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.width = cfg.heads, cfg.dim // cfg.heads
        self.limit, self.recompute = cfg.context_plies, cfg.checkpoint_attention
        self.q = nn.Linear(cfg.dim,cfg.dim)
        self.k = nn.Linear(cfg.dim,cfg.dim)
        self.v = nn.Linear(cfg.dim,cfg.dim)
        self.out = nn.Linear(cfg.dim,cfg.dim)

    def split(self, x):
        return x.reshape(x.shape[0],90,self.heads,self.width).transpose(1,2)

    def forward(self, query, source, valid, cache=None):
        q, k, v = self.split(self.q(query)), self.split(self.k(source)), self.split(self.v(source))
        if cache is None:
            a = F.scaled_dot_product_attention(q,k,v,is_causal=False)
        else:
            cache.append(k,v,valid,self.limit)
            # Store references to per-board K/V, not all growing concatenations in autograd.
            n = len(cache.keys)
            key_valid = torch.stack(cache.valid,dim=1).repeat_interleave(90,dim=1)
            # A fully padded member gets one harmless key; its query/output is zeroed.
            safe_mask = key_valid.clone()
            safe_mask[:,0] |= ~safe_mask.any(dim=1)
            mask = safe_mask[:,None,None,:]

            def attend(query_heads, mask_tensor, *blocks):
                keys = torch.cat(blocks[:n],dim=2)
                values = torch.cat(blocks[n:],dim=2)
                return F.scaled_dot_product_attention(query_heads,keys,values,
                                                      attn_mask=mask_tensor,is_causal=False)

            args = (q,mask,*cache.keys,*cache.values)
            # PPO replays in eval mode with autograd; recomputation still saves memory.
            if torch.is_grad_enabled() and self.recompute:
                a = checkpoint(attend,*args,use_reentrant=False)
            else:
                a = attend(*args)
        return self.out(a.transpose(1,2).reshape(query.shape[0],90,-1))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.ffn = nn.Sequential(nn.Linear(cfg.dim,cfg.ffn_dim),nn.GELU(),nn.Linear(cfg.ffn_dim,cfg.dim))
        # Keep existing parameter names/order for strict loading of older weights.
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.ffn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, valid, cache=None, query=None, source=None):
        a = self.attn(x if query is None else query,x if source is None else source,valid,cache)
        h = self.norm1(x + self.attn_dropout(F.gelu(a)))
        return self.norm2(h + self.ffn_dropout(self.ffn(h))).masked_fill(~valid[:,None,None],0)


class Core(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.mix = nn.LayerNorm(cfg.dim)
        self.q_content = nn.LayerNorm(cfg.dim,elementwise_affine=False)
        self.q_mix = nn.LayerNorm(cfg.dim)
        self.history = nn.ModuleList([Block(cfg) for _ in range(cfg.history_depth)])
        self.spatial = nn.ModuleList([Block(cfg) for _ in range(cfg.spatial_depth)])
        self.cross = Block(cfg)

    def forward(self, x, time_code, valid, cache):
        e = self.mix(x + self.beta * time_code)
        for block, kv in zip(self.history,cache.history):
            e = block(e,valid,kv)
        s = x
        for block in self.spatial:
            s = block(s,valid)
        q = self.q_mix(self.q_content(s) + self.beta * time_code)
        # E already passed history FFN + LayerNorm; cross K/V use fresh projections.
        return self.cross(s,valid,cache.cross,query=q,source=e)



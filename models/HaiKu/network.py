from dataclasses import replace
import math
import torch
from torch import nn
from .config import Haiku5Config
from models.common import KVCache, CoreCache, SequenceCache, Attention, Block, Core


class Haiku5(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Haiku5Config()
        c = self.config
        self.cnn = nn.Sequential(nn.Conv2d(15,c.cnn_channels,3,padding=1),nn.GELU(),
                                 nn.Conv2d(c.cnn_channels,c.cnn_channels,3,padding=1),nn.GELU(),
                                 nn.Conv2d(c.cnn_channels,c.dim,1))
        self.content_norm = nn.LayerNorm(c.dim,elementwise_affine=False)
        self.position_norm = nn.LayerNorm(c.dim,elementwise_affine=False)
        self.row_embed, self.col_embed = nn.Embedding(10,c.dim), nn.Embedding(9,c.dim)
        nn.init.normal_(self.row_embed.weight,std=0.02)
        nn.init.normal_(self.col_embed.weight,std=0.02)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('rows',torch.arange(90)//9)
        self.register_buffer('cols',torch.arange(90)%9)
        self.cores = nn.ModuleList([Core(c) for _ in range(c.cores)])
        self.policy_head = nn.Linear(c.dim,90)

    def set_dropout(self, probability):
        """Update residual dropout and its saved config without changing weights."""
        config = replace(self.config,dropout=probability)
        for module in self.modules():
            if isinstance(module,nn.Dropout): module.p = config.dropout
        self.config = config

    def set_context_plies(self, context_plies):
        """Set the saved K/V window independently of sequence length.

        Existing sequence caches are trimmed on their next use. Their time
        index is unchanged, so sliding the window never restarts time codes.
        """
        config = replace(self.config,context_plies=context_plies)
        for module in self.modules():
            if isinstance(module,Attention):
                module.limit = config.context_plies
        self.config = config

    def new_cache(self, batch_size):
        return SequenceCache([CoreCache([KVCache() for _ in range(self.config.history_depth)])
                              for _ in self.cores],batch_size)

    def time_encoding(self, t, device, dtype):
        frequencies = torch.exp(torch.arange(0,self.config.dim,2,device=device,dtype=torch.float32)
                                * (-math.log(10000.0)/self.config.dim))
        e = torch.empty(self.config.dim,device=device,dtype=torch.float32)
        e[0::2], e[1::2] = torch.sin(t*frequencies), torch.cos(t*frequencies)
        return e.to(dtype).reshape(1,1,-1)

    def forward_step(self, x, cache=None, valid=None):
        if x.ndim != 4 or x.shape[1:] != (15,10,9): raise ValueError('X_t 必须是 [B,15,10,9]')
        batch = x.shape[0]
        cache = self.new_cache(batch) if cache is None else cache
        if cache.batch_size != batch: raise ValueError('缓存 batch 不匹配；新 batch 必须重建缓存')
        valid = torch.ones(batch,device=x.device,dtype=torch.bool) if valid is None else valid.to(x.device).bool()
        if valid.shape != (batch,): raise ValueError('valid 必须是 [B]')
        content = self.cnn(x).flatten(2).transpose(1,2)
        position = self.position_norm(self.row_embed(self.rows)+self.col_embed(self.cols))
        h = self.content_norm(content) + self.alpha * position[None]
        h = h.masked_fill(~valid[:,None,None],0)
        e = self.time_encoding(cache.time,x.device,h.dtype)
        for core, core_cache in zip(self.cores,cache.cores):
            h = core(h,e,valid,core_cache)
        logits = self.policy_head(h).flatten(1).masked_fill(~valid[:,None],0)
        cache.time += 1
        return logits, cache

    def forward(self, x, lengths=None):
        if x.ndim == 4:
            return self.forward_step(x)[0]
        if x.ndim != 5: raise ValueError('X 必须是 [T,B,15,10,9]')
        t_count, batch = x.shape[:2]
        if t_count < 1 or batch < 1: raise ValueError('T 和 B 必须为正')
        lengths = torch.full((batch,),t_count,device=x.device,dtype=torch.long) if lengths is None else lengths.to(x.device)
        if lengths.shape != (batch,) or ((lengths < 0) | (lengths > t_count)).any():
            raise ValueError('lengths 越界')
        cache, outputs = self.new_cache(batch), []
        # Fresh graph/cache every call; optimizer steps never reuse old K/V.
        real_steps = int(lengths.max())
        for t in range(real_steps):
            logits, cache = self.forward_step(x[t],cache,t < lengths)
            outputs.append(logits)
        if not outputs:
            return x.new_zeros(t_count,batch,8100) + self.policy_head.weight.sum()*0
        output = torch.stack(outputs)
        if real_steps < t_count:
            output = torch.cat((output,output.new_zeros(t_count-real_steps,batch,8100)),dim=0)
        return output

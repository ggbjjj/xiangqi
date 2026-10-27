"""Five independent cores, two single-head bridges, policy and WDL outputs."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .config import OpusConfig
from models.common import Core, Attention
from models.HaiKu.network import Haiku5


@dataclass
class OpusOutput:
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    valid: torch.Tensor

    @property
    def value_probabilities(self):
        return self.value_logits.float().softmax(-1).masked_fill(~self.valid[..., None], 0)

    @property
    def expected_value(self):
        p = self.value_probabilities
        return p[..., 0] - p[..., 2]


class SingleHeadCross(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))

    def forward(self, query, source, valid):
        q, k, v = self.q(query)[:, None], self.k(source)[:, None], self.v(source)[:, None]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)[:, 0]
        return out.masked_fill(~valid[:, None, None], 0)


class AttentionPolicyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def forward(self, x):
        # Raw source/destination scores; no V aggregation or row Softmax.
        return (self.q(x) @ self.k(x).transpose(-2, -1) / self.scale).flatten(1)


class Opus5(Haiku5):
    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.config = config or OpusConfig()
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
        self.cores = nn.ModuleList([
            Core(SimpleNamespace(**{**c.to_dict(), 'heads': heads, 'context_plies': window}))
            for heads,window in zip(c.heads,c.context_windows)
        ])
        self.bridge_a, self.bridge_b = SingleHeadCross(c.dim), SingleHeadCross(c.dim)
        self.policy_head = AttentionPolicyHead(c.dim)
        self.value_head = nn.Sequential(nn.LayerNorm(c.dim), nn.Linear(c.dim,c.value_hidden),
                                        nn.GELU(), nn.Linear(c.value_hidden,3))

    def set_context_windows(self, windows):
        c = replace(self.config, context_windows=tuple(windows))
        for core, limit in zip(self.cores, c.context_windows):
            for module in core.modules():
                if isinstance(module, Attention): module.limit = limit
        self.config = c

    def set_context_plies(self, windows):
        if isinstance(windows, int):
            raise ValueError('Opus 请设置五个 context_windows，而非单个 context_plies')
        self.set_context_windows(windows)

    def forward_step(self, x, cache=None, valid=None):
        if x.ndim != 4 or x.shape[1:] != (15,10,9): raise ValueError('X_t 必须是 [B,15,10,9]')
        batch = x.shape[0]
        cache = self.new_cache(batch) if cache is None else cache
        if cache.batch_size != batch or len(cache.cores) != 5: raise ValueError('Opus 缓存 batch 或核心数量不匹配')
        valid = torch.ones(batch,device=x.device,dtype=torch.bool) if valid is None else valid.to(x.device).bool()
        if valid.shape != (batch,): raise ValueError('valid 必须是 [B]')
        content = self.cnn(x).flatten(2).transpose(1,2)
        position = self.position_norm(self.row_embed(self.rows)+self.col_embed(self.cols))
        z = (self.content_norm(content)+self.alpha*position[None]).masked_fill(~valid[:,None,None],0)
        e = self.time_encoding(cache.time,x.device,z.dtype)
        f1 = self.cores[0](z,e,valid,cache.cores[0])
        f2 = self.cores[1](f1,e,valid,cache.cores[1])
        f3 = self.cores[2](f2,e,valid,cache.cores[2])
        f4 = self.cores[3](z,e,valid,cache.cores[3])
        p = self.bridge_a(f4,f2,valid)
        f5 = self.cores[4](p,e,valid,cache.cores[4])
        o = self.bridge_b(f5,f3,valid)
        policy = self.policy_head(f3).masked_fill(~valid[:,None],0)
        value = self.value_head(o.mean(1)).masked_fill(~valid[:,None],0)
        cache.time += 1
        return OpusOutput(policy,value,valid), cache

    def forward(self, x, lengths=None):
        if x.ndim == 4: return self.forward_step(x)[0]
        if x.ndim != 5: raise ValueError('X 必须是 [T,B,15,10,9]')
        t_count,batch = x.shape[:2]
        if min(t_count,batch)<1: raise ValueError('T 和 B 必须为正')
        from models.HaiKu.loss import valid_steps
        lengths = torch.full((batch,),t_count,device=x.device,dtype=torch.long) if lengths is None else lengths.to(x.device)
        valid = valid_steps(lengths,t_count,batch,x.device)
        cache, policies, values = self.new_cache(batch), [], []
        for t in range(int(lengths.max())):
            out,cache = self.forward_step(x[t],cache,valid[t])
            policies.append(out.policy_logits); values.append(out.value_logits)
        def stack(items,width):
            if not items: return x.new_zeros(t_count,batch,width)+next(self.parameters()).sum()*0
            result = torch.stack(items)
            if len(items)<t_count:
                result = torch.cat((result,result.new_zeros(t_count-len(items),batch,width)),0)
            return result
        return OpusOutput(stack(policies,8100),stack(values,3),valid)

    def forward_chunks(self, x, lengths, chunk_size=None):
        """Backward each yielded chunk before advancing; step optimizer after all chunks."""
        from models.HaiKu.loss import valid_steps
        if x.ndim != 5 or x.shape[2:] != (15,10,9): raise ValueError('X 必须是 [T,B,15,10,9]')
        t_count,batch = x.shape[:2]
        if min(t_count,batch)<1: raise ValueError('T 和 B 必须为正')
        valid = valid_steps(lengths,t_count,batch,x.device)
        chunk_size = self.config.tbptt_steps if chunk_size is None else chunk_size
        if type(chunk_size) is not int or chunk_size<1: raise ValueError('截断长度必须为正整数')
        cache = self.new_cache(batch)
        for start in range(0,int(lengths.max()),chunk_size):
            stop = min(start+chunk_size,int(lengths.max()))
            policies,values = [],[]
            for t in range(start,stop):
                out,cache = self.forward_step(x[t],cache,valid[t])
                policies.append(out.policy_logits); values.append(out.value_logits)
            yield start,stop,OpusOutput(torch.stack(policies),torch.stack(values),valid[start:stop])
            cache.detach_()
            del policies,values,out

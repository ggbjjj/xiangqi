"""L = λ_book CE + λ_ppo PPO + λ_value WDL_CE - λ_entropy H.

Each component uses its own valid count. Supply whole-logical-batch counts
for chunk/microbatch backward; zero-weight branches are never evaluated.
"""
from dataclasses import dataclass, fields
import math
import torch
from torch.nn import functional as F
from models.HaiKu.loss import valid_steps
from .exploration import behavior_log_probs,behavior_entropy


@dataclass(frozen=True)
class LossWeights:
    book: float = 1.
    ppo: float = 0.
    value: float = 0.
    entropy: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in (self.book,self.ppo,self.value,self.entropy)):
            raise ValueError('损失权重必须为非负有限数')

    @classmethod
    def selfplay(cls,with_value=True): return cls(0.,1.,1. if with_value else 0.,.01)

    @classmethod
    def mixed(cls,with_value=True): return cls(.1,1.,1. if with_value else 0.,.01)


@dataclass(frozen=True)
class LossCounts:
    book: int = 0
    ppo: int = 0
    value: int = 0
    entropy: int = 0

    def __add__(self,other):
        return LossCounts(*(getattr(self,f.name)+getattr(other,f.name) for f in fields(self)))


@dataclass
class LossBatch:
    actions: torch.Tensor             # [T,B], int64
    legal: torch.Tensor               # [T,B,8100]
    lengths: torch.Tensor             # [B]
    book_mask: torch.Tensor | None = None
    selfplay_mask: torch.Tensor | None = None
    value_targets: torch.Tensor | None = None   # W/D/L = 0/1/2, unknown=-100
    value_mask: torch.Tensor | None = None
    old_log_mu: torch.Tensor | None = None
    old_value: torch.Tensor | None = None
    temperature: object = 1.
    epsilon: object = 0.

    @classmethod
    def books(cls,actions,legal,lengths,**kwargs):
        valid=valid_steps(lengths,*actions.shape,actions.device)
        return cls(actions,legal,lengths,book_mask=valid,**kwargs)

    def to(self,device):
        return LossBatch(**{f.name:(getattr(self,f.name).to(device) if isinstance(getattr(self,f.name),torch.Tensor)
                                    else getattr(self,f.name)) for f in fields(self)})

    def slice(self,start,stop):
        values={}
        for f in fields(self):
            v=getattr(self,f.name)
            if f.name=='lengths': v=(v-start).clamp(0,stop-start)
            elif isinstance(v,torch.Tensor) and v.ndim>=2: v=v[start:stop]
            values[f.name]=v
        return LossBatch(**values)

    def masks(self):
        if self.actions.ndim!=2 or self.actions.dtype!=torch.long: raise ValueError('actions 必须是 [T,B] int64')
        if self.legal.shape!=(*self.actions.shape,8100): raise ValueError('legal 维度错误')
        valid=valid_steps(self.lengths,*self.actions.shape,self.actions.device)
        def mask(value):
            if value is None: return torch.zeros_like(valid)
            if value.shape!=valid.shape: raise ValueError('损失掩码维度错误')
            return value.to(valid.device).bool() & valid
        book,own=mask(self.book_mask),mask(self.selfplay_mask)
        if (book & own).any(): raise ValueError('同一步不能同时作为棋谱标签与 PPO 样本')
        known=torch.zeros_like(valid)
        if self.value_targets is not None:
            if self.value_targets.shape!=valid.shape or self.value_targets.dtype!=torch.long:
                raise ValueError('value_targets 必须是 [T,B] int64')
            known=(self.value_targets!=-100) & valid
        value=known & (mask(self.value_mask) if self.value_mask is not None else book | own)
        return valid,book,own & known,value

    def counts(self):
        _,book,own,value=self.masks()
        return LossCounts(int(book.sum()),int(own.sum()),int(value.sum()),int(own.sum()))


@dataclass
class LossResult:
    total: torch.Tensor
    sums: dict
    counts: LossCounts
    diagnostics: dict


def split_output(output):
    if isinstance(output,torch.Tensor): return output,None
    return output.policy_logits,getattr(output,'value_logits',None)


def mixed_loss(output,batch,weights=LossWeights(),*,denominators=None,clip=.2):
    policy,value_logits=split_output(output)
    batch=batch.to(policy.device)
    valid,book,own,value=batch.masks()
    if policy.shape!=batch.legal.shape: raise ValueError('策略输出维度错误')
    if not math.isfinite(clip) or not 0<clip<1: raise ValueError('PPO clip 必须在 0..1 内')
    counts=batch.counts(); denominators=counts if denominators is None else denominators
    for field in fields(counts):
        if getattr(denominators,field.name)<getattr(counts,field.name): raise ValueError('整批分母不能小于当前段样本数')
    zero=policy.float().reshape(-1)[:0].sum()
    sums={name:zero for name in ('book','ppo','value','entropy')}; diagnostics={}
    def chosen_log(log_mu,mask):
        actions=batch.actions[mask]
        if ((actions<0)|(actions>=8100)).any(): raise ValueError('动作标签越界')
        if not batch.legal[mask].gather(1,actions[:,None]).all(): raise ValueError('标签包含非法动作')
        return log_mu.gather(-1,actions[:,None]).squeeze(-1)
    def targets(mask):
        y=batch.value_targets[mask]
        if ((y<0)|(y>2)).any(): raise ValueError('价值标签越界')
        return y
    if weights.book and counts.book:
        log_pi=behavior_log_probs(policy[book],batch.legal[book])
        sums['book']=-chosen_log(log_pi,book).sum()
    if (weights.ppo or weights.entropy) and counts.ppo:
        def select(v):
            v=torch.as_tensor(v,device=policy.device,dtype=torch.float32)
            return torch.broadcast_to(v,valid.shape)[own]
        log_mu=behavior_log_probs(policy[own],batch.legal[own],select(batch.temperature),select(batch.epsilon))
        if weights.ppo:
            for field in ('old_log_mu','old_value'):
                v=getattr(batch,field)
                if v is None or v.shape!=valid.shape or not torch.isfinite(v[own]).all():
                    raise ValueError(f'PPO 需要有效的 {field}')
            old_log=batch.old_log_mu[own].float().detach()
            old_value=batch.old_value[own].float().detach()
            if (old_log>1e-5).any() or (old_value.abs()>1.00001).any(): raise ValueError('旧概率/旧价值超出范围')
            current=chosen_log(log_mu,own)
            log_ratio=current-old_log
            ratio=log_ratio.exp()
            if not torch.isfinite(ratio).all(): raise FloatingPointError('PPO 概率比溢出；请检查采样版本和数据')
            y=targets(own)
            returns=(y==0).float()-(y==2).float()
            advantage=(returns-old_value).detach()
            sums['ppo']=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).sum()
            diagnostics={'ratio_sum':ratio.detach().sum(),
                         'clip_count':((ratio-1).abs()>clip).float().sum().detach(),
                         'approx_kl_sum':((ratio-1)-log_ratio).detach().sum()}
        if weights.entropy: sums['entropy']=behavior_entropy(log_mu,batch.legal[own]).sum()
    if weights.value and counts.value:
        if value_logits is None: raise ValueError('当前模型没有价值头；请将 lambda-value 设为 0')
        if value_logits.shape!=(*valid.shape,3): raise ValueError('价值输出必须是 [T,B,3] logits')
        if not torch.isfinite(value_logits[value]).all(): raise ValueError('价值 logits 非有限值')
        sums['value']=F.cross_entropy(value_logits[value].float(),targets(value),reduction='sum')
    total=zero
    for name in sums:
        weight=getattr(weights,name); n=getattr(denominators,name)
        if weight and n: total=total+(-1 if name=='entropy' else 1)*weight*sums[name]/n
    return LossResult(total,sums,counts,diagnostics)

"""Stable legal behavior distribution shared by rollout sampling and PPO replay."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Exploration:
    temperature: float = 1.2
    epsilon: float = 0.02

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature<=0: raise ValueError('温度必须为正有限数')
        if not math.isfinite(self.epsilon) or not 0<=self.epsilon<=1: raise ValueError('epsilon 必须在 0..1')


def behavior_log_probs(logits, legal, temperature=1., epsilon=0., valid=None):
    """log μ = log[(1-ε) softmax(legal_logits/τ) + ε / legal_count].

    τ/ε may be scalars or one value per position in logits.shape[:-1].
    Illegal/padded entries are -inf; empty legal sets are allowed only for padding.
    """
    if logits.shape!=legal.shape or logits.shape[-1]!=8100: raise ValueError('logits/legal 必须以 8100 维动作结尾')
    legal=legal.to(device=logits.device,dtype=torch.bool)
    valid=torch.ones(logits.shape[:-1],device=logits.device,dtype=torch.bool) if valid is None else valid.to(logits.device).bool()
    if valid.shape!=logits.shape[:-1]: raise ValueError('valid 维度错误')
    if (valid & ~legal.any(-1)).any(): raise ValueError('真实局面没有合法动作，应先判终局')
    if not torch.isfinite(logits[valid]).all(): raise ValueError('策略 logits 非有限值')
    def parameter(value,name):
        try: return torch.broadcast_to(torch.as_tensor(value,device=logits.device,dtype=torch.float32),valid.shape).detach()
        except RuntimeError as error: raise ValueError(f'{name} 维度错误') from error
    tau,eps=parameter(temperature,'temperature'),parameter(epsilon,'epsilon')
    if not torch.isfinite(tau[valid]).all() or (tau[valid]<=0).any(): raise ValueError('温度必须为正有限数')
    if not torch.isfinite(eps[valid]).all() or ((eps[valid]<0)|(eps[valid]>1)).any(): raise ValueError('epsilon 必须在 0..1')
    tau=tau.masked_fill(~valid,1); eps=eps.masked_fill(~valid,0)
    safe_legal=legal | ~valid[...,None]
    z=logits.float().masked_fill(~valid[...,None],0)/tau[...,None]
    q=F.log_softmax(z.masked_fill(~safe_legal,-torch.inf),-1)
    # Avoid logaddexp(-inf,-inf) at illegal positions (NaN backward at ε=0).
    q=q.masked_fill(~safe_legal,0)
    n=legal.sum(-1).clamp_min(1).float()
    log_mu=torch.logaddexp(q+torch.log1p(-eps)[...,None],
                           (eps.log()-n.log())[...,None])
    return log_mu.masked_fill(~legal | ~valid[...,None],-torch.inf)


def behavior_entropy(log_mu, legal):
    return -(log_mu.exp()*log_mu.masked_fill(~legal.bool(),0)).sum(-1)


def sample_action(logits, legal, exploration=Exploration(), generator=None):
    if logits.shape!=(8100,) or legal.shape!=(8100,): raise ValueError('单局采样需要 [8100] logits/legal')
    log_mu=behavior_log_probs(logits,legal,exploration.temperature,exploration.epsilon)
    action=int(torch.multinomial(log_mu.exp().cpu(),1,generator=generator))
    return action,float(log_mu[action])


def sample_actions(logits,legal,exploration,generators,valid=None):
    """Batch behavior probabilities on the model device; independent seeded RNGs per game."""
    if logits.ndim!=2 or logits.shape[-1]!=8100 or len(generators)!=len(logits):
        raise ValueError('批量采样需要 [B,8100] logits 和 B 个独立随机生成器')
    active=[True]*len(logits) if valid is None else list(valid)
    mask=torch.tensor(active,device=logits.device,dtype=torch.bool)
    log_mu=behavior_log_probs(logits,legal,exploration.temperature,exploration.epsilon,mask).detach().cpu()
    actions=[]; log_probs=[]
    for i,is_active in enumerate(active):
        if not is_active: actions.append(None); log_probs.append(None); continue
        action=int(torch.multinomial(log_mu[i].exp(),1,generator=generators[i]))
        actions.append(action); log_probs.append(float(log_mu[i,action]))
    return actions,log_probs

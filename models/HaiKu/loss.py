"""Legal-action CE, with safe padding before normalization (no 0 * -inf)."""
import torch
from torch.nn import functional as F


def valid_steps(lengths, t_count, batch, device):
    lengths = lengths.to(device=device)
    if lengths.shape != (batch,) or lengths.dtype != torch.long or ((lengths<0)|(lengths>t_count)).any():
        raise ValueError('lengths 必须是 [B] int64，范围 0..T')
    return torch.arange(t_count,device=device)[:,None] < lengths[None,:]


def masked_probabilities(logits, legal, valid=None):
    legal = legal.to(device=logits.device,dtype=torch.bool)
    if legal.shape != logits.shape or logits.shape[-1] != 8100: raise ValueError('动作维度错误')
    valid = torch.ones(logits.shape[:-1],device=logits.device,dtype=torch.bool) if valid is None else valid
    if (valid & ~legal.any(-1)).any(): raise ValueError('真实局面没有合法动作')
    if not torch.isfinite(logits[valid]).all(): raise ValueError('logits 非有限值')
    z = logits.float().masked_fill(~valid[...,None],0)
    z = z.masked_fill(~(legal | ~valid[...,None]),-torch.inf)
    return F.softmax(z,dim=-1).masked_fill(~valid[...,None],0)


def masked_policy_ce(logits, y, legal, lengths, reduction='mean'):
    if logits.ndim != 3 or logits.shape[-1] != 8100 or legal.shape != logits.shape:
        raise ValueError('logits/legal 必须是 [T,B,8100]')
    valid = valid_steps(lengths,*logits.shape[:2],logits.device)
    legal = legal.to(logits.device).bool()
    y = y.to(logits.device)
    if y.shape == logits.shape:
        if not torch.isfinite(y[valid]).all() or (y[valid]<0).any(): raise ValueError('非法标签值')
        if not torch.allclose(y.sum(-1)[valid],torch.ones_like(y.sum(-1)[valid]),atol=1e-6):
            raise ValueError('真实标签概率之和必须为 1')
        if ((y>0)&~legal&valid[...,None]).any(): raise ValueError('标签包含非法动作')
        dense = True
    elif y.shape == logits.shape[:2] and y.dtype == torch.long:
        target = y[valid]
        if ((target<0)|(target>=8100)).any(): raise ValueError('动作标签越界')
        if not legal[valid].gather(1,target[:,None]).all(): raise ValueError('标签包含非法动作')
        dense = False
    else:
        raise ValueError('Y 必须是 [T,B,8100] 分布或 [T,B] int64 动作编号')
    probs = masked_probabilities(logits,legal,valid)
    z = logits[valid].float().masked_fill(~legal[valid],-torch.inf)
    logp = F.log_softmax(z,dim=-1)
    if dense:
        target = y[valid].float()
        losses = -(target * logp.masked_fill(target==0,0)).sum(-1)
    else:
        losses = -logp.gather(1,y[valid][:,None]).squeeze(1)
    loss_tb = logits.float().masked_fill(~valid[...,None],0).sum(-1)*0
    loss_tb = loss_tb.masked_scatter(valid,losses)
    if reduction == 'sum': loss = loss_tb.sum()
    elif reduction == 'mean': loss = loss_tb.sum()/valid.sum().clamp_min(1)
    else: raise ValueError('reduction 必须是 sum 或 mean')
    return loss,loss_tb,probs

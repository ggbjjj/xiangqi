"""Outcome supervision for the current side: win=0, draw=1, loss=2.

Unknown outcomes use -100 and contribute no loss. Book move labels alone
must never be converted into invented outcome labels.
"""
import torch
from torch.nn import functional as F
from models.HaiKu.loss import valid_steps, masked_policy_ce, masked_probabilities


def masked_value_ce(logits, targets, lengths, reduction='mean'):
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError('价值 logits 必须是 [T,B,3]')
    targets = targets.to(logits.device)
    if targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError('价值标签必须是 [T,B] int64；胜/和/负=0/1/2，未知=-100')
    valid = valid_steps(lengths,*logits.shape[:2],logits.device) & (targets != -100)
    if ((targets[valid]<0)|(targets[valid]>2)).any(): raise ValueError('价值标签越界')
    if not torch.isfinite(logits[valid]).all(): raise ValueError('价值 logits 非有限值')
    losses = F.cross_entropy(logits[valid].float(),targets[valid],reduction='none')
    loss_tb = logits.float().masked_fill(~valid[...,None],0).sum(-1)*0
    loss_tb = loss_tb.masked_scatter(valid,losses)
    if reduction == 'sum': loss = loss_tb.sum()
    elif reduction == 'mean': loss = loss_tb.sum()/valid.sum().clamp_min(1)
    else: raise ValueError('reduction 必须是 sum 或 mean')
    return loss,loss_tb

"""Held-out book evaluation, replaying each complete history without gradients."""
from contextlib import nullcontext
import math
import torch

from models.HaiKu.loss import valid_steps
from .control import check_stop
from .data import collate_examples
from .engine import iter_chunks
from .loss import LossWeights,mixed_loss


@torch.inference_mode()
def evaluate_books(model,dataset,*,micro_batch=16,chunk_size=16,precision='none',cancel_event=None,progress=None):
    if not len(dataset): raise ValueError('古谱验证集为空')
    if micro_batch<1: raise ValueError('验证微批量必须为正')
    if precision not in ('none','bf16','fp16'): raise ValueError('未知 precision')
    device=next(model.parameters()).device
    if precision=='fp16' and device.type!='cuda': raise ValueError('fp16 仅支持 CUDA')
    previous_training=model.training; model.eval()
    loss_sum=0.; count=0; top1=0; top3=0
    try:
        for start_game in range(0,len(dataset),micro_batch):
            check_stop(cancel_event)
            if progress:
                progress({'micro_index':start_game//micro_batch+1,'micro_total':math.ceil(len(dataset)/micro_batch)})
            batch=collate_examples([dataset[i] for i in range(start_game,min(len(dataset),start_game+micro_batch))])
            x=batch.x.to(device); labels=batch.loss.to(device)
            context=(nullcontext() if precision=='none' else torch.autocast(device.type,
                dtype=torch.bfloat16 if precision=='bf16' else torch.float16))
            with context:
                for start,stop,output in iter_chunks(model,x,labels.lengths,chunk_size):
                    check_stop(cancel_event)
                    part=labels.slice(start,stop)
                    result=mixed_loss(output,part,LossWeights())
                    valid=valid_steps(part.lengths,stop-start,len(part.lengths),device)
                    policy=output.policy_logits[valid].float().masked_fill(~part.legal[valid],-torch.inf)
                    ranking=policy.topk(3,dim=-1).indices
                    expected=part.actions[valid]
                    loss_sum+=float(result.sums['book']); count+=result.counts.book
                    top1+=int((ranking[:,0]==expected).sum()); top3+=int((ranking==expected[:,None]).any(-1).sum())
                    del output,result,policy,ranking
        check_stop(cancel_event)
        if not count or not math.isfinite(loss_sum): raise ValueError('古谱验证集没有有效步或损失非有限值')
        return {'loss':loss_sum/count,'top1':top1/count,'top3':top3/count,'games':len(dataset),'steps':count}
    finally: model.train(previous_training)

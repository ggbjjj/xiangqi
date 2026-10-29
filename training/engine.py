from contextlib import nullcontext
from dataclasses import dataclass
import time
import torch
from models.HaiKu.loss import valid_steps
from .loss import LossWeights,LossCounts,mixed_loss,split_output
from .control import check_stop


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    value_logits: torch.Tensor | None = None


def iter_chunks(model,x,lengths,chunk_size=16):
    """Fresh cache for each micro-batch; retain detached K/V across its chunks.

    Common incremental contract: forward_step(input,cache,valid) -> output,cache.
    No cache is attached to the model or reused by the next call.
    """
    if type(chunk_size) is not int or chunk_size<1: raise ValueError('chunk_size 必须为正整数')
    if x.ndim!=5 or x.shape[2:]!=(15,10,9): raise ValueError('X 必须是 [T,B,15,10,9]')
    valid=valid_steps(lengths,*x.shape[:2],x.device); cache=None
    for start in range(0,int(lengths.max()),chunk_size):
        stop=min(start+chunk_size,int(lengths.max())); policies=[]; values=[]
        for t in range(start,stop):
            output,cache=model.forward_step(x[t],cache,valid[t])
            policy,value=split_output(output); policies.append(policy)
            if value is not None: values.append(value)
        if values and len(values)!=len(policies): raise ValueError('模型不能在序列中途改变价值输出接口')
        yield start,stop,ModelOutput(torch.stack(policies),torch.stack(values) if values else None)
        if cache is not None:
            if not hasattr(cache,'detach_'): raise ValueError('模型序列缓存必须提供 detach_()')
            cache.detach_()
        del output,policies,values


def train_logical_batch(model,optimizer,batches,weights=LossWeights(),*,chunk_size=16,clip=.2,
                        max_grad_norm=1.,precision='none',scaler=None,
                        behavior_checkpoint_id=None,behavior_sha256=None,cancel_event=None,progress=None):
    """Accumulate all source-specific normalized chunks; update once per logical batch.

    PPO provenance must match the explicitly selected rollout snapshot. The
    caller loads that snapshot before its first update, then may run a bounded
    number of PPO epochs against the same fixed old measurements.
    """
    batches=list(batches)
    if not batches: raise ValueError('逻辑批量不能为空')
    counts=LossCounts()
    for batch in batches:
        counts=counts+batch.loss.counts()
        if weights.ppo and batch.loss.counts().ppo:
            expected=(behavior_checkpoint_id,behavior_sha256,model.config.architecture)
            if not behavior_checkpoint_id or not behavior_sha256 or batch.provenance!={expected}:
                raise ValueError('PPO 数据必须来自明确指定的同一采样参数版本；请使用近期本轮 run_id')
    device=next(model.parameters()).device
    if precision not in ('none','bf16','fp16'): raise ValueError('未知 precision')
    if precision=='fp16' and device.type!='cuda': raise ValueError('fp16 仅支持 CUDA')
    if max_grad_norm<=0: raise ValueError('梯度裁剪上限必须为正')
    if scaler is None: scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    optimizer.zero_grad(set_to_none=True)
    # Supervised training uses dropout. Active PPO must match the deterministic
    # rollout policy, so disable dropout for that entire logical batch (with grad).
    model.train(not bool(weights.ppo and counts.ppo))
    sums={k:0. for k in ('book','ppo','value','entropy')}
    diagnostics={'ratio_sum':0.,'clip_count':0.,'approx_kl_sum':0.}
    active=False
    batch_started=time.perf_counter()
    for micro_index,batch in enumerate(batches,1):
        check_stop(cancel_event)
        x=batch.x.to(device); loss_batch=batch.loss.to(device)
        real_steps=int(loss_batch.lengths.max()); previous_stop=0
        micro_sums={k:0. for k in sums}; micro_counts=LossCounts()
        if not real_steps:
            if progress:
                progress({'advance':1.,'steps':0,'micro_index':micro_index,'micro_total':len(batches),
                          'games':len(loss_batch.lengths),'loss':0.,'chunk_loss':0.,
                          'chunk_terms':dict(micro_sums),'chunk_counts':micro_counts.__dict__,
                          'terms':dict(micro_sums),'counts':micro_counts.__dict__,'finished_micro':True})
            continue
        context=(nullcontext() if precision=='none' else torch.autocast(device.type,
                  dtype=torch.bfloat16 if precision=='bf16' else torch.float16))
        with context:
            for start,stop,output in iter_chunks(model,x,loss_batch.lengths,chunk_size):
                check_stop(cancel_event)
                part=loss_batch.slice(start,stop)
                result=mixed_loss(output,part,weights,denominators=counts,clip=clip)
                if not torch.isfinite(result.total): raise FloatingPointError('混合损失非有限值')
                usable=any(getattr(weights,k)>0 and getattr(result.counts,k)>0 for k in sums)
                if usable:
                    scaler.scale(result.total).backward(); active=True
                chunk_sums={k:float(result.sums[k].detach()) for k in sums}
                for k in sums: sums[k]+=chunk_sums[k]
                for k,v in result.diagnostics.items(): diagnostics[k]+=float(v)
                for k in micro_sums: micro_sums[k]+=chunk_sums[k]
                micro_counts=micro_counts+result.counts
                if progress:
                    means={k:micro_sums[k]/max(1,getattr(micro_counts,k)) for k in micro_sums}
                    chunk_means={k:chunk_sums[k]/max(1,getattr(result.counts,k)) for k in chunk_sums}
                    chunk_work=int(loss_batch.lengths.clamp(min=start,max=stop).sub(start).clamp_min(0).sum())
                    progress({'advance':(stop-previous_stop)/max(1,real_steps),'steps':chunk_work,
                              'micro_index':micro_index,'micro_total':len(batches),'games':len(loss_batch.lengths) if stop==real_steps else 0,
                              'loss':sum((-1 if k=='entropy' else 1)*getattr(weights,k)*means[k] for k in means),
                              'chunk_loss':sum((-1 if k=='entropy' else 1)*getattr(weights,k)*chunk_means[k] for k in chunk_means),
                              'chunk_terms':chunk_means,'chunk_counts':result.counts.__dict__,
                              'terms':means,'counts':micro_counts.__dict__,'finished_micro':stop==real_steps})
                previous_stop=stop
                del output,result
    grad_norm=0.; updated=False
    if active:
        check_stop(cancel_event)
        scaler.unscale_(optimizer)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm))
        if not torch.isfinite(torch.tensor(grad_norm)) and not scaler.is_enabled():
            raise FloatingPointError('混合训练梯度非有限值')
        scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        updated=scaler.get_scale()>=scale
        if updated and weights.value and counts.value: model.value_trained=True
        if not torch.isfinite(torch.tensor(grad_norm)): grad_norm=None  # skipped fp16 update; JSON stays finite
    optimizer.zero_grad(set_to_none=True)
    means={k:sums[k]/max(1,getattr(counts,k)) for k in sums}
    elapsed=max(1e-9,time.perf_counter()-batch_started)
    return {'loss':sum((-1 if k=='entropy' else 1)*getattr(weights,k)*means[k] for k in sums),
            'terms':means,'counts':counts.__dict__,'grad_norm':grad_norm,'updated':updated,
            'duration_seconds':elapsed,'micro_batches':len(batches),'micro_batches_per_second':len(batches)/elapsed,
            'ratio_mean':diagnostics['ratio_sum']/max(1,counts.ppo),
            'clip_fraction':diagnostics['clip_count']/max(1,counts.ppo),
            'approx_kl':diagnostics['approx_kl_sum']/max(1,counts.ppo)}

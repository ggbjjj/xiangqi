"""One probability-based perplexity per held-out book slice."""
from contextlib import nullcontext
import math
import statistics
import torch
import torch.nn.functional as F

from data_pipeline.loader import collate_games
from .control import check_stop
from .data import collate_examples
from .engine import iter_chunks


PERPLEXITY_VERSION='book-slice-ppl-sliding-context-v1'


def validate_evaluation(length,ngram):
    if type(length) is not int or length<1: raise ValueError('验证切片 L 必须为正整数')
    if type(ngram) is not int or not 1<=ngram<=length:
        raise ValueError('验证 n 必须满足 1 <= n <= L')


def _score_slices(model,x,actions,legal,lengths,ngram,chunk_size,micro_batch,cancel_event):
    """Replay sliding contexts afresh so stacked caches cannot leak older inputs.

    The first window scores its entire prefix; later windows score only their
    final action. Each real action contributes exactly once. Window batches
    are bounded by micro_batch, independently of the number of slice windows.
    """
    windows=[]
    for b,length in enumerate(lengths.tolist()):
        if length:
            windows.append((b,0,min(ngram,length),True))
            windows.extend((b,t-ngram+1,t+1,False) for t in range(ngram,length))
    totals=torch.zeros(len(lengths),dtype=torch.float64,device=x.device)
    for offset in range(0,len(windows),micro_batch):
        check_stop(cancel_event)
        group=windows[offset:offset+micro_batch]
        owners=torch.tensor([w[0] for w in group],device=x.device)
        starts=torch.tensor([w[1] for w in group],device=x.device)
        sizes=torch.tensor([w[2]-w[1] for w in group],device=x.device)
        first=torch.tensor([w[3] for w in group],device=x.device)
        time=torch.arange(int(sizes.max()),device=x.device)[:,None]
        source_time=(time+starts[None,:]).clamp_max(len(x)-1)
        wx=x[source_time,owners[None,:]].clone()
        wx.masked_fill_((time>=sizes[None,:])[...,None,None,None],0)
        for start,stop,output in iter_chunks(model,wx,sizes,chunk_size):
            check_stop(cancel_event)
            t=time[start:stop]
            score=(t<sizes[None,:]) & (first[None,:] | (t==sizes[None,:]-1))
            row,col=score.nonzero(as_tuple=True)
            if not len(row): continue
            game=owners[col]; ply=starts[col]+row+start
            expected=actions[ply,game]
            mask=legal[ply,game]
            if ((expected<0)|(expected>=mask.shape[-1])).any() or not mask.gather(1,expected[:,None]).all():
                raise ValueError('验证有效步的目标动作必须合法')
            logits=output.policy_logits[row,col].float().masked_fill(~mask,-torch.inf)
            nll=-F.log_softmax(logits,dim=-1).gather(1,expected[:,None]).squeeze(1)
            if not torch.isfinite(nll).all(): raise FloatingPointError('验证动作概率非有限值')
            totals.scatter_add_(0,game,nll.double())
    return totals.cpu().tolist()


@torch.inference_mode()
def evaluate_books(model,dataset,*,micro_batch=16,ngram=16,chunk_size=16,precision='none',cancel_event=None,progress=None):
    """Treat each L-ply reader slice as a sequence, with padding probability 1/2.

    n is the maximum input context, including the current board. No window
    crosses a slice boundary. The mean of per-slice PPLs selects best;
    corpus_perplexity is also supplied as a diagnostic geometric mean.
    GameDataset and BookDataset use the same reader and scoring code.
    """
    if not len(dataset): raise ValueError('棋谱评估集为空')
    if type(micro_batch) is not int or micro_batch<1: raise ValueError('验证微批量必须为正整数')
    if type(chunk_size) is not int or chunk_size<1: raise ValueError('验证分段必须为正整数')
    if precision not in ('none','bf16','fp16'): raise ValueError('未知 precision')
    reader=getattr(dataset,'dataset',dataset)
    length=reader.T
    validate_evaluation(length,ngram)
    device=next(model.parameters()).device
    if precision=='fp16' and device.type!='cuda': raise ValueError('fp16 仅支持 CUDA')
    previous_training=model.training; model.eval(); samples=[]
    try:
        for offset in range(0,len(dataset),micro_batch):
            check_stop(cancel_event)
            items=[dataset[i] for i in range(offset,min(len(dataset),offset+micro_batch))]
            if hasattr(items[0],'x'):
                batch=collate_examples(items)
                x,actions,legal,lengths=batch.x,batch.loss.actions,batch.loss.legal,batch.loss.lengths
            else:
                x,actions,legal,lengths=collate_games(items)
            if len(x)!=length or (lengths<0).any() or (lengths>length).any():
                raise ValueError('验证读取长度必须与 L 一致')
            context=(nullcontext() if precision=='none' else torch.autocast(device.type,
                dtype=torch.bfloat16 if precision=='bf16' else torch.float16))
            with context:
                sums=_score_slices(model,x.to(device),actions.to(device),legal.to(device),
                    lengths,ngram,chunk_size,micro_batch,cancel_event)
            for b,(nll,valid) in enumerate(zip(sums,lengths.tolist())):
                key,start,stop=reader.segments[offset+b]
                padding=length-valid; log_ppl=(nll+padding*math.log(2))/length
                try: ppl=math.exp(log_ppl)
                except OverflowError: raise FloatingPointError('验证困惑度超出浮点范围') from None
                samples.append({'slice_index':offset+b,'source_game_id':key,'start_ply':start,
                    'source_stop_ply':stop,'length':valid,'padding':padding,'nll_sum':nll,
                    'log_perplexity':log_ppl,'perplexity':ppl})
            if progress:
                progress({'micro_index':offset//micro_batch+1,'micro_total':math.ceil(len(dataset)/micro_batch)})
        check_stop(cancel_event)
        values=[s['perplexity'] for s in samples]
        steps=sum(s['length'] for s in samples); nll=sum(s['nll_sum'] for s in samples)
        return {'metric':'mean_slice_perplexity','metric_version':PERPLEXITY_VERSION,
                'perplexity':statistics.fmean(values),'median_perplexity':statistics.median(values),
                'min_perplexity':min(values),'max_perplexity':max(values),
                'corpus_perplexity':math.exp(statistics.fmean(s['log_perplexity'] for s in samples)),
                'loss':nll/steps if steps else 0.,
                'games':len(samples),'source_games':len(dataset.ids),'segments':len(samples),
                'steps':steps,'padding_steps':length*len(samples)-steps,'L':length,'n':ngram,
                'padding_probability':.5,'book_selection':dataset.summary,'samples':samples}
    finally: model.train(previous_training)

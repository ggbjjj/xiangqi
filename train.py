"""HaiKu/Opus book-policy pretraining; books without outcomes do not train value."""
import argparse
from contextlib import nullcontext
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_pipeline.loader import GameDataset, collate_games
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from models.HaiKu.loss import masked_policy_ce, valid_steps
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from training.loss import LossBatch,LossWeights,LossCounts,mixed_loss
from training.progress import ConsoleProgress
from model_store import (save_checkpoint, load_checkpoint, resolve_alias,load_parameter_model,
                         get_latest_parameter_id,list_parameter_configs)
from project_paths import DATA_DB, MODEL_CATALOG, ROOT, family_model_db, model_family


def make_loader(dataset, args, training=False, epoch=0):
    return DataLoader(dataset,batch_size=args.micro_batch,shuffle=training,num_workers=args.workers,
        collate_fn=collate_games,pin_memory=args.device.startswith('cuda'),
        generator=torch.Generator().manual_seed(args.seed+epoch),persistent_workers=args.workers>0)


def amp_context(device, precision):
    if precision == 'none': return nullcontext()
    return torch.autocast(device_type=torch.device(device).type,
                          dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def policy_chunks(model, x, lengths):
    """Opus truncates graphs, not memory; caller backpropagates before next yield."""
    if hasattr(model,'forward_chunks'):
        for start,stop,output in model.forward_chunks(x,lengths):
            yield start,stop,output.policy_logits
            del output
    else:
        yield 0,len(x),model(x,lengths)


def configure_model(model, args):
    family = model_family(model.config.architecture)
    if args.architecture and model_family(args.architecture) != family:
        raise ValueError('--architecture 与检查点的模型系列不一致')
    args.architecture = family
    previous = model.config.context_plies
    if family == 'Opus':
        if args.context_plies is not None: raise ValueError('Opus 请用 --context-windows 指定五个窗口')
        if args.context_windows is not None: model.set_context_windows(args.context_windows)
        if args.heads is not None and tuple(args.heads) != model.config.heads:
            raise ValueError('恢复/评估不能更改已保存的注意力头数')
        if args.tbptt_steps is not None: model.config = replace(model.config,tbptt_steps=args.tbptt_steps)
        args.tbptt_steps = model.config.tbptt_steps
    else:
        if any(v is not None for v in (args.context_windows,args.heads,args.tbptt_steps)):
            raise ValueError('--context-windows、--heads、--tbptt-steps 是 Opus 的配置项')
        if args.context_plies is not None: model.set_context_plies(args.context_plies)
    args.context_plies = model.config.context_plies
    return previous != model.config.context_plies


@torch.inference_mode()
def evaluate(model, loader, device, precision='none', desc='评估', progress=None):
    model.eval()
    totals = {'loss_sum':0.,'steps':0,'top1':0,'top3':0,'games':0}
    for x,y,legal,lengths in (loader if progress is not None else tqdm(loader,desc=desc,leave=False)):
        if progress is not None: progress.update(phase=desc)
        x,y,legal,lengths = [v.to(device,non_blocking=True) for v in (x,y,legal,lengths)]
        totals['games'] += len(lengths)
        with amp_context(device,precision):
            for start,stop,logits in policy_chunks(model,x,lengths):
                chunk_lengths = (lengths-start).clamp(0,stop-start)
                loss,_,probs = masked_policy_ce(logits,y[start:stop],legal[start:stop],chunk_lengths,'sum')
                valid = valid_steps(chunk_lengths,stop-start,len(lengths),device)
                labels, ranking = y[start:stop][valid], probs[valid].topk(3,dim=-1).indices
                totals['loss_sum'] += loss.item()
                totals['steps'] += int(valid.sum())
                totals['top1'] += int((ranking[:,0]==labels).sum())
                totals['top3'] += int((ranking==labels[:,None]).any(-1).sum())
    if not totals['steps']: raise ValueError('评估集没有有效走子')
    n = totals['steps']
    return {'loss':totals['loss_sum']/n,'top1':totals['top1']/n,'top3':totals['top3']/n,
            'games':totals['games'],'steps':n}


def plot_history(path, title='HaiKu / Opus'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with path.open(encoding='utf-8',newline='') as stream: rows = list(csv.DictReader(stream))
    fig,ax = plt.subplots(figsize=(8,4.5))
    ax.plot([int(r['epoch']) for r in rows],[float(r['train_loss']) for r in rows],label='Train')
    ax.plot([int(r['epoch']) for r in rows],[float(r['validation_loss']) for r in rows],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Legal-action cross entropy',title=title)
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=160); plt.close(fig)


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'auto': args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('当前 PyTorch/CUDA 环境没有可用 GPU')
    if args.precision == 'auto':
        args.precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if args.device.startswith('cuda') else 'none'
    if args.precision == 'fp16' and not args.device.startswith('cuda'): raise ValueError('fp16 训练仅支持 CUDA')
    if args.precision == 'bf16' and args.device.startswith('cuda') and not torch.cuda.is_bf16_supported():
        raise ValueError('此 GPU 不支持 bf16，请使用 --precision fp16')
    if args.model_db is None:
        args.model_db = (MODEL_CATALOG if (args.resume or args.evaluate) and not args.architecture
                         else family_model_db(args.architecture or 'HaiKu'))
    if args.evaluate:
        model,_,_ = load_checkpoint(args.model_db,args.evaluate,args.device)
        configure_model(model,args)
        print(f'测试配置：T={args.T} | KV 历史窗口={args.context_plies}',flush=True)
        dataset = GameDataset(args.T,'test',args.data,args.limit_games)
        report = evaluate(model,make_loader(dataset,args),args.device,args.precision,'测试集')
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    train_set = GameDataset(args.T,'train',args.data,args.limit_games)
    val_set = GameDataset(args.T,'validation',args.data,args.limit_games)
    test_set = GameDataset(args.T,'test',args.data,args.limit_games)
    if not all(map(len,(train_set,val_set,test_set))): raise ValueError('训练/验证/测试集均须非空，请先完成入库')
    fingerprint = hashlib.sha256(('\n'.join(train_set.ids+val_set.ids+test_set.ids)).encode()).hexdigest()
    start_epoch, global_step, best_loss = 0,0,float('inf')
    run_id = args.run_name
    state = {}
    micro_step = 0
    segment_step = 0
    logical_step = 0
    base_completed=0; parent_checkpoint=None
    if args.resume:
        model,meta,state = load_checkpoint(args.model_db,args.resume,args.device,resume=True)
        if state.get('dataset_fingerprint') != fingerprint or state.get('T') != args.T or state.get('seed') != args.seed:
            raise ValueError('恢复训练须使用相同数据划分、T 和 seed')
        if 'optimizer' not in state: raise ValueError('该检查点没有优化器状态')
        start_epoch,global_step,run_id = meta['epoch'],meta['global_step'],meta['run_id']
        base_completed=meta.get('completed_epochs',meta['epoch']); parent_checkpoint=meta['checkpoint_id']
        best_loss = state.get('best_loss',float('inf'))
        micro_step = state.get('micro_step',0)
        segment_step = state.get('segment_step',0)
        logical_step = state.get('logical_step',global_step)
    else:
        family=model_family(args.architecture or 'HaiKu')
        existing=list_parameter_configs(family,path=args.model_db)
        if args.parameter_id is not None or existing:
            ident=args.parameter_id if args.parameter_id is not None else get_latest_parameter_id(family,path=args.model_db)
            model,meta,_=load_parameter_model(family,ident,path=args.model_db,device=args.device)
            base_completed=meta['completed_epochs']; parent_checkpoint=meta['checkpoint_id']; global_step=meta['global_step']
            print(f'加载 {family}/ID{meta["parameter_id"]}，已实际完成 {base_completed} 轮，本次追加 {args.epochs} 轮。',flush=True)
        elif family == 'Opus':
            options = {'checkpoint_attention':not args.no_recompute}
            if args.heads is not None: options['heads'] = tuple(args.heads)
            model = Opus5(OpusConfig(**options)).to(args.device)
        else:
            model = Haiku5(Haiku5Config(checkpoint_attention=not args.no_recompute)).to(args.device)
    if configure_model(model,args) and args.resume:
        # Validation scores under different history windows are not comparable.
        best_loss = float('inf')
        print(f'KV 历史窗口已改为 {args.context_plies}；重新选择本窗口的 best。',flush=True)
    run_id = run_id or datetime.now(timezone.utc).strftime(f'{args.architecture.lower()}5-%Y%m%d-%H%M%S-%f')
    if args.epochs <= start_epoch: raise ValueError('--epochs 是总 epoch 数，必须大于已完成 epoch')
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
    if state:
        optimizer.load_state_dict(state['optimizer'])
        if state.get('scaler'): scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['rng_state'].cpu())
        if args.device.startswith('cuda') and state.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng_state']])
    out = args.output/run_id
    out.mkdir(parents=True,exist_ok=True)
    writer = SummaryWriter(str(out/'tensorboard'),flush_secs=1)
    config = {**vars(args),'data':str(args.data),'model_db':str(args.model_db),'output':str(args.output),
              'model':model.config.to_dict(),'dataset_fingerprint':fingerprint,'torch':str(torch.__version__),
              'train_games':len(train_set),'validation_games':len(val_set),'test_games':len(test_set)}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    history = out/'history.csv'
    val_loader, test_loader = make_loader(val_set,args),make_loader(test_set,args)
    print(f'{model.config.architecture} | {args.device} | 参数 {sum(p.numel() for p in model.parameters()):,} | '
          f'T={args.T} | KV 历史窗口={args.context_plies} | '
          f'逻辑批量={args.batch_size} | 微批量={args.micro_batch}',flush=True)
    if args.architecture == 'Opus':
        print(f'梯度截断={args.tbptt_steps}；每段反向累积，逻辑批量结束更新。棋谱仅训练策略，价值头等待可靠胜和负标签。',flush=True)
    print(f'实时曲线：python -m training.dashboard --logdir "{args.output}" --port 6006',flush=True)
    display=ConsoleProgress(f'{args.architecture} 训练',(args.epochs-start_epoch)*math.ceil(len(train_set)/args.micro_batch),
        device=args.device,interval=args.progress_interval,enabled=not args.no_progress).__enter__()
    try:
        for epoch in range(start_epoch+1,args.epochs+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loader = make_loader(train_set,args,True,epoch)
            epoch_loss,epoch_steps,group_steps,group_games,group_loss = 0.,0,0,0,0.
            display.update(phase=f'轮 {epoch}/{args.epochs} 读取数据')
            for index,(x,y,legal,lengths) in enumerate(loader):
                x,y,legal,lengths = [v.to(args.device,non_blocking=True) for v in (x,y,legal,lengths)]
                # Constant normalization avoids very large fp16 gradients before unscale.
                normalizer = args.batch_size * args.T
                loss_value = 0.; processed_steps=0; previous_stop=0
                real_steps=int(lengths.max())
                with amp_context(args.device,args.precision):
                    for start,stop,logits in policy_chunks(model,x,lengths):
                        chunk_lengths = (lengths-start).clamp(0,stop-start)
                        result = mixed_loss(logits,LossBatch.books(y[start:stop],legal[start:stop],chunk_lengths),
                                            LossWeights(),denominators=LossCounts(book=normalizer))
                        if not torch.isfinite(result.total): raise FloatingPointError('训练 loss 非有限值')
                        scaler.scale(result.total).backward()
                        loss_value += result.sums['book'].item()
                        segment_step+=1
                        segment_loss=float(result.sums['book'].detach())/max(1,result.counts.book)
                        writer.add_scalar('loss/segment_total',segment_loss,segment_step)
                        writer.add_scalar('loss/segment_book',segment_loss,segment_step)
                        writer.add_scalar('optimization/parameter_updates',global_step,segment_step)
                        writer.flush()
                        chunk_steps=int(chunk_lengths.sum()); processed_steps+=chunk_steps
                        display.update((min(stop,real_steps)-previous_stop)/max(1,real_steps),work=chunk_steps,
                            games=len(lengths) if stop>=real_steps else 0,
                            phase=f'轮 {epoch}/{args.epochs} 微批 {index+1}/{len(loader)}',
                            L=loss_value/max(1,processed_steps),棋谱=loss_value/max(1,processed_steps))
                        previous_stop=min(stop,real_steps)
                        del logits,result
                steps = int(lengths.sum())
                micro_step += 1
                writer.add_scalar('loss/train_micro',loss_value/steps,micro_step)
                writer.add_scalar('loss/micro_total',loss_value/steps,micro_step)
                writer.flush()
                group_steps += steps; group_games += len(lengths); group_loss += loss_value
                epoch_loss += loss_value; epoch_steps += steps
                if group_games >= args.batch_size or index+1 == len(loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None: parameter.grad.mul_(normalizer/group_steps)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad,error_if_nonfinite=False)
                    if not torch.isfinite(grad_norm) and not scaler.is_enabled():
                        raise FloatingPointError('梯度非有限值')
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= scale_before: global_step += 1
                    logical_step+=1
                    writer.add_scalar('loss/logical_total',group_loss/group_steps,logical_step)
                    writer.add_scalar('optimization/updates_after_batch',global_step,logical_step)
                    writer.add_scalar('loss/train_batch',group_loss/group_steps,global_step)
                    writer.add_scalar('optimization/grad_norm',float(grad_norm),global_step)
                    writer.add_scalar('encoding/alpha',model.alpha.item(),global_step)
                    for j,core in enumerate(model.cores,1): writer.add_scalar(f'encoding/beta_{j}',core.beta.item(),global_step)
                    group_steps,group_games,group_loss = 0,0,0.
                    writer.flush()
                if args.no_progress: print(json.dumps({'epoch':epoch,'micro_batch':index+1,'loss':loss_value/steps,
                    'epoch_loss':epoch_loss/epoch_steps,'global_step':global_step},ensure_ascii=False),flush=True)
            metrics = evaluate(model,val_loader,args.device,args.precision,'验证集',progress=display)
            train_loss = epoch_loss/epoch_steps
            best = metrics['loss'] < best_loss
            best_loss = min(best_loss,metrics['loss'])
            display.update(phase=f'保存第 {epoch} 轮参数')
            checkpoint_id = save_checkpoint(model,path=args.model_db,run_id=run_id,epoch=epoch,
                global_step=global_step,metrics={'train_loss':train_loss,'validation':metrics},optimizer=optimizer,
                training_state={'T':args.T,'context_plies':args.context_plies,'seed':args.seed,
                  'dataset_fingerprint':fingerprint,'best_loss':best_loss,
                  'rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                  'scaler':scaler.state_dict(),'micro_step':micro_step,'segment_step':segment_step,
                  'logical_step':logical_step,'tbptt_steps':args.tbptt_steps},
                best=best,published=not args.smoke,set_default=args.set_default and best,
                completed_epochs=base_completed+epoch-start_epoch,parent_checkpoint_id=parent_checkpoint,
                description=(f'{args.description} / 累计实际训练 {base_completed+epoch-start_epoch} 轮' if args.description else ''))
            parent_checkpoint=checkpoint_id
            if best:
                best_parameter=next(r['parameter_id'] for r in list_parameter_configs(args.architecture,path=args.model_db)
                                    if r['checkpoint_id']==checkpoint_id)
                writer.add_scalar('selection/best_parameter_id',best_parameter,epoch)
                writer.add_scalar('selection/best_validation_ce',metrics['loss'],epoch)
            row = {'epoch':epoch,'train_loss':train_loss,'validation_loss':metrics['loss'],
                   'validation_top1':metrics['top1'],'validation_top3':metrics['top3'],'checkpoint_id':checkpoint_id}
            exists = history.exists()
            with history.open('a',encoding='utf-8',newline='') as stream:
                csv_writer = csv.DictWriter(stream,fieldnames=list(row))
                if not exists: csv_writer.writeheader()
                csv_writer.writerow(row)
            for tag,value in [('loss/train_epoch',train_loss),('loss/validation',metrics['loss']),
                              ('accuracy/validation_top1',metrics['top1']),('accuracy/validation_top3',metrics['top3']),
                              ('book_ce/train',train_loss),('book_ce/validation',metrics['loss']),
                              ('validation/loss',metrics['loss']),('validation/top1',metrics['top1']),('validation/top3',metrics['top3'])]:
                writer.add_scalar(tag,value,epoch)
            writer.flush(); plot_history(history,model.config.architecture)
            display.write(f'Epoch {epoch}: train={train_loss:.4f}, val={metrics["loss"]:.4f}, '
                       f'top1={metrics["top1"]:.2%}, top3={metrics["top3"]:.2%}, checkpoint={checkpoint_id}')
        best_id = resolve_alias(args.model_db,run_id,'best')
        del model,optimizer
        best_model,_,_ = load_checkpoint(args.model_db,best_id,args.device)
        report = {'run_id':run_id,'checkpoint_id':best_id,'smoke_only':args.smoke,
                  'test':evaluate(best_model,test_loader,args.device,args.precision,'最终测试集',progress=display),
                  'metric_note':'棋谱动作命中率，并非对弈胜率；不使用测试集选择模型'}
        (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ('loss','top1','top3'): writer.add_scalar(f'test/{key}',report['test'][key],args.epochs)
        display.close()
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    finally:
        if not display.closed: display.close('已停止')
        writer.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA_DB)
    p.add_argument('--model-db',type=Path,help='默认按模型系列分别保存到 ModelParameters/HaiKu 或 Opus')
    p.add_argument('--architecture','--family',choices=['HaiKu','Opus','Haiku-5','Opus-5'],help='默认 HaiKu；自动加载实际轮数最多的参数，恢复/评估自动识别')
    p.add_argument('--parameter-id','--id',dest='parameter_id',help='显式指定起始参数 ID，例如 ID2；省略自动选实际轮数最多的 ID')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--T',type=int,default=120,help='每局读取的最大训练单步数，与 KV 历史窗口独立')
    p.add_argument('--context-plies',type=int,
                   help='K/V 缓存的最大历史局面数（含当前）；新训练默认 120，续训/评估默认沿用检查点')
    p.add_argument('--context-windows',type=int,nargs=5,metavar='W',help='Opus 五个独立窗口，默认 10 20 30 20 30')
    p.add_argument('--heads',type=int,nargs=5,metavar='H',help='Opus 五个核心头数，新训练默认 16 8 4 8 4')
    p.add_argument('--tbptt-steps',type=int,help='Opus 梯度截断长度，默认 16，与 T、窗口独立')
    p.add_argument('--description',default='',help='本次训练各版本的说明')
    p.add_argument('--set-default',action='store_true',help='将本次训练的 best 设为本系列默认参数，不自动发布')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=128,help='梯度更新的逻辑整局批量')
    p.add_argument('--micro-batch',type=int,default=16,help='一次前向/反向的局数，默认 16')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--clip-grad',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--precision',choices=['auto','none','bf16','fp16'],default='auto')
    p.add_argument('--no-recompute',action='store_true',help='关闭注意力重计算，显存占用更高')
    p.add_argument('--resume',help='恢复 SQLite checkpoint_id；epochs 指总训练轮数')
    p.add_argument('--evaluate',help='仅在测试集评估指定 checkpoint_id')
    p.add_argument('--run-name')
    p.add_argument('--limit-games',type=int,default=0,help='每个划分最多使用 N 局，0 为全部')
    p.add_argument('--smoke',action='store_true',help='仅流程验证，检查点不进入游戏模型列表')
    p.add_argument('--progress-interval',type=float,default=1.,help='终端进度与 GPU 信息刷新间隔秒数')
    p.add_argument('--no-progress',action='store_true',help='关闭动态进度，保留逐微批训练日志')
    args = p.parse_args(argv)
    if args.parameter_id is not None and (args.resume or args.evaluate): p.error('--id 与 --resume/--evaluate 分别使用')
    if min(args.T,args.epochs,args.batch_size,args.micro_batch,args.cpu_threads)<=0 or args.workers<0 or args.limit_games<0:
        p.error('批量、T、epochs、threads 必须为正，workers/limit-games 非负')
    if args.context_plies is not None and args.context_plies <= 0:
        p.error('--context-plies 必须为正数')
    if args.tbptt_steps is not None and args.tbptt_steps <= 0: p.error('--tbptt-steps 必须为正数')
    if any(v is not None and min(v)<=0 for v in (args.context_windows,args.heads)):
        p.error('窗口和头数必须为正数')
    if args.batch_size % args.micro_batch: p.error('batch-size 必须能被 micro-batch 整除')
    if not math.isfinite(args.progress_interval) or args.progress_interval<=0: p.error('progress-interval 必须为正有限数')
    if min(args.lr,args.clip_grad)<=0 or args.weight_decay<0: p.error('无效优化器参数')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.','..')): p.error('run-name 只能是目录名')
    run(args)


if __name__ == '__main__': main()

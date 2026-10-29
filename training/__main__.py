"""CLI for the same mixed-loss interface used by both model families."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import sqlite3
import torch
from torch.utils.data import ConcatDataset
from data_pipeline.loader import BOOK_CROP_VERSION,BOOK_READER_VERSION,BOOK_SOURCE_LABELS
from model_store import load_version,load_parameter_model,save_checkpoint,list_versions,get_latest_parameter_id
from project_paths import DATA_DB,ENDGAME_DB,ROOT,selfplay_db,family_model_db
from .data import BookDataset,SelfPlayDataset,collate_examples
from .engine import train_logical_batch
from .loss import LossWeights
from .control import StopController,WorkInterrupted,check_stop
from .progress import ConsoleProgress
from .curves import TrainingCurves,aggregate_reports
from .evaluation import evaluate_books,validate_evaluation,PERPLEXITY_VERSION
from .perplexity import save_perplexity_report


def main(argv=None,*,cancel_event=None):
    p=argparse.ArgumentParser(description='古谱/近期自弈混合训练，共用损失，明确指定起始参数版本')
    p.add_argument('--family',choices=['HaiKu','Opus'],required=True)
    selector=p.add_mutually_exclusive_group()
    selector.add_argument('--version',help='版本名称或完整 checkpoint_id；省略选择器时自动加载实际训练轮数最多的 ID')
    selector.add_argument('--parameter-id','--id',dest='parameter_id',help='例如 ID2 或 2，与 --family 一起定位')
    p.add_argument('--source',choices=['books','selfplay','mixed'],default='selfplay')
    p.add_argument('--model-db',type=Path)
    p.add_argument('--book-db',type=Path,default=DATA_DB)
    p.add_argument('--endgame-db',type=Path,default=ENDGAME_DB)
    p.add_argument('--book-source',choices=BOOK_SOURCE_LABELS,default='main',
                   help='棋谱读取方式：main 主棋谱库、endgames 残局库、both 两库合读并去重；训练/验证共用')
    p.add_argument('--selfplay-db',type=Path)
    p.add_argument('--rollout-run',help='本轮冻结快照生成的 run_id；禁止自动合并历史版本')
    p.add_argument('--T',type=int,default=120,help='所选棋谱来源训练裁剪长度；按 T 单步连续分段，尾段补零')
    p.add_argument('--context-plies',type=int,help='HaiKu KV 窗口；古谱训练默认 30，自弈/混合默认沿用采样检查点')
    p.add_argument('--validation-length','--L',dest='validation_length',type=int,default=32,help='验证独立切片长度 L，默认 32')
    p.add_argument('--ngram','--n',dest='ngram',type=int,default=16,help='困惑度评分最多使用 n 个输入局面，1 <= n <= L，默认 16')
    p.add_argument('--C',type=int,default=128,help='仅自弈：按 C 个单步连续分段，尾段补零；独立于古谱 T')
    p.add_argument('--book-limit',type=int,default=0,help='所选方式去重后的训练原局总上限，再保留全部片段；0 为全部')
    p.add_argument('--validation',choices=['auto','books','none'],default='auto',
                   help='auto 在 books/mixed 来源有古谱验证集时启用；books 要求非空；none 关闭')
    p.add_argument('--validation-limit',type=int,default=0,help='同一棋谱来源方式的验证原局总上限，保留全部片段；0 为全部')
    p.add_argument('--max-selfplay-plies',type=int,help='可选：每局先限制读取的新生成步数，再按 C 分段；默认读取全部')
    p.add_argument('--lambda-policy','--lambda-book',dest='lambda_book',type=float,help='古谱下一手策略损失系数；--lambda-book 为兼容别名')
    for name,label in (('value','价值'),('ppo','PPO'),('entropy','动作熵')):
        p.add_argument(f'--lambda-{name}',type=float,help=f'{label}损失系数；0 关闭该项及其曲线')
    p.add_argument('--ppo-clip',type=float,default=.2)
    p.add_argument('--tbptt-steps',type=int,default=16)
    p.add_argument('--epochs',type=int,default=1,help='本次追加训练轮数；只记录实际完整结束的轮')
    p.add_argument('--batch-size',type=int,default=128,help='一次更新的逻辑样本数；古谱、自弈均按片段计数')
    p.add_argument('--micro-batch',type=int,default=16,help='每次 GPU 处理样本数，默认 16')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,help='残差分支丢弃概率 [0,1]；省略沿用检查点，旧检查点默认 0.1')
    p.add_argument('--clip-grad',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--precision',choices=['none','bf16','fp16'],default='none')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'mixed')
    p.add_argument('--run-name')
    p.add_argument('--description',default='')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--set-default',action='store_true',help='有古谱验证集时只将本次 best 设为默认；否则每完整轮设置')
    p.add_argument('--progress-interval',type=float,default=1.,help='终端进度与 GPU 信息刷新间隔秒数')
    p.add_argument('--no-progress',action='store_true',help='关闭动态进度展示，保留结构化日志')
    args=p.parse_args(argv)
    stop=cancel_event if cancel_event is not None else StopController()
    if cancel_event is None: stop.__enter__()
    checkpoints=[]; base_completed=0; last_saved_step=0; out=None; run_id=None; db=None; display=None; curves=None
    best_record=None; best_loss=float('inf')
    try:
        validate_evaluation(args.validation_length,args.ngram)
        if min(args.T,args.C,args.epochs,args.batch_size,args.micro_batch,args.tbptt_steps,args.cpu_threads)<=0:
            raise ValueError('T、C、轮数、批量、分段和线程必须为正')
        if args.batch_size%args.micro_batch: raise ValueError('batch-size 必须能被 micro-batch 整除')
        if min(args.book_limit,args.validation_limit)<0 or args.lr<=0 or args.weight_decay<0: raise ValueError('无效数据或优化器参数')
        if args.context_plies is not None and args.context_plies<=0: raise ValueError('context-plies 必须为正整数')
        if args.family!='HaiKu' and args.context_plies is not None: raise ValueError('--context-plies 仅适用于 HaiKu，Opus 使用五个独立窗口')
        if not math.isfinite(args.progress_interval) or args.progress_interval<=0: raise ValueError('progress-interval 必须为正有限数')
        if args.run_name and (Path(args.run_name).name!=args.run_name or args.run_name in ('.','..')):
            raise ValueError('run-name 只能是目录名')
        torch.manual_seed(args.seed); torch.set_num_threads(args.cpu_threads)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        db=args.model_db or family_model_db(args.family)
        check_stop(stop)
        if args.version is not None:
            model,base,_=load_version(args.family,args.version,path=db,device=device)
        else:
            ident=args.parameter_id if args.parameter_id is not None else get_latest_parameter_id(args.family,path=db)
            model,base,_=load_parameter_model(args.family,ident,path=db,device=device)
        base_completed=base.get('completed_epochs',base['epoch']); last_saved_step=base['global_step']
        if args.dropout is not None: model.set_dropout(args.dropout)
        args.dropout=model.config.dropout
        if args.family=='HaiKu':
            if args.context_plies is None and args.source=='books':
                from models.HaiKu.config import Haiku5Config
                args.context_plies=Haiku5Config().context_plies
            if args.context_plies is not None: model.set_context_plies(args.context_plies)
            args.context_plies=model.config.context_plies
        defaults=(LossWeights.books(hasattr(model,'value_head')) if args.source=='books' else
                  LossWeights.selfplay(hasattr(model,'value_head')) if args.source=='selfplay' else
                  LossWeights.mixed(hasattr(model,'value_head')))
        weights=LossWeights(**{k:(getattr(args,f'lambda_{k}') if getattr(args,f'lambda_{k}') is not None else v)
                               for k,v in asdict(defaults).items()})
        if weights.value and not hasattr(model,'value_head'):
            raise ValueError('当前模型没有价值头；请将 --lambda-value 设为 0')
        if args.family=='Opus':
            from dataclasses import replace
            model.config=replace(model.config,tbptt_steps=args.tbptt_steps)
        datasets=[]; books=None; book_games=0; book_segments=0; selfplay_games=0; selfplay_segments=0
        book_options={'source':args.book_source,'endgame_db':args.endgame_db}
        if args.source in ('books','mixed'):
            books=BookDataset(args.T,'train',args.book_db,args.book_limit,**book_options)
            book_games=len(books.ids); book_segments=len(books)
            datasets.append(books)
        if args.source in ('selfplay','mixed'):
            dataset=SelfPlayDataset(args.selfplay_db or selfplay_db(args.family),args.rollout_run,
                                    family=args.family,max_generated_plies=args.max_selfplay_plies,C=args.C)
            if dataset.run['checkpoint_id']!=base['checkpoint_id'] or dataset.run['weights_sha256']!=base['sha256']:
                raise ValueError('采样版本与起始训练版本不一致；请使用同一冻结快照生成的近期数据')
            if args.family=='HaiKu' and weights.ppo:
                sampled=json.loads(dataset.run['model_config_json'])
                if sampled['context_plies']!=model.config.context_plies:
                    raise ValueError('PPO 的 KV 窗口必须与采样检查点一致；请沿用采样窗口，或先用新窗口古谱训练并重新生成自弈数据')
            selfplay_games=len(dataset.ids); selfplay_segments=len(dataset)
            datasets.append(dataset)
        if not all(map(len,datasets)): raise ValueError('所选来源没有可训练棋局')
        dataset=ConcatDataset(datasets)
        validation_set=None
        if args.validation=='books' or (args.validation=='auto' and args.source in ('books','mixed')):
            validation_set=BookDataset(args.validation_length,'validation',args.book_db,args.validation_limit,**book_options)
            if not len(validation_set):
                if args.validation=='books': raise ValueError('古谱验证集为空，无法评优；请准备独立 validation 划分')
                validation_set=None
                print('古谱 validation 划分为空：记录实时曲线，跳过验证评优；不会用训练损失挑选 best。',flush=True)
        run_id=args.run_name or datetime.now(timezone.utc).strftime(f'{args.family.lower()}-mixed-%Y%m%d-%H%M%S-%f')
        out=args.output/run_id
        if out.exists(): raise ValueError('训练输出目录已存在，请指定新的 run-name')
        out.mkdir(parents=True)
        config={**vars(args),'weights':asdict(weights),'base_checkpoint_id':base['checkpoint_id'],
                'evaluation_version':PERPLEXITY_VERSION,'selection_metric':'mean_slice_perplexity',
                'training_samples':len(dataset),'book_games':book_games,'book_segments':book_segments,
                'book_crop_version':BOOK_CROP_VERSION,
                'book_reader_version':BOOK_READER_VERSION,
                'book_selection':books.summary if books is not None else None,
                'validation_selection':validation_set.summary if validation_set is not None else None,
                'selfplay_games':selfplay_games,'selfplay_segments':selfplay_segments,
                'validation_games':len(validation_set.ids) if validation_set is not None else 0,
                'validation_segments':len(validation_set) if validation_set is not None else 0,
                'base_version':base['version_name'],'base_sha256':base['sha256'],'model':model.config.to_dict()}
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        curves=TrainingCurves(out,config)
        if books is not None:
            print(f'棋谱读取方式：{books.summary["label"]}（--book-source {args.book_source}）',flush=True)
            for kind,counts in books.summary['by_source'].items():
                print(f'  {BOOK_SOURCE_LABELS[kind]}：{counts["games"]:,} 局 → {counts["segments"]:,} 段；'
                      f'路径：{books.summary["databases"][kind]}',flush=True)
            if args.book_source=='both':
                print(f'  本次所选棋局跨库去重：{books.summary["duplicate_games"]:,} 局；'
                      f'跨库胜负冲突停用价值标签：{books.summary["value_conflicts"]:,} 局',flush=True)
        else:
            print('棋谱读取方式：未读取棋谱（仅自弈）',flush=True)
        label=BOOK_SOURCE_LABELS[args.book_source]
        print(f'本次训练总局数（切段后）：{len(dataset):,}；原始棋局数：{book_games+selfplay_games:,}\n'
              f'棋谱（{label}）：{book_games:,} 局 → {book_segments:,} 段（T={args.T}）；'
              f'自弈：{selfplay_games:,} 局 → {selfplay_segments:,} 段（C={args.C}）',flush=True)
        print(f'Dropout p={args.dropout:g}；每核心 10 层，共 {model.config.cores*10} 层；'
              '监督训练启用，含有效 PPO 损失的逻辑批量关闭，验证/对弈关闭。',flush=True)
        print(f'K/V 缓存窗口（含当前局面）：{model.config.context_plies}；每个新微批量重建空缓存，分段反传只截断梯度。',flush=True)
        if validation_set is not None:
            print(f'本次验证总局数（切段后）：{len(validation_set):,}；'
                  f'原始棋谱（{validation_set.summary["label"]}）：{len(validation_set.ids):,} 局（L={args.validation_length}，n={args.ngram}）',flush=True)
        print(f'实时曲线：python -m training.dashboard --logdir "{args.output.resolve()}" --port 6006\n'
              '中文曲线窗口 http://127.0.0.1:6006/；AutoDL 使用端口映射或 SSH 转发。',flush=True)
        print(json.dumps({'training_version':base['version_name'],'checkpoint_id':base['checkpoint_id'],
                          'parameter_id':base['parameter_id'],'completed_epochs':base_completed,
                          'planned_additional_epochs':args.epochs,'T':args.T,'C':args.C,'micro_batch':args.micro_batch,
                          'book_games':book_games,'book_segments':book_segments,
                          'book_source':args.book_source,'book_selection':books.summary if books is not None else None,
                          'selfplay_games':selfplay_games,'selfplay_segments':selfplay_segments,
                          'dropout':args.dropout,'weights':asdict(weights),'games':len(dataset)},ensure_ascii=False),flush=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
        global_step=base['global_step']
        scope=(label+('＋自弈' if args.source=='mixed' else '')) if books is not None else '自弈'
        display=ConsoleProgress(f'{args.family} 训练（{scope}）',args.epochs*math.ceil(len(dataset)/args.micro_batch),
                                device=device,interval=args.progress_interval,enabled=not args.no_progress).__enter__()
        with (out/'metrics.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,args.epochs+1):
                check_stop(stop)
                indices=list(range(len(dataset))); random.Random(args.seed+epoch).shuffle(indices)
                reports=[]
                for start in range(0,len(indices),args.batch_size):
                    check_stop(stop)
                    display.update(phase=f'轮 {epoch}/{args.epochs} 读取数据')
                    group=indices[start:start+args.batch_size]
                    batches=[collate_examples([dataset[i] for i in group[m:m+args.micro_batch]])
                             for m in range(0,len(group),args.micro_batch)]
                    def on_progress(event):
                        terms=event['terms']
                        display.update(event['advance'],work=event['steps'],games=event['games'],
                            phase=f'轮 {epoch}/{args.epochs} 微批 {event["micro_index"]}/{event["micro_total"]}',
                            **{'总损失':event['loss'],**{
                                label:(-1 if key=='entropy' else 1)*getattr(weights,key)*terms[key]
                                for key,label in (('book','策略损失'),('value','价值损失'),('ppo','PPO损失'),('entropy','动作熵损失'))
                                if getattr(weights,key)}})
                        curves.segment(event,epoch=epoch,global_step=global_step,progress=display.snapshot())
                    report=train_logical_batch(model,optimizer,batches,weights,chunk_size=args.tbptt_steps,
                        clip=args.ppo_clip,max_grad_norm=args.clip_grad,precision=args.precision,scaler=scaler,
                        behavior_checkpoint_id=base['checkpoint_id'],behavior_sha256=base['sha256'],cancel_event=stop,progress=on_progress)
                    global_step+=int(report['updated']); report.update(epoch=epoch,global_step=global_step)
                    report['progress']=display.snapshot()
                    curves.logical_batch(report)
                    reports.append(report); log.write(json.dumps(report,ensure_ascii=False)+'\n'); log.flush()
                    if args.no_progress: print(json.dumps(report,ensure_ascii=False),flush=True)
                if not any(r['updated'] for r in reports): raise ValueError('本轮没有可计算的有效损失，未保存训练版本')
                epoch_metrics=aggregate_reports(reports,weights)
                validation=None; validation_interrupted=False
                if validation_set is not None:
                    try:
                        validation=evaluate_books(model,validation_set,micro_batch=args.micro_batch,
                            ngram=args.ngram,chunk_size=args.tbptt_steps,precision=args.precision,cancel_event=stop,
                            progress=lambda e:display.update(phase=f'轮 {epoch}/{args.epochs} 验证微批 {e["micro_index"]}/{e["micro_total"]}'))
                    except (WorkInterrupted,KeyboardInterrupt):
                        # The training epoch is already complete; preserve it without an incomplete score.
                        validation_interrupted=True
                if validation is not None:
                    validation=save_perplexity_report(validation,out/'validation'/f'epoch-{epoch:04d}',
                        title=f'{args.family} / 第 {epoch} 轮验证集困惑度',latest=out/'validation'/'latest.json')
                    display.write(f'验证：{validation["source_games"]:,} 原局 → {validation["segments"]:,} 切片局；'
                                  f'平均困惑度={validation["perplexity"]:.4f}；分布图：{validation["artifacts"]["png"]}')
                is_best=validation is not None and validation['perplexity']<best_loss
                curves.epoch(epoch,epoch_metrics,validation)
                display.update(phase=f'保存第 {epoch} 轮参数')
                checkpoint=save_checkpoint(model,path=db,run_id=run_id,epoch=epoch,global_step=global_step,
                    metrics={'last_batch':reports[-1],'logical_batches':len(reports),'train':epoch_metrics,
                             'validation':validation,'validation_interrupted':validation_interrupted},optimizer=optimizer,
                    stage='mixed' if args.source=='mixed' else 'ppo' if args.source=='selfplay' else 'pretrain',
                    training_state={'behavior_checkpoint_id':base['checkpoint_id'],'behavior_sha256':base['sha256'],
                                    'rollout_run':args.rollout_run,'loss_weights':asdict(weights),
                                    'tbptt_steps':args.tbptt_steps,'T':args.T,'C':args.C,'context_plies':model.config.context_plies,
                                    'validation_length':args.validation_length,'ngram':args.ngram,
                                    'evaluation_version':PERPLEXITY_VERSION,'selection_metric':'mean_slice_perplexity',
                                    'book_crop_version':BOOK_CROP_VERSION,'book_reader_version':BOOK_READER_VERSION,
                                    'book_source':args.book_source,'book_selection':books.summary if books is not None else None,
                                    'rng_state':torch.get_rng_state()},
                    published=not args.smoke,best=is_best,
                    set_default=args.set_default and (is_best if validation_set is not None else True),
                    completed_epochs=base_completed+epoch,
                    parent_checkpoint_id=checkpoints[-1] if checkpoints else base['checkpoint_id'],
                    description=f'{args.description or args.source} / 累计实际训练 {base_completed+epoch} 轮 / '
                                f'本次第 {epoch} 轮 / 起始 {args.family}/ID{base["parameter_id"]}')
                checkpoints.append(checkpoint)
                last_saved_step=global_step
                metadata=next(r for r in list_versions(db) if r['checkpoint_id']==checkpoint)
                if is_best:
                    best_loss=validation['perplexity']
                    best_record={'checkpoint_id':checkpoint,'parameter_id':metadata['parameter_id'],
                        'epoch_in_run':epoch,'completed_epochs':base_completed+epoch,
                        'metric':'mean_slice_perplexity','validation':validation}
                    curves.best(epoch,metadata['parameter_id'],best_loss)
                    (out/'best.json').write_text(json.dumps(best_record,ensure_ascii=False,indent=2),encoding='utf-8')
                saved={'saved_checkpoint':checkpoint,'parameter_id':metadata['parameter_id'],
                       'completed_epochs':base_completed+epoch,'completed_epochs_this_run':epoch,
                       'validation':validation,'is_best':is_best,'best_parameter_id':best_record['parameter_id'] if best_record else None}
                display.write(json.dumps(saved,ensure_ascii=False))
                (out/'last_completed.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
                if validation_interrupted: raise WorkInterrupted('训练轮已完整保存，验证中断，未参与本轮评优')
                check_stop(stop)
        report={'run_id':run_id,'checkpoints':checkpoints,'global_step':global_step,'model_db':str(db),
                'completed_epochs':base_completed+len(checkpoints),'completed_epochs_this_run':len(checkpoints),'interrupted':False,
                'best':best_record,'tensorboard_dir':str(curves.directory),
                'note':'训练损失与自弈结果不等于棋力提升；本入口没有使用固定对手评估选优。'}
        display.close()
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    except (WorkInterrupted,KeyboardInterrupt):
        if display is not None: display.close('已中断')
        report={'run_id':run_id,'checkpoints':checkpoints,'global_step':last_saved_step,
                'completed_epochs':base_completed+len(checkpoints),'completed_epochs_this_run':len(checkpoints),
                'model_db':str(db) if db else None,'interrupted':True,
                'best':best_record,'tensorboard_dir':str(curves.directory) if curves else None,
                'note':'仅保留完整轮检查点；本轮未完成的参数和梯度均未写入参数库。'}
        if out is not None: (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True); return report
    except (ValueError,OSError,sqlite3.Error) as error: p.exit(1,f'{error}\n')
    finally:
        if curves is not None: curves.close()
        if display is not None and not display.closed: display.close('已停止')
        if cancel_event is None: stop.__exit__(None,None,None)


if __name__=='__main__': main()

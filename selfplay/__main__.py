import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3
import sys


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='count':
        from .stats import main as count_main
        return count_main(argv[1:])
    p=argparse.ArgumentParser(description='指定模型与明确参数版本，冻结快照生成自弈棋局')
    p.add_argument('--family',choices=['HaiKu','Opus'],required=True)
    selector=p.add_mutually_exclusive_group()
    selector.add_argument('--version',help='确切版本名称或 checkpoint_id')
    selector.add_argument('--parameter-id','--id',dest='parameter_id',help='ID1 或 1；省略选择器时解析并冻结默认 ID')
    selector.add_argument('--latest',action='store_true',help='自动冻结实际训练轮数最多的 ID，方便与下一轮 PPO 衔接')
    p.add_argument('--model-db',type=Path)
    p.add_argument('--data-db',type=Path,help='默认 Data/SelfPlay/模型系列/games.sqlite3，所有版本共用')
    p.add_argument('--games',type=int,default=1)
    p.add_argument('--save-every','--K',type=int,default=16,help='每 K 局完整对局原子提交；默认 16，中断丢弃当前缓冲')
    p.add_argument('--micro-batch',type=int,help='GPU 同时生成的局数：2 的幂且整除 K；默认选择不超过 16 的最大可用值')
    p.add_argument('--max-plies',type=int,default=1000)
    p.add_argument('--quiet-plies',type=int,default=120)
    p.add_argument('--repetition',type=int,default=3,help='相同循环连续重复的局面次数；默认3，下一走方须变招')
    p.add_argument('--temperature',type=float,default=1.2)
    p.add_argument('--epsilon',type=float,default=.02)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--start-fen',help='可选初始 FEN，前缀从此局面开始重放')
    p.add_argument('--prefix-json',type=Path,help='有序动作 ID 的 JSON 数组；仅暖启动不作 PPO 标签')
    p.add_argument('--progress-interval',type=float,default=1.,help='终端进度与 GPU 信息刷新间隔秒数')
    p.add_argument('--no-progress',action='store_true',help='关闭动态进度展示，保留逐局结构化日志')
    args=p.parse_args(argv)
    import torch
    from .generator import SelfPlayGenerator,SelfPlayRules,generation_micro_batch
    from training.exploration import Exploration
    from training.control import StopController
    from training.progress import ConsoleProgress
    from model_store import get_default_parameter_id,get_latest_parameter_id,parameter_number
    try:
        if args.cpu_threads<1: raise ValueError('cpu-threads 必须为正')
        if not math.isfinite(args.progress_interval) or args.progress_interval<=0: raise ValueError('progress-interval 必须为正有限数')
        micro_batch=generation_micro_batch(args.save_every,args.micro_batch)
        torch.set_num_threads(args.cpu_threads)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        version=args.version
        if version is None:
            number=(get_latest_parameter_id(args.family,path=args.model_db) if args.latest else
                    parameter_number(args.parameter_id) if args.parameter_id is not None else
                    get_default_parameter_id(args.family,path=args.model_db))
            version=f'ID{number}'
        generator=SelfPlayGenerator.from_version(args.family,version,model_db=args.model_db,
            exploration=Exploration(args.temperature,args.epsilon),device=device,seed=args.seed,
            rules=SelfPlayRules(args.max_plies,args.quiet_plies,args.repetition))
        print(json.dumps({'sampling_snapshot':asdict(generator.snapshot),'value_baseline':
                          'model_wdl' if generator.has_value_head else 'zero_no_value_head',
                          'device':device,'micro_batch':micro_batch,'save_every':args.save_every},ensure_ascii=False),flush=True)
        kwargs={'path':args.data_db,'save_every':args.save_every,'micro_batch':micro_batch}
        if args.start_fen: kwargs['start_fen']=args.start_fen
        if args.prefix_json: kwargs['prefix']=json.loads(args.prefix_json.read_text(encoding='utf-8'))
        with StopController() as stop, ConsoleProgress(f'{args.family} 自弈',args.games,device=device,unit='局',
                interval=args.progress_interval,enabled=not args.no_progress,
                fallback_work_total=args.games*args.max_plies) as display:
            display.update(已保存=0,缓冲=0,微批上限=micro_batch,并行局数=0)
            def on_game(row):
                display.update(1,games=1,phase='批量生成',已保存=row['saved_games'],缓冲=row['buffered_games'],微批上限=micro_batch)
                if args.no_progress: print(json.dumps(row,ensure_ascii=False),flush=True)
            def on_step(row):
                display.update(work=row['steps'],batches=row['batches'],phase=row['phase'],并行局数=row['active'])
            report=generator.generate(args.games,cancel_event=stop,progress=on_game,on_step=on_step,**kwargs)
            display.update(phase='结束',已保存=report['saved_games'],缓冲=0,并行局数=0)
            display.close('已中断' if report['interrupted'] else '完成')
            report['progress']=display.snapshot()
        report['model']=asdict(report['model'])
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    except (ValueError,OSError,sqlite3.Error) as error: p.exit(1,f'{error}\n')


if __name__=='__main__': main()

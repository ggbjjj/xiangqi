"""Read-only parameter evaluation: python -m training.evaluate --family Opus."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import torch

from data_pipeline.loader import BOOK_SOURCE_LABELS
from model_store import get_latest_parameter_id,load_parameter_model
from project_paths import DATA_DB,ENDGAME_DB,ROOT
from .control import StopController
from .data import BookDataset
from .evaluation import evaluate_books,validate_evaluation
from .perplexity import save_perplexity_report
from .progress import ConsoleProgress


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('HaiKu','Opus'),required=True)
    parser.add_argument('--parameter-id','--id',dest='parameter_id')
    parser.add_argument('--model-db',type=Path)
    parser.add_argument('--book-source',choices=BOOK_SOURCE_LABELS,default='main')
    parser.add_argument('--book-db',type=Path,default=DATA_DB)
    parser.add_argument('--endgame-db',type=Path,default=ENDGAME_DB)
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    parser.add_argument('--validation-length','--L',dest='validation_length',type=int,default=32)
    parser.add_argument('--ngram','--n',dest='ngram',type=int,default=16)
    parser.add_argument('--limit-games',type=int,default=0,help='原局上限，0 表示全部；每局保留所有切片')
    parser.add_argument('--micro-batch',type=int,default=16)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--precision',choices=('none','bf16','fp16'),default='none')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-progress',action='store_true')
    args=parser.parse_args(argv)
    try:
        validate_evaluation(args.validation_length,args.ngram)
        if min(args.cpu_threads,args.micro_batch)<1 or args.limit_games<0: raise ValueError('线程、微批量须为正；原局上限不能为负')
        torch.set_num_threads(args.cpu_threads)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        ident=args.parameter_id if args.parameter_id is not None else get_latest_parameter_id(args.family,path=args.model_db)
        model,metadata,_=load_parameter_model(args.family,ident,path=args.model_db,device=device)
        dataset=BookDataset(args.validation_length,args.split,args.book_db,args.limit_games,
                            source=args.book_source,endgame_db=args.endgame_db)
        print(f'棋谱读取方式：{dataset.summary["label"]}；划分：{args.split}\n'
              f'本次评估总局数（切段后）：{len(dataset):,}；原局：{len(dataset.ids):,}；'
              f'L={args.validation_length}，n={args.ngram}；{args.family}/ID{ident}',flush=True)
        output=args.output or ROOT/'runs'/'evaluation'/datetime.now().strftime(f'{args.family.lower()}-%Y%m%d-%H%M%S-%f')
        if output.exists(): raise ValueError('评估输出目录已存在，请指定新目录')
        with StopController() as stop,ConsoleProgress(f'{args.family} 困惑度评估',
                (len(dataset)+args.micro_batch-1)//args.micro_batch,device=device,enabled=not args.no_progress) as display:
            report=evaluate_books(model,dataset,micro_batch=args.micro_batch,ngram=args.ngram,
                precision=args.precision,cancel_event=stop,progress=lambda e:display.update(1,phase=f'验证微批 {e["micro_index"]}/{e["micro_total"]}'))
            report.update(family=args.family,parameter_id=metadata['parameter_id'],checkpoint_id=metadata['checkpoint_id'],
                          weights_sha256=metadata['sha256'],split=args.split,limit_games=args.limit_games)
            label='验证集' if args.split=='validation' else '测试集'
            suffix=f'（限定 {args.limit_games} 原局预览）' if args.limit_games else ''
            report=save_perplexity_report(report,output,title=f'{args.family} / ID{metadata["parameter_id"]} / {label}困惑度{suffix}')
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    except (ValueError,OSError,FloatingPointError) as error: parser.exit(1,f'评估失败：{error}\n')


if __name__=='__main__': main()

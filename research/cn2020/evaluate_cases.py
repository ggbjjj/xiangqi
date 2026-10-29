"""Isolated, bounded diagnostics for an unfinished CN2020 implementation."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]


def evaluate_one(case,connection):
    # This is the only entry point that opts into the research implementation.
    sys.path.insert(0,str(PROJECT))
    sys.path.insert(0,str(HERE))
    try:
        from data_pipeline.notation import from_fen
        from xiangqi_rules.cn2020 import assess_move
        board,turn=from_fen(case['fen'])
        previous=None;related=set();probe=board;assessments=[]
        for move in case['moves']:
            related.add(id(probe.grid[move[0]][move[1]]))
            probe=probe.apply_move(*move)
        kinds={'将':'check','杀':'kill','捉':'chase'}
        for index,move in enumerate(case['moves']):
            move=tuple(move);after=board.apply_move(*move)
            actual=assess_move(board,after,turn,move,previous,frozenset(related))
            assessments.append({'index':index,'notation':case['notations'][index],
                                'expected':kinds.get(case['labels'][index],'idle'),
                                'actual':actual.kind})
            previous=board,move;board=after;turn='b' if turn=='r' else 'r'
        mismatches=sum(item['expected']!=item['actual'] for item in assessments)
        connection.send({'status':'mismatch' if mismatches else 'passed',
                         'assessments':assessments,'mismatches':mismatches})
    except Exception as error:
        connection.send({'status':'error','error':f'{type(error).__name__}: {error}'})
    finally:
        connection.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='未接入正式程序的 CN2020 棋例诊断')
    parser.add_argument('--case',action='append',dest='names',help='例如 图6、参考图1；可重复指定')
    parser.add_argument('--timeout-seconds',type=float,default=10)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    if args.timeout_seconds<=0:parser.error('--timeout-seconds 必须大于0')
    cases=json.loads((HERE/'cases.json').read_text(encoding='utf-8'))
    if args.names:
        missing=set(args.names)-{case['case'] for case in cases}
        if missing:parser.error('不存在的棋例：'+', '.join(sorted(missing)))
        cases=[case for case in cases if case['case'] in args.names]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    report=args.report or PROJECT/'runs/rules-cn2020'/f'{stamp}.json'
    report=report.resolve()
    report.parent.mkdir(parents=True,exist_ok=True)
    data={'status':'running','enabled_in_application':False,
          'timeout_seconds_per_case':args.timeout_seconds,'cases':[]}
    context=mp.get_context('spawn')
    labels={'passed':'类别一致','mismatch':'存在不一致','timeout':'超时，未确定','error':'错误'}
    interrupted=False
    try:
        for case in cases:
            read,write=context.Pipe(duplex=False)
            process=context.Process(target=evaluate_one,args=(case,write))
            started=time.monotonic()
            process.start();write.close()
            try:
                if read.poll(args.timeout_seconds):
                    try:result=read.recv()
                    except EOFError:result={'status':'error','error':'诊断子进程未返回结果'}
                else:result={'status':'timeout'}
            finally:
                if process.is_alive():process.terminate()
                process.join();read.close()
            result.update(case=case['case'],page=case['page'],seconds=round(time.monotonic()-started,3))
            data['cases'].append(result)
            report.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(f"{case['case']}：{labels[result['status']]}（{result['seconds']:.2f}秒）",flush=True)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        data['status']='interrupted' if interrupted else 'completed'
        data['summary']={status:sum(c['status']==status for c in data['cases']) for status in labels}
        report.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('报告：'+str(report),flush=True)
    return 130 if interrupted else int(any(c['status']!='passed' for c in data['cases']))


if __name__=='__main__':
    raise SystemExit(main())

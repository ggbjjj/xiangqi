"""Read-only training-run inventory; no torch, CUDA, or weight loading required."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

from model_store import list_versions
from project_paths import ROOT,MODEL_CATALOG,model_family


STATUS_LABELS={'complete':'已完成','interrupted':'已中断','unknown':'结束状态未知'}
VALIDATION_FIELDS=('metric','metric_version','loss','perplexity','median_perplexity',
                   'corpus_perplexity','source_games','segments','games','steps','padding_steps','L','n')


def _read_json(path,warnings):
    if not path.is_file(): return {}
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict): raise ValueError('应为 JSON 对象')
        return value
    except (OSError,ValueError) as error:
        warnings.append(f'暂时无法读取 {path}：{error}')
        return {}


def _validation(value):
    return {k:value[k] for k in VALIDATION_FIELDS if k in value} if isinstance(value,dict) else None


def _last_metric(path):
    """Read a bounded tail; ignore a partially written final JSONL record."""
    if not path.is_file(): return None
    with path.open('rb') as stream:
        stream.seek(0,2); size=stream.tell(); stream.seek(max(0,size-1024*1024))
        lines=stream.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:
            value=json.loads(line)
            if isinstance(value,dict): return value
        except ValueError: pass
    return None


def _number(value):
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value): return value
    return None


def _max_known(*values):
    known=[v for value in values if (v:=_number(value)) is not None]
    return max(known) if known else None


def training_runs(*,logdir=ROOT/'runs',model_db=MODEL_CATALOG,family=None):
    """Merge log directories and checkpoint metadata by (family, run_id).

    Old resume directories may share a run_id. Keep every log path, and use
    the most recently updated config/report pair for invocation-level fields.
    A missing end report never establishes whether the process is still alive.
    """
    family=model_family(family) if family else None
    groups={}
    def group(f,run_id):
        return groups.setdefault((f,run_id),{'run_id':run_id,'family':f,'logs':[],'parameters':[]})
    versions=list_versions(model_db)
    aliases={}
    for db in {row['model_db'] for row in versions}:
        with closing(sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True)) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='aliases'").fetchone():
                aliases.update({(str(db),r[0]):r[1] for r in conn.execute(
                    "SELECT run_id,checkpoint_id FROM aliases WHERE name='best'")})
    for row in versions:
        if row['stage']=='initial': continue
        f=model_family(row['architecture'])
        if family and f!=family: continue
        entry=group(f,row['run_id'])
        metrics=json.loads(row['metrics_json'])
        train=metrics.get('train',{})
        entry['parameters'].append({k:row.get(k) for k in (
            'parameter_id','checkpoint_id','epoch','completed_epochs','global_step','version_name',
            'parent_checkpoint_id','is_default','model_db','created_at')})
        parameter=entry['parameters'][-1]
        parameter.update(is_best=aliases.get((str(row['model_db']),row['run_id']))==row['checkpoint_id'],
                         train_loss=train.get('loss',metrics.get('train_loss')),
                         train_terms=train.get('terms'),validation=_validation(metrics.get('validation')))
    root=Path(logdir).resolve()
    paths=sorted(root.rglob('config.json')) if root.is_dir() else []
    for path in paths:
        warnings=[]; config=_read_json(path,warnings)
        # Standalone evaluations do not have a training config. Generation
        # configs are explicitly labelled and must not become training runs.
        if not config or config.get('kind')=='selfplay' or config.get('evaluate'): continue
        architecture=config.get('family') or config.get('architecture') or config.get('model',{}).get('architecture')
        if not architecture or 'epochs' not in config: continue
        try: f=model_family(architecture)
        except (ValueError,AttributeError): continue
        if family and f!=family: continue
        directory=path.parent
        report=_read_json(directory/'report.json',warnings)
        legacy=_read_json(directory/'evaluation.json',warnings)
        run_id=report.get('run_id') or legacy.get('run_id') or config.get('run_name') or directory.name
        entry=group(f,run_id)
        entry['logs'].append({'directory':str(directory),'config':config,'report':report,'legacy':legacy,
                              'last':_read_json(directory/'last_completed.json',warnings),
                              'best':_read_json(directory/'best.json',warnings),'warnings':warnings,
                              'updated':max(p.stat().st_mtime for p in directory.iterdir() if p.is_file())})
    result=[]
    for entry in groups.values():
        logs=sorted(entry.pop('logs'),key=lambda log:(log['updated'],log['directory']))
        current=logs[-1] if logs else {}
        config=current.get('config',{}); report=current.get('report',{}); last=current.get('last',{})
        parameters=entry['parameters']; parameters.sort(key=lambda p:(p['epoch'],p['created_at'],p['parameter_id'] or 0))
        latest=parameters[-1] if parameters else {}
        best=(report.get('best') or current.get('best') or {})
        saved_best=next((p for p in parameters if p['is_best']),{})
        entry.update(status=('interrupted' if report.get('interrupted') else 'complete') if report else
                     'complete' if current.get('legacy') else 'unknown',
                     log_dirs=[log['directory'] for log in logs],config=config or None,
                     config_history=[{'log_dir':log['directory'],'config':log['config']} for log in logs],
                     completed_epochs_this_run=report.get('completed_epochs_this_run',last.get('completed_epochs_this_run')),
                     completed_epochs=_max_known(report.get('completed_epochs'),last.get('completed_epochs'),latest.get('completed_epochs')),
                     last_parameter_id=latest.get('parameter_id',last.get('parameter_id')),
                     best_parameter_id=saved_best.get('parameter_id',best.get('parameter_id')),
                     best_validation=saved_best.get('validation') or _validation(best.get('validation')),
                     latest_validation=latest.get('validation') or _validation(last.get('validation')),
                     saved_checkpoints=len(parameters) if parameters else len(report.get('checkpoints',[])),
                     latest_metrics=_last_metric(Path(current['directory'])/'metrics.jsonl') if current else None,
                     warnings=[message for log in logs for message in log['warnings']])
        base=next((p for p in versions if p['checkpoint_id']==config.get('base_checkpoint_id')),None)
        entry['base_parameter_id']=base['parameter_id'] if base else config.get('parameter_id')
        if not logs: entry['warnings'].append('仅找到参数库记录：训练配置和进程结束状态未记录。')
        elif not report and not current.get('legacy'):
            entry['warnings'].append('没有结束报告：可能正在训练，也可能异常退出，不能仅据此判断进程状态。')
        if not parameters: entry['warnings'].append('指定参数库中未找到本次运行的检查点；参数 ID 按可用日志显示。')
        if len(logs)>1: entry['warnings'].append('同一 run_id 有多个日志目录，配置和本次完成轮数取最近更新的目录。')
        # These are training invocations, not an inference of wall-clock start.
        entry['updated_at']=max([p['created_at'] for p in parameters],default=None)
        result.append(entry)
    return sorted(result,key=lambda r:(r['updated_at'] or '',r['run_id'],r['family']),reverse=True)


def _text(value):
    if value is None: return '未记录'
    if isinstance(value,float): return f'{value:.6g}'
    return str(value)


def _id(value):
    if value is None: return '未记录'
    return str(value) if str(value).upper().startswith('ID') else f'ID{value}'


def print_run(entry,*,detail=False):
    print(f'{entry["run_id"]}  [{entry["family"]}]')
    print(f'  状态：{STATUS_LABELS[entry["status"]]}；本次完成轮数：{_text(entry["completed_epochs_this_run"])}；'
          f'累计完成轮数：{_text(entry["completed_epochs"])}；最新参数：{_id(entry["last_parameter_id"])}；'
          f'最佳参数：{_id(entry["best_parameter_id"])}')
    validation=entry['latest_validation'] or {}
    print(f'  最近完整验证平均困惑度：{_text(validation.get("perplexity"))}；'
          f'最佳平均困惑度：{_text((entry["best_validation"] or {}).get("perplexity"))}')
    if not detail: return
    config=entry['config'] or {}
    for directory in entry['log_dirs']: print(f'  日志目录：{directory}')
    print(f'  训练来源：{config.get("source", "books" if config else "未记录")}；'
          f'棋谱读取方式：{_text(config.get("book_source"))}；计划轮数：{_text(config.get("epochs"))}')
    if config.get('source') in ('selfplay','mixed'):
        print(f'  自弈生成 run_id：{_text(config.get("rollout_run"))}（仅此运行的已保存棋局）')
    print(f'  起始参数：{_id(entry["base_parameter_id"])}；起始检查点：{_text(config.get("base_checkpoint_id",config.get("resume")))}')
    print('  学习率：'+_text(config.get('lr'))+'；权重衰减：'+_text(config.get('weight_decay'))+
          '；逻辑批量：'+_text(config.get('batch_size'))+'；微批量：'+_text(config.get('micro_batch')))
    print(f'  T={_text(config.get("T"))}；C={_text(config.get("C"))}；'
          f'L={_text(config.get("validation_length"))}；n={_text(config.get("ngram"))}；'
          f'Dropout={_text(config.get("dropout",config.get("model",{}).get("dropout")))}；'
          f'精度={_text(config.get("precision"))}；设备={_text(config.get("device"))}')
    weights=config.get('weights') or {}
    print('  损失系数：'+' / '.join(f'{label}={_text(weights.get(key))}' for key,label in
          (('book','策略'),('value','价值'),('ppo','PPO'),('entropy','动作熵'))))
    print(f'  训练棋谱原局：{_text(config.get("book_games",config.get("train_games")))}；'
          f'棋谱片段：{_text(config.get("book_segments",config.get("train_segments")))}；'
          f'自弈原局：{_text(config.get("selfplay_games"))}；自弈片段：{_text(config.get("selfplay_segments"))}；'
          f'总训练片段：{_text(config.get("training_samples"))}')
    print(f'  验证原局：{_text(config.get("validation_games"))}；验证切片：{_text(config.get("validation_segments"))}')
    print(f'  最近验证交叉熵：{_text(validation.get("loss"))}；'
          f'验证 L={_text(validation.get("L"))} / n={_text(validation.get("n"))}')
    progress=entry['latest_metrics'] or {}
    if progress:
        print(f'  最近批次：轮 {_text(progress.get("epoch"))}；参数更新 {_text(progress.get("global_step"))}；'
              f'总损失 {_text(progress.get("loss"))}（批次进度不代表完整轮已保存）')
    print(f'  已查到检查点：{entry["saved_checkpoints"]} 个')
    for p in entry['parameters']:
        v=p['validation'] or {}
        print(f'    {_id(p["parameter_id"])} | 轮={p["epoch"]} | 累计={p["completed_epochs"]} | '
              f'更新={p["global_step"]} | 训练损失={_text(p["train_loss"])} | '
              f'验证PPL={_text(v.get("perplexity"))} | best={p["is_best"]} | {p["checkpoint_id"]}')
    for warning in entry['warnings']: print(f'  提示：{warning}')


def main(argv=None):
    parser=argparse.ArgumentParser(description='查询训练 run_id、配置、完成轮数、参数 ID 和验证结果；只读且无需 GPU')
    parser.add_argument('--run-id',help='省略时列出全部训练运行；指定后查询详情')
    parser.add_argument('--family',choices=('HaiKu','Opus'),help='仅查看一个模型系列')
    parser.add_argument('--logdir',type=Path,default=ROOT/'runs',help='日志根目录，也可直接指定某次运行目录')
    parser.add_argument('--model-db',type=Path,default=MODEL_CATALOG,help='参数库根目录或某个模型的 SQLite')
    parser.add_argument('--json',action='store_true',help='输出完整结构化信息，包含配置与检查点列表')
    args=parser.parse_args(argv)
    try:
        entries=training_runs(logdir=args.logdir,model_db=args.model_db,family=args.family)
        if args.run_id is not None:
            entries=[r for r in entries if r['run_id']==args.run_id]
            if not entries: raise ValueError('训练 run_id 不存在；先省略 --run-id 查看列表，或指定正确的 --logdir / --model-db')
            if len(entries)>1: raise ValueError('两个模型系列存在同名 run_id，请用 --family 指定系列')
        if args.json:
            print(json.dumps(entries[0] if args.run_id is not None else entries,ensure_ascii=False,indent=2))
        else:
            print(f'找到 {len(entries)} 个训练运行')
            for entry in entries: print_run(entry,detail=args.run_id is not None)
        return entries[0] if args.run_id is not None else entries
    except (ValueError,OSError,sqlite3.Error) as error: parser.exit(1,f'查询失败：{error}\n')


if __name__=='__main__': main()

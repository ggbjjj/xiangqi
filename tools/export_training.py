"""Export model parameters and selected run logs without requiring PyTorch."""
import argparse
from contextlib import closing
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile

CODE_ROOT=Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from project_paths import ROOT,model_family


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def snapshot_sqlite(source,target):
    """SQLite backup includes committed WAL pages; never copy a live main file alone."""
    source,target=Path(source).resolve(),Path(target)
    if not source.is_file(): raise ValueError(f'数据库不存在：{source}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise ValueError(f'不覆盖已有快照：{target}')
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=60)) as old:
        with closing(sqlite3.connect(target)) as new:
            old.backup(new)
            if new.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError(f'数据库快照完整性检查失败：{source}')


def code_revision():
    try:
        commit=subprocess.run(['git','-C',str(CODE_ROOT),'rev-parse','HEAD'],capture_output=True,
                              text=True,check=True,timeout=5).stdout.strip()
        status=subprocess.run(['git','-C',str(CODE_ROOT),'status','--porcelain','--untracked-files=normal'],
                              capture_output=True,text=True,check=True,timeout=5).stdout
        return {'commit':commit,'has_local_changes':bool(status)}
    except (OSError,subprocess.SubprocessError):
        return {'commit':None,'has_local_changes':None}


def export_training(family,run_dirs=(),*,root=ROOT,output=None,include_selfplay=False):
    if family not in ('HaiKu','Opus'): raise ValueError('family 必须是 HaiKu 或 Opus')
    root=Path(root).resolve(); output=Path(output or root/'exports').resolve()
    databases=[Path('ModelParameters')/family/'models.sqlite3']
    if include_selfplay: databases.append(Path('Data')/'SelfPlay'/family/'games.sqlite3')
    for relative in databases:
        if not (root/relative).is_file(): raise ValueError(f'数据库不存在：{root/relative}')
    selected=[]
    for value in run_dirs:
        path=Path(value); path=(path if path.is_absolute() else root/path).resolve()
        if not path.is_dir() or not (path/'config.json').is_file():
            raise ValueError(f'run-dir 应为包含 config.json 的具体训练运行目录：{path}')
        config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        recorded=config.get('family') or config.get('architecture') or config.get('model',{}).get('architecture')
        if recorded and model_family(recorded)!=family:
            raise ValueError(f'运行日志与导出的模型系列不一致：{path}')
        if output.is_relative_to(path): raise ValueError('导出位置不能在被导出的日志目录内')
        if any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError(f'日志包含符号链接，请先整理成独立的运行目录：{path}')
        relative=(path.relative_to(root) if path.is_relative_to(root/'runs')
                  else Path('runs')/'external'/path.name)
        if relative in [r for _,r in selected]: raise ValueError('运行目录名称冲突，请分别导出')
        selected.append((path,relative))
    output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone(timedelta(hours=8)))
    name=f'return-{family}-{stamp:%Y%m%d-%H%M%S-%f}'
    bundle=output/name; bundle.mkdir(exist_ok=False)
    for relative in databases: snapshot_sqlite(root/relative,bundle/relative)
    for source,relative in selected:
        target=bundle/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,target)
    files=[{'path':path.relative_to(bundle).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)}
           for path in sorted(bundle.rglob('*')) if path.is_file()]
    manifest={'schema_version':1,'family':family,'created_at':stamp.isoformat(timespec='seconds'),
              'code':code_revision(),'selfplay_included':include_selfplay,'files':files,
              'note':'数据库使用 SQLite backup 生成独立一致快照；导出前请先停止训练和生成，以使日志与参数对应。'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=output/(name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar: tar.add(bundle,arcname=name)
    return {'family':family,'directory':str(bundle),'archive':str(archive),
            'archive_sha256':digest(archive),'files':len(files),'model_database':str(bundle/databases[0])}


def main(argv=None):
    parser=argparse.ArgumentParser(description='训练完成后导出参数、所选运行日志和可选自弈数据；不依赖 torch')
    parser.add_argument('--family',required=True,choices=['HaiKu','Opus'])
    parser.add_argument('--run-dir',type=Path,action='append',default=[],help='本次训练的具体输出目录，可多次指定')
    parser.add_argument('--include-selfplay',action='store_true',help='同时取回该模型系列的完整自弈库')
    parser.add_argument('--output',type=Path,help='导出位置，默认 exports/')
    args=parser.parse_args(argv)
    try:
        report=export_training(args.family,args.run_dir,output=args.output,include_selfplay=args.include_selfplay)
        print(json.dumps(report,ensure_ascii=False,indent=2)); return report
    except (OSError,ValueError,sqlite3.Error,tarfile.TarError) as error: parser.exit(1,f'导出失败：{error}\n')


if __name__=='__main__': main()

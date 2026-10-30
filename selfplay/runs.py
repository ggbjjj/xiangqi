"""Query and delete individual generation runs without loading torch or models."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from project_paths import ROOT,model_family,selfplay_db
from .store import connect,validate_store


DEFAULT_LOGDIR=ROOT/'runs'/'selfplay'
STATUS_LABELS={'complete':'已完成','interrupted':'已中断','error':'出错',
               'running':'标记为生成中（进程状态未确认）','unknown':'未记录'}


class RunNotFound(ValueError):
    pass


def _json(path):
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError): return {}


def _log_records(logdir):
    root=Path(logdir)
    if not root.is_dir(): return []
    result=[]
    for path in sorted(root.rglob('config.json')):
        config=_json(path)
        if config.get('kind')!='selfplay' or not config.get('run_id'): continue
        directory=path.parent
        result.append({'directory':str(directory.resolve()),'config':config,
                       'state':_json(directory/'selfplay.json'),'report':_json(directory/'report.json')})
    return result


def _basename(value):
    return str(value or '').replace('\\','/').rstrip('/').rsplit('/',1)[-1]


def _matching_logs(logs,family,run_id):
    matches=[]
    for entry in logs:
        config=entry['config']; model=config.get('model')
        stored_family=config.get('family') or (model.get('family') if isinstance(model,dict) else None)
        if config.get('run_id')==run_id and stored_family==family: matches.append(entry)
    return matches


def _resolve_id(conn,identifier,family,logs):
    row=conn.execute('SELECT run_id FROM runs WHERE run_id=? AND family=?',(identifier,family)).fetchone()
    if row: return row[0]
    rows=[dict(r) for r in conn.execute('SELECT * FROM runs WHERE family=?',(family,))]
    # Older releases gave the log folder and the SQLite run different names.
    # Resolve only exact names backed by a database record, never UUID prefixes.
    matches={r['run_id'] for r in rows if r.get('log_directory') and
             identifier in (r['log_directory'],_basename(r['log_directory']))}
    existing={r['run_id'] for r in rows}
    for entry in logs:
        config=entry['config']; ident=config.get('run_id')
        if ident in existing and _matching_logs([entry],family,ident) and identifier in (
                entry['directory'],_basename(entry['directory'])):
            matches.add(ident)
    if len(matches)>1: raise ValueError('日志目录名对应多个自弈 run_id，请查询列表后使用完整 run_id')
    if not matches: raise RunNotFound('自弈 run_id 不存在；请用 python -m selfplay.runs 查询对应模型的完整 run_id')
    return matches.pop()


def resolve_run_id(path,identifier,*,family=None,logdir=DEFAULT_LOGDIR):
    if not identifier: raise ValueError('必须指定自弈 run_id，不能加载全部自弈棋局')
    if not Path(path).is_file(): raise ValueError('自弈棋局库不存在，请先生成对局或指定 --selfplay-db')
    with closing(connect(path,True)) as conn:
        family=validate_store(conn,family)
        # The common case needs neither log discovery nor a full games scan.
        row=conn.execute('SELECT run_id FROM runs WHERE run_id=? AND family=?',(identifier,family)).fetchone()
        return row[0] if row else _resolve_id(conn,identifier,family,_log_records(logdir))


def generation_runs(family=None,*,path=None,logdir=DEFAULT_LOGDIR,run_id=None):
    if path is not None and family is None: raise ValueError('使用 --data-db 时必须同时指定 --family')
    families=(model_family(family),) if family else ('HaiKu','Opus')
    logs=_log_records(logdir); result=[]
    for f in families:
        db=Path(path or selfplay_db(f))
        if not db.is_file(): continue
        with closing(connect(db,True)) as conn:
            conn.execute('BEGIN')
            validate_store(conn,f)
            selected=None
            if run_id is not None:
                try: selected=_resolve_id(conn,run_id,f,logs)
                except RunNotFound: continue
            where='WHERE r.family=?'+(' AND r.run_id=?' if selected else '')
            params=(f,selected) if selected else (f,)
            rows=conn.execute(f'''SELECT r.*,COUNT(g.game_id) AS actual_saved_games,
                COALESCE(SUM(g.ply_count),0) AS saved_steps,
                MIN(g.ply_count) AS min_plies,MAX(g.ply_count) AS max_plies,AVG(g.ply_count) AS mean_plies,
                SUM(CASE WHEN g.result='1-0' THEN 1 ELSE 0 END) AS red_wins,
                SUM(CASE WHEN g.result='0-1' THEN 1 ELSE 0 END) AS black_wins,
                SUM(CASE WHEN g.result='1/2-1/2' THEN 1 ELSE 0 END) AS draws,
                SUM(CASE WHEN g.result='*' THEN 1 ELSE 0 END) AS unknown_results
                FROM runs r LEFT JOIN games g ON r.run_id=g.run_id {where}
                GROUP BY r.run_id ORDER BY r.created_at DESC,r.run_id DESC''',params).fetchall()
            for row in rows:
                entry=dict(row); ident=entry['run_id']; warnings=[]
                for source,target in (('model_config_json','model_config'),('exploration_json','exploration'),
                                      ('rules_json','rules'),('prefix_json','prefix')):
                    raw=entry.pop(source,None)
                    try: entry[target]=json.loads(raw) if raw is not None else None
                    except (TypeError,ValueError):
                        entry[target]=None; warnings.append(f'{source} 无法解析')
                matched=_matching_logs(logs,f,ident)
                entry['log_dirs']=sorted({e['directory'] for e in matched}|(
                    {entry['log_directory']} if entry.get('log_directory') else set()))
                entry['data_db']=str(db.resolve()); entry['recorded_saved_games']=entry.get('saved_games')
                entry['saved_games']=entry.pop('actual_saved_games')
                entry['saved_outcomes']={k:entry.pop(k) for k in ('red_wins','black_wins','draws')}
                for log in matched:
                    state=log['report'] or log['state']
                    if state.get('run_id')!=ident: continue
                    for key in ('elapsed_seconds','generated_steps','inference_batches','start_fen','prefix'):
                        if entry.get(key) is None:
                            entry[key]=state.get(key,log['config'].get(key))
                    for key in ('completed_games','discarded_games'):
                        value=state.get(key)
                        if type(value) is int: entry[key]=max(entry.get(key) or 0,value)
                entry['completed_games']=max(entry.get('completed_games') or 0,entry['saved_games'])
                entry['terminations']={r[0]:r[1] for r in conn.execute(
                    'SELECT termination,COUNT(*) FROM games WHERE run_id=? GROUP BY termination',(ident,))}
                if entry.get('recorded_saved_games') not in (None,entry['saved_games']):
                    warnings.append('保存计数与记录不一致，局数和胜负均以实际已提交棋局为准')
                entry['warnings']=warnings
                result.append(entry)
    return sorted(result,key=lambda r:(r['created_at'],r['run_id'],r['family']),reverse=True)


def delete_run(family,run_id,*,path=None,logdir=DEFAULT_LOGDIR,dry_run=False,force_running=False):
    """Delete steps, games and their run together; never touch weights or logs."""
    family=model_family(family); db=Path(path or selfplay_db(family))
    ident=resolve_run_id(db,run_id,family=family,logdir=logdir)
    with closing(connect(db,readonly=dry_run)) as conn,conn:
        conn.execute('BEGIN' if dry_run else 'BEGIN IMMEDIATE')
        validate_store(conn,family)
        row=conn.execute('SELECT * FROM runs WHERE run_id=? AND family=?',(ident,family)).fetchone()
        if row is None: raise ValueError('自弈 run_id 不存在')
        row=dict(row)
        if row.get('status')=='running' and not force_running:
            raise ValueError('该 run 标记为生成中，拒绝删除；先停止生成器。确认进程已退出的遗留记录可加 --force-running')
        games=conn.execute('SELECT COUNT(*) FROM games WHERE run_id=?',(ident,)).fetchone()[0]
        steps=conn.execute('''SELECT COUNT(*) FROM steps s JOIN games g ON s.game_id=g.game_id
                              WHERE g.run_id=?''',(ident,)).fetchone()[0]
        outcomes={r[0]:r[1] for r in conn.execute(
            'SELECT result,COUNT(*) FROM games WHERE run_id=? GROUP BY result',(ident,))}
        result={'family':family,'run_id':ident,'data_db':str(db.resolve()),'dry_run':dry_run,
                'deleted_games':games,'deleted_steps':steps,'saved_outcomes':{
                    'red_wins':outcomes.get('1-0',0),'black_wins':outcomes.get('0-1',0),
                    'draws':outcomes.get('1/2-1/2',0)},'unknown_results':outcomes.get('*',0),
                'logs_retained':True}
        if not dry_run:
            conn.execute('DELETE FROM steps WHERE game_id IN (SELECT game_id FROM games WHERE run_id=?)',(ident,))
            conn.execute('DELETE FROM games WHERE run_id=?',(ident,))
            conn.execute('DELETE FROM runs WHERE run_id=? AND family=?',(ident,family))
        return result


def _text(value):
    return '未记录' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)


def print_run(entry,*,detail=False):
    print(f'{entry["run_id"]}  [{entry["family"]}]')
    print(f'  状态：{STATUS_LABELS.get(entry.get("status"),_text(entry.get("status")))}；'
          f'计划 {_text(entry.get("requested_games"))} 局；完成 {entry["completed_games"]} 局；'
          f'实际已保存 {entry["saved_games"]} 局；丢弃 {_text(entry.get("discarded_games"))} 局')
    counts=entry['saved_outcomes']; total=entry['saved_games']
    print('  已保存胜负：'+' / '.join(f'{label} {counts[key]} 局（{counts[key]/total*100 if total else 0:.2f}%）'
          for key,label in (('red_wins','红胜'),('black_wins','黑胜'),('draws','和棋')))+
          f'；未知结果 {entry["unknown_results"]} 局')
    print(f'  采样参数：{entry["family"]}/'+('未记录' if entry.get('parameter_id') is None else f'ID{entry["parameter_id"]}')+
          f'；累计完成轮数 {_text(entry.get("completed_epochs"))}；版本 {entry["version_name"]}')
    if not detail: return
    print(f'  棋局库：{entry["data_db"]}；创建时间：{entry["created_at"]}；结束时间：{_text(entry.get("finished_at"))}')
    print(f'  检查点：{entry["checkpoint_id"]}；权重 SHA-256：{entry["weights_sha256"]}')
    exploration=entry.get('exploration') or {}; rules=entry.get('rules') or {}
    print(f'  温度：{_text(exploration.get("temperature"))}；epsilon：{_text(exploration.get("epsilon"))}；seed：{entry["seed"]}')
    print(f'  生成步数上限：{_text(rules.get("max_plies"))}；连续未吃子上限：{_text(rules.get("quiet_plies"))}；'
          f'重复变招阈值：{_text(rules.get("repetition"))}')
    print(f'  每次保存 K={_text(entry.get("save_every"))}；并行微批量={_text(entry.get("micro_batch"))}；设备={_text(entry.get("device"))}')
    print(f'  耗时：{_text(entry.get("elapsed_seconds"))} 秒；生成总步数：{_text(entry.get("generated_steps"))}；'
          f'推理批次：{_text(entry.get("inference_batches"))}')
    print(f'  已保存步数：{entry["saved_steps"]}；最短/平均/最长：'
          f'{_text(entry["min_plies"])}/{_text(entry["mean_plies"])}/{_text(entry["max_plies"])}；'
          f'结束原因：{json.dumps(entry["terminations"],ensure_ascii=False)}')
    for directory in entry['log_dirs']: print(f'  日志目录：{directory}')
    for warning in entry['warnings']: print(f'  提示：{warning}')


def main(argv=None):
    p=argparse.ArgumentParser(description='查询自弈生成 run_id、模型配置及胜负；按 run 删除棋局，无需 GPU')
    p.add_argument('--family',choices=('HaiKu','Opus'))
    p.add_argument('--run-id',help='完整自弈 run_id；也接受能在日志中准确定位的旧目录名')
    p.add_argument('--data-db',type=Path,help='自定义自弈棋局库，须同时指定 --family')
    p.add_argument('--logdir',type=Path,default=DEFAULT_LOGDIR,help='生成日志根目录，也可指定某次运行目录')
    p.add_argument('--json',action='store_true')
    p.add_argument('--delete',action='store_true',help='原子删除指定运行及其全部棋局、逐步记录；保留历史日志')
    p.add_argument('--dry-run',action='store_true',help='与 --delete 一起预览将删除的局数，不写入数据库')
    p.add_argument('--force-running',action='store_true',help='确认生成进程已退出后，允许删除仍标记 running 的遗留记录')
    args=p.parse_args(argv)
    try:
        if args.delete:
            if not args.family or not args.run_id: raise ValueError('删除必须同时指定 --family 和 --run-id')
            result=delete_run(args.family,args.run_id,path=args.data_db,logdir=args.logdir,
                              dry_run=args.dry_run,force_running=args.force_running)
            if args.json: print(json.dumps(result,ensure_ascii=False,indent=2))
            else:
                print(f'{"预计删除" if args.dry_run else "已删除"} {result["family"]}/{result["run_id"]}：'
                      f'{result["deleted_games"]} 局，{result["deleted_steps"]} 条逐步记录。历史日志保留。')
            return result
        if args.dry_run or args.force_running: raise ValueError('--dry-run 和 --force-running 仅用于 --delete')
        entries=generation_runs(args.family,path=args.data_db,logdir=args.logdir,run_id=args.run_id)
        if args.run_id is not None:
            if not entries: raise ValueError('自弈 run_id 不存在；先省略 --run-id 查询列表')
            if len(entries)!=1: raise ValueError('多个模型系列匹配该 run_id，请指定 --family')
        result=entries[0] if args.run_id is not None else entries
        if args.json: print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            print(f'找到 {len(entries)} 个自弈生成运行')
            for entry in entries: print_run(entry,detail=args.run_id is not None)
        return result
    except (ValueError,OSError,sqlite3.Error) as error: p.exit(1,f'自弈运行操作失败：{error}\n')


if __name__=='__main__': main()

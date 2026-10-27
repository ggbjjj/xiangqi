"""Read-only counts of committed games, aggregated across parameter versions."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from project_paths import model_family,selfplay_db
from .store import connect


def count_games(family,*,path=None):
    family=model_family(family); path=Path(path or selfplay_db(family))
    if not path.exists(): return 0
    with closing(connect(path,True)) as conn:
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'metadata','runs','games','steps'}<=tables:
            raise ValueError(f'{path} 不是自弈棋局库')
        row=conn.execute("SELECT value FROM metadata WHERE key='family'").fetchone()
        if row is None or row[0]!=family: raise ValueError(f'{path} 的模型系列与 {family} 不一致')
        # Count actual committed rows, not planned games or cached run counters.
        return conn.execute('''SELECT COUNT(*) FROM games g JOIN runs r ON r.run_id=g.run_id
                               WHERE r.family=?''',(family,)).fetchone()[0]


def game_counts(family=None,*,paths=None):
    families=('HaiKu','Opus') if family is None else (model_family(family),)
    paths=paths or {}
    return [{'family':f,'games':count_games(f,path=paths.get(f)),
             'data_db':str(paths.get(f) or selfplay_db(f))} for f in families]


def main(argv=None):
    p=argparse.ArgumentParser(description='分别统计 HaiKu、Opus 已保存的自弈局数，汇总所有参数版本')
    p.add_argument('--family',choices=['HaiKu','Opus'],help='省略时分别显示两种模型')
    p.add_argument('--haiku-db',type=Path,help='HaiKu 自定义自弈库，与生成时 --data-db 路径一致')
    p.add_argument('--opus-db',type=Path,help='Opus 自定义自弈库，与生成时 --data-db 路径一致')
    p.add_argument('--json',action='store_true',help='输出 JSON，便于后台读取')
    args=p.parse_args(argv)
    try:
        rows=game_counts(args.family,paths={'HaiKu':args.haiku_db,'Opus':args.opus_db})
    except (ValueError,sqlite3.Error,OSError) as error: p.exit(1,f'{error}\n')
    if args.json: print(json.dumps(rows,ensure_ascii=False,indent=2))
    else:
        for row in rows: print(f'{row["family"]}：{row["games"]} 局')
    return rows


if __name__=='__main__': main()

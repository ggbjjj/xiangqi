from contextlib import closing
from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4
from project_paths import model_family,selfplay_db

SCHEMA="""
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 run_id TEXT PRIMARY KEY, family TEXT NOT NULL, architecture TEXT NOT NULL,
 checkpoint_id TEXT NOT NULL, version_name TEXT NOT NULL, weights_sha256 TEXT NOT NULL,
 model_config_json TEXT NOT NULL, exploration_json TEXT NOT NULL, rules_json TEXT NOT NULL,
 has_value_head INTEGER NOT NULL, seed INTEGER NOT NULL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS games(
 game_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),
 start_fen TEXT NOT NULL,prefix_json TEXT NOT NULL,moves_json TEXT NOT NULL,
 result TEXT NOT NULL CHECK(result IN ('1-0','0-1','1/2-1/2','*')),
 termination TEXT NOT NULL,final_fen TEXT NOT NULL,ply_count INTEGER NOT NULL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS steps(
 game_id TEXT NOT NULL REFERENCES games(game_id),ply INTEGER NOT NULL,
 turn TEXT NOT NULL CHECK(turn IN ('r','b')),action INTEGER NOT NULL CHECK(action>=0 AND action<8100),
 old_log_mu REAL NOT NULL,old_value REAL NOT NULL,
 temperature REAL NOT NULL,epsilon REAL NOT NULL,
 PRIMARY KEY(game_id,ply));
CREATE INDEX IF NOT EXISTS games_run ON games(run_id,game_id);
CREATE INDEX IF NOT EXISTS runs_checkpoint ON runs(checkpoint_id);
"""


@dataclass(frozen=True)
class Snapshot:
    family: str
    architecture: str
    checkpoint_id: str
    version_name: str
    weights_sha256: str
    model_config: dict
    parameter_id: int | None = None
    completed_epochs: int = 0

    @classmethod
    def from_metadata(cls,meta):
        return cls(model_family(meta['architecture']),meta['architecture'],meta['checkpoint_id'],
                   meta['version_name'],meta['sha256'],json.loads(meta['config_json']),
                   meta.get('parameter_id'),meta.get('completed_epochs',meta['epoch']))


def connect(path,readonly=False):
    path=Path(path).resolve()
    if not readonly: path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) if readonly else sqlite3.connect(path,timeout=60)
    conn.row_factory=sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_store(family,path=None):
    family=model_family(family); path=Path(path or selfplay_db(family))
    with closing(connect(path)) as conn, conn:
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and not {'metadata','runs','games','steps'}<=tables:
            raise ValueError('目标不是自弈数据 SQLite；不可写入棋谱库或参数库')
        if 'metadata' in tables:
            row=conn.execute("SELECT value FROM metadata WHERE key='family'").fetchone()
            if row and row[0]!=family: raise ValueError('自弈数据库模型系列不匹配')
        conn.executescript(SCHEMA)
        columns={r[1] for r in conn.execute('PRAGMA table_info(runs)')}
        for name,declaration in {'save_every':'INTEGER NOT NULL DEFAULT 16','requested_games':'INTEGER',
             'status':"TEXT NOT NULL DEFAULT 'complete'",'saved_games':'INTEGER NOT NULL DEFAULT 0',
             'completed_games':'INTEGER NOT NULL DEFAULT 0','discarded_games':'INTEGER NOT NULL DEFAULT 0',
             'parameter_id':'INTEGER','completed_epochs':'INTEGER NOT NULL DEFAULT 0',
             'micro_batch':'INTEGER NOT NULL DEFAULT 1','device':'TEXT'}.items():
            if name not in columns: conn.execute(f'ALTER TABLE runs ADD COLUMN {name} {declaration}')
        if 'generation_index' not in {r[1] for r in conn.execute('PRAGMA table_info(games)')}:
            conn.execute('ALTER TABLE games ADD COLUMN generation_index INTEGER')
        conn.execute("INSERT OR IGNORE INTO metadata VALUES ('family',?)",(family,))
        conn.execute("INSERT INTO metadata VALUES ('schema_version','3') ON CONFLICT(key) DO UPDATE SET value='3'")
    return path


def create_run(snapshot,exploration,rules,has_value_head,seed,path=None,*,save_every=16,requested_games=None,
               micro_batch=1,device=None):
    path=init_store(snapshot.family,path); ident=uuid4().hex
    if model_family(snapshot.architecture)!=snapshot.family: raise ValueError('快照模型系列不一致')
    with closing(connect(path)) as conn,conn:
        conn.execute('''INSERT INTO runs(run_id,family,architecture,checkpoint_id,version_name,weights_sha256,
                        model_config_json,exploration_json,rules_json,has_value_head,seed,save_every,requested_games,
                        status,parameter_id,completed_epochs,micro_batch,device) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                     (ident,snapshot.family,snapshot.architecture,snapshot.checkpoint_id,snapshot.version_name,
                      snapshot.weights_sha256,json.dumps(snapshot.model_config),json.dumps(asdict(exploration)),
                      json.dumps(asdict(rules)),int(has_value_head),seed,save_every,requested_games,'running',
                      snapshot.parameter_id,snapshot.completed_epochs,micro_batch,device))
    return path,ident


def _insert_game(conn,run_id,*,start_fen,prefix,steps,result,termination,final_fen,generation_index=None):
    if not steps: raise ValueError('没有新生成动作的局面不保存为对局')
    for step in steps:
        if not math.isfinite(step['old_log_mu']) or step['old_log_mu']>1e-5: raise ValueError('旧采样概率无效')
        if not math.isfinite(step['old_value']) or abs(step['old_value'])>1.00001: raise ValueError('旧价值无效')
        if not math.isfinite(step['temperature']) or step['temperature']<=0 or not 0<=step['epsilon']<=1:
            raise ValueError('采样配置无效')
    ident=uuid4().hex
    conn.execute('''INSERT INTO games(game_id,run_id,start_fen,prefix_json,moves_json,result,
                         termination,final_fen,ply_count,generation_index) VALUES (?,?,?,?,?,?,?,?,?,?)''',
                     (ident,run_id,start_fen,json.dumps(list(prefix)),json.dumps([s['action'] for s in steps]),
                      result,termination,final_fen,len(steps),generation_index))
    conn.executemany('INSERT INTO steps VALUES (?,?,?,?,?,?,?,?)',
                         [(ident,t,s['turn'],s['action'],s['old_log_mu'],s['old_value'],s['temperature'],s['epsilon'])
                          for t,s in enumerate(steps)])
    return ident


def save_games(path,run_id,games):
    """All games and their steps commit in one transaction or none do."""
    games=list(games)
    if not games: return []
    with closing(connect(path)) as conn,conn:
        ids=[_insert_game(conn,run_id,**game) for game in games]
        conn.execute('UPDATE runs SET saved_games=saved_games+? WHERE run_id=?',(len(ids),run_id))
    return ids


def save_game(path,run_id,**game):
    return save_games(path,run_id,[game])[0]


def finish_run(path,run_id,*,completed,saved,discarded,interrupted):
    with closing(connect(path)) as conn,conn:
        conn.execute('UPDATE runs SET status=?,completed_games=?,saved_games=?,discarded_games=? WHERE run_id=?',
                     ('interrupted' if interrupted else 'complete',completed,saved,discarded,run_id))


def committed_game_ids(path,run_id):
    with closing(connect(path,True)) as conn:
        return [r[0] for r in conn.execute('SELECT game_id FROM games WHERE run_id=? ORDER BY rowid',(run_id,))]


OUTCOME_KEYS={'1-0':'red_wins','0-1':'black_wins','1/2-1/2':'draws'}


def empty_outcomes():
    return dict.fromkeys(OUTCOME_KEYS.values(),0)


def committed_outcomes(path,run_id):
    """Count only committed games from this run, independently of cached counters."""
    counts=empty_outcomes()
    with closing(connect(path,True)) as conn:
        for row in conn.execute('SELECT result,COUNT(*) FROM games WHERE run_id=? GROUP BY result',(run_id,)):
            if row[0] in OUTCOME_KEYS: counts[OUTCOME_KEYS[row[0]]]=row[1]
    return counts


def list_runs(path):
    with closing(connect(path,True)) as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM runs ORDER BY created_at,run_id')]


def load_game(path,game_id):
    with closing(connect(path,True)) as conn:
        row=conn.execute('SELECT * FROM games WHERE game_id=?',(game_id,)).fetchone()
        if row is None: raise ValueError('自弈棋局不存在')
        game=dict(row)
        run=dict(conn.execute('SELECT * FROM runs WHERE run_id=?',(game['run_id'],)).fetchone())
        steps=[dict(r) for r in conn.execute('SELECT * FROM steps WHERE game_id=? ORDER BY ply',(game_id,))]
        return game,run,steps

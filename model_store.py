"""Transactional SQLite checkpoints. Importing this file does not import torch."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from contextlib import closing
from uuid import uuid4

from project_paths import (MODEL_DB, MODEL_CATALOG, LEGACY_MODEL_DB, HAIKU_MODEL_DB,
                           family_model_db, model_family)

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS architectures(name TEXT PRIMARY KEY,config_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(
 checkpoint_id TEXT PRIMARY KEY, architecture TEXT NOT NULL REFERENCES architectures(name),
 run_id TEXT NOT NULL, stage TEXT NOT NULL, epoch INTEGER NOT NULL, global_step INTEGER NOT NULL,
 config_json TEXT NOT NULL, weights_blob BLOB NOT NULL, training_state_blob BLOB,
 metrics_json TEXT NOT NULL, sha256 TEXT NOT NULL, published INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS aliases(run_id TEXT NOT NULL,name TEXT NOT NULL,
 checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id),PRIMARY KEY(run_id,name));
CREATE TABLE IF NOT EXISTS model_releases(
 name TEXT PRIMARY KEY, checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id),
 enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)), sort_order INTEGER NOT NULL DEFAULT 0,
 updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS model_defaults(
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id));
CREATE TABLE IF NOT EXISTS store_info(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


def _is_catalog(path):
    return Path(path).is_dir() or Path(path).resolve() == MODEL_CATALOG.resolve()


def _db_path(path=None, architecture=None):
    path = Path(path) if path is not None else family_model_db(architecture or 'HaiKu')
    if _is_catalog(path): path = path/model_family(architecture or 'HaiKu')/'models.sqlite3'
    # Old resume records continue writing to the new HaiKu store after migration.
    if path.resolve() == LEGACY_MODEL_DB.resolve() and HAIKU_MODEL_DB.exists():
        return HAIKU_MODEL_DB
    return path


def _catalog_paths(path):
    path = Path(path or MODEL_CATALOG)
    if not _is_catalog(path): return [_db_path(path)]
    paths = [path/family/'models.sqlite3' for family in ('HaiKu','Opus')]
    if not paths[0].exists() and (path/'models.sqlite3').exists(): paths[0] = path/'models.sqlite3'
    return [p for p in paths if p.exists()]


def _checkpoint_path(path, checkpoint_id=None):
    path = Path(path or MODEL_CATALOG)
    if not _is_catalog(path): return _db_path(path)
    if checkpoint_id is None: return _db_path(path)
    matches = []
    for candidate in _catalog_paths(path):
        with closing(_read_connection(candidate)) as conn:
            if conn.execute('SELECT 1 FROM checkpoints WHERE checkpoint_id=?',(checkpoint_id,)).fetchone():
                matches.append(candidate)
    if len(matches) != 1: raise ValueError('模型版本不存在或参数 ID 不唯一')
    return matches[0]


def init_store(path=None, architecture=None):
    from models.registry import ARCHITECTURES
    path = _db_path(path,architecture)
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        # Check the family before a schema migration can commit any changes.
        if architecture is not None:
            family = model_family(architecture)
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'store_info' in tables:
                row = conn.execute("SELECT value FROM store_info WHERE key='family'").fetchone()
                if row and row[0] != family: raise ValueError('参数库系列不匹配，HaiKu 与 Opus 必须分库')
            if 'checkpoints' in tables:
                existing = conn.execute('SELECT DISTINCT architecture FROM checkpoints').fetchall()
                if any(model_family(r[0]) != family for r in existing): raise ValueError('参数库包含其他模型系列')
        conn.executescript(SCHEMA)
        conn.execute('BEGIN IMMEDIATE')
        columns = {row[1] for row in conn.execute('PRAGMA table_info(checkpoints)')}
        if 'parameter_id' not in columns:
            conn.execute('ALTER TABLE checkpoints ADD COLUMN parameter_id INTEGER')
        if 'completed_epochs' not in columns:
            conn.execute('ALTER TABLE checkpoints ADD COLUMN completed_epochs INTEGER NOT NULL DEFAULT 0')
            conn.execute('UPDATE checkpoints SET completed_epochs=epoch')
        if 'parent_checkpoint_id' not in columns:
            conn.execute('ALTER TABLE checkpoints ADD COLUMN parent_checkpoint_id TEXT')
        next_id=conn.execute('SELECT COALESCE(MAX(parameter_id),0)+1 FROM checkpoints').fetchone()[0]
        for row in conn.execute('SELECT checkpoint_id FROM checkpoints WHERE parameter_id IS NULL ORDER BY created_at,epoch,checkpoint_id').fetchall():
            conn.execute('UPDATE checkpoints SET parameter_id=? WHERE checkpoint_id=?',(next_id,row[0]))
            next_id+=1
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS parameter_numbers ON checkpoints(parameter_id)')
        for name in ('version_name','description'):
            if name not in columns:
                conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        for row in conn.execute("SELECT checkpoint_id,architecture,epoch,global_step,stage FROM checkpoints WHERE version_name='' OR description=''").fetchall():
            ident,arch,epoch,step,stage = row
            conn.execute('''UPDATE checkpoints SET version_name=CASE WHEN version_name='' THEN ? ELSE version_name END,
                description=CASE WHEN description='' THEN ? ELSE description END WHERE checkpoint_id=?''',
                (f'{arch}-e{epoch}-{ident[:8]}',f'{arch} / {stage} / epoch {epoch} / step {step}',ident))
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS checkpoint_versions ON checkpoints(version_name)')
        if not conn.execute('SELECT 1 FROM model_defaults WHERE singleton=1').fetchone():
            latest=conn.execute('SELECT checkpoint_id FROM checkpoints ORDER BY completed_epochs DESC,parameter_id DESC LIMIT 1').fetchone()
            if latest: conn.execute('INSERT INTO model_defaults VALUES (1,?)',(latest[0],))
        for name,config in ARCHITECTURES.items():
            conn.execute('INSERT OR IGNORE INTO architectures VALUES (?,?)',(name,json.dumps(config().to_dict())))
        row = conn.execute("SELECT value FROM store_info WHERE key='family'").fetchone()
        if architecture is not None:
            family = model_family(architecture)
            if row and row[0] != family: raise ValueError('参数库系列不匹配，HaiKu 与 Opus 必须分库')
            existing = conn.execute('SELECT DISTINCT architecture FROM checkpoints').fetchall()
            if any(model_family(r[0]) != family for r in existing): raise ValueError('参数库包含其他模型系列')
            conn.execute("INSERT OR IGNORE INTO store_info VALUES ('family',?)",(family,))
    return path


def _read_connection(path):
    conn = sqlite3.connect(_db_path(path).resolve().as_uri()+'?mode=ro',uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_models(path=MODEL_CATALOG, *, include_unpublished=False):
    """Private training/admin checkpoint inventory; never return this to the UI."""
    if _is_catalog(path):
        return [row for p in _catalog_paths(path) for row in list_models(p,include_unpublished=include_unpublished)]
    path = _db_path(path)
    if not path.exists(): return []
    conn = _read_connection(path)
    try:
        where = '' if include_unpublished else 'WHERE published=1 AND global_step>0'
        columns = {row[1] for row in conn.execute('PRAGMA table_info(checkpoints)')}
        extra = ',version_name,description' if 'version_name' in columns else ''
        extra += ',config_json,sha256'
        extra += ',parameter_id' if 'parameter_id' in columns else ',NULL AS parameter_id'
        extra += ',completed_epochs' if 'completed_epochs' in columns else ',epoch AS completed_epochs'
        extra += ',parent_checkpoint_id' if 'parent_checkpoint_id' in columns else ',NULL AS parent_checkpoint_id'
        return [dict(row) for row in conn.execute(f'''SELECT checkpoint_id,architecture,run_id,stage,epoch,
          global_step,metrics_json,published,created_at {extra} FROM checkpoints {where}
          ORDER BY created_at DESC,checkpoint_id DESC''')]
    finally: conn.close()


def list_hint_models(path=MODEL_CATALOG):
    """Public catalog: names only. Legacy stores require explicit publication."""
    if _is_catalog(path):
        rows = [row for p in _catalog_paths(path) for row in list_hint_models(p)]
        if len({r['name'] for r in rows}) != len(rows): raise ValueError('两个参数库的公开名称重复，请修改名称')
        return rows
    path = _db_path(path)
    if not path.exists(): return []
    conn = _read_connection(path)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_releases'").fetchone():
            return []
        return [{'name':row[0]} for row in conn.execute('''SELECT r.name FROM model_releases r
          JOIN checkpoints c ON c.checkpoint_id=r.checkpoint_id
          WHERE r.enabled=1 AND c.published=1 AND c.global_step>0
          ORDER BY r.sort_order,r.name''')]
    finally: conn.close()


def publish_model(name, checkpoint_id, *, path=MODEL_CATALOG, sort_order=None):
    """Administrator selects the weights behind a public name; no automatic epoch mapping."""
    if not isinstance(name,str): raise ValueError('模型名称必须是字符串')
    name = name.strip()
    if not name or len(name)>60 or any(ord(c)<32 or ord(c)==127 for c in name):
        raise ValueError('模型名称须为 1–60 个可显示字符')
    catalog = path if _is_catalog(path) else None
    path = _checkpoint_path(path,checkpoint_id)
    if catalog is not None:
        if any(p != path and {'name':name} in list_hint_models(p) for p in _catalog_paths(catalog)):
            raise ValueError('另一个模型系列已使用此公开名称')
    init_store(path)
    with closing(sqlite3.connect(path,timeout=60)) as conn, conn:
        conn.execute('PRAGMA foreign_keys=ON')
        row = conn.execute('SELECT published,global_step FROM checkpoints WHERE checkpoint_id=?',(checkpoint_id,)).fetchone()
        if row is None: raise ValueError('检查点不存在')
        if not row[0] or row[1]<1: raise ValueError('测试或未训练的检查点不能发布到游戏')
        if sort_order is None:
            old = conn.execute('SELECT sort_order FROM model_releases WHERE name=?',(name,)).fetchone()
            sort_order = old[0] if old else conn.execute('SELECT COALESCE(MAX(sort_order),-1)+1 FROM model_releases').fetchone()[0]
        conn.execute('''INSERT INTO model_releases(name,checkpoint_id,enabled,sort_order) VALUES (?,?,1,?)
          ON CONFLICT(name) DO UPDATE SET checkpoint_id=excluded.checkpoint_id,enabled=1,
          sort_order=excluded.sort_order,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')''',
          (name,checkpoint_id,int(sort_order)))
        conn.execute('INSERT OR IGNORE INTO model_defaults VALUES (1,?)',(checkpoint_id,))


def unpublish_model(name, *, path=MODEL_CATALOG):
    path = _release_path(path,name)
    init_store(path)
    with closing(sqlite3.connect(path,timeout=60)) as conn, conn:
        if not conn.execute('UPDATE model_releases SET enabled=0 WHERE name=?',(name,)).rowcount:
            raise ValueError('模型名称不存在')


def _release_path(path, name):
    path = Path(path or MODEL_CATALOG)
    if not _is_catalog(path): return _db_path(path)
    matches = [p for p in _catalog_paths(path) if {'name':name} in list_hint_models(p)]
    if len(matches) != 1: raise ValueError('该模型暂不可用或名称不唯一')
    return matches[0]


def load_hint_model(name, *, path=MODEL_CATALOG, device='cpu'):
    """Resolve the public name inside the backend; return no checkpoint metadata."""
    path = _release_path(path,name)
    if not path.exists(): raise ValueError('该模型暂不可用')
    conn = _read_connection(path)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_releases'").fetchone():
            raise ValueError('该模型暂不可用')
        row = conn.execute('''SELECT r.checkpoint_id FROM model_releases r
          JOIN checkpoints c ON c.checkpoint_id=r.checkpoint_id
          WHERE r.name=? AND r.enabled=1 AND c.published=1 AND c.global_step>0''',(name,)).fetchone()
        if row is None: raise ValueError('该模型暂不可用')
        checkpoint_id = row[0]
    finally: conn.close()
    model,_,_ = load_checkpoint(path,checkpoint_id,device)
    model.display_name = name
    return model


def save_checkpoint(model, *, path=None, run_id, epoch, global_step, metrics,
                    optimizer=None, training_state=None, best=False, published=True, stage='pretrain',
                    version_name=None, description='', set_default=False,
                    completed_epochs=None,parent_checkpoint_id=None):
    import torch
    path = init_store(path,model.config.architecture)
    data = io.BytesIO()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},data)
    payload = data.getvalue()
    state = dict(training_state or {})
    metrics=dict(metrics)
    if hasattr(model,'value_head'):
        metrics.setdefault('value_trained',bool(getattr(model,'value_trained',False)))
    if optimizer is not None: state['optimizer'] = optimizer.state_dict()
    state_bytes = io.BytesIO()
    torch.save(state,state_bytes)
    ident = uuid4().hex
    completed_epochs=epoch if completed_epochs is None else completed_epochs
    if type(completed_epochs) is not int or completed_epochs<0: raise ValueError('实际完成轮数必须为非负整数')
    version_name = _version_text(version_name or f'{model.config.architecture}-e{completed_epochs}-{ident[:8]}')
    description = description.strip() or f'{model.config.architecture} / {stage} / 累计实际训练 {completed_epochs} 轮 / 本次轮次 {epoch} / step {global_step}'
    with closing(sqlite3.connect(path,timeout=60)) as conn, conn:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('BEGIN IMMEDIATE')
        parameter_id=conn.execute('SELECT COALESCE(MAX(parameter_id),0)+1 FROM checkpoints').fetchone()[0]
        conn.execute('''INSERT INTO checkpoints(checkpoint_id,architecture,run_id,stage,epoch,global_step,
          config_json,weights_blob,training_state_blob,metrics_json,sha256,published,version_name,description,
          parameter_id,completed_epochs,parent_checkpoint_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          (ident,model.config.architecture,run_id,stage,epoch,global_step,json.dumps(model.config.to_dict()),
           payload,state_bytes.getvalue(),json.dumps(metrics,allow_nan=False),hashlib.sha256(payload).hexdigest(),
           int(published and global_step>0),version_name,description,parameter_id,completed_epochs,parent_checkpoint_id))
        for alias in ['latest']+(['best'] if best else []):
            conn.execute('INSERT OR REPLACE INTO aliases VALUES (?,?,?)',(run_id,alias,ident))
        if set_default:
            conn.execute('INSERT OR REPLACE INTO model_defaults VALUES (1,?)',(ident,))
        else:
            conn.execute('INSERT OR IGNORE INTO model_defaults VALUES (1,?)',(ident,))
    return ident


def resolve_alias(path, run_id, alias='best'):
    if _is_catalog(path):
        matches = []
        for candidate in _catalog_paths(path):
            try: matches.append(resolve_alias(candidate,run_id,alias))
            except ValueError: pass
        if len(matches) != 1: raise ValueError(f'没有唯一的 {run_id}/{alias} 检查点')
        return matches[0]
    conn = _read_connection(path)
    try:
        row = conn.execute('SELECT checkpoint_id FROM aliases WHERE run_id=? AND name=?',(run_id,alias)).fetchone()
        if row is None: raise ValueError(f'没有 {run_id}/{alias} 检查点')
        return row[0]
    finally: conn.close()


def get_best_parameter_config(family,run_id,*,path=None):
    """Best held-out score within one training run; does not change defaults."""
    family=model_family(family); path=path or family_model_db(family)
    checkpoint=resolve_alias(path,run_id,'best')
    for row in list_parameter_configs(family,path=path):
        if row['checkpoint_id']==checkpoint: return row
    raise ValueError('本次运行的 best 不属于指定模型系列')


def _version_text(value):
    if not isinstance(value,str) or not value.strip() or len(value)>120 or any(ord(c)<32 for c in value):
        raise ValueError('版本名称须为 1–120 个可显示字符')
    return value.strip()


def default_checkpoint_id(path=MODEL_DB):
    path = _db_path(path)
    if not path.exists(): return None
    with closing(_read_connection(path)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='model_defaults'").fetchone(): return None
        row = conn.execute('SELECT checkpoint_id FROM model_defaults WHERE singleton=1').fetchone()
        return row[0] if row else None


def set_default_checkpoint(checkpoint_id, *, path=MODEL_CATALOG):
    path = _checkpoint_path(path,checkpoint_id)
    init_store(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute('PRAGMA foreign_keys=ON')
        if not conn.execute('SELECT 1 FROM checkpoints WHERE checkpoint_id=?',(checkpoint_id,)).fetchone():
            raise ValueError('模型版本不存在')
        conn.execute('INSERT OR REPLACE INTO model_defaults VALUES (1,?)',(checkpoint_id,))


def default_model_name(path=MODEL_CATALOG):
    for candidate in _catalog_paths(path):
        ident = default_checkpoint_id(candidate)
        if ident is None: continue
        with closing(_read_connection(candidate)) as conn:
            row = conn.execute('''SELECT r.name FROM model_releases r JOIN checkpoints c
                ON c.checkpoint_id=r.checkpoint_id WHERE r.checkpoint_id=? AND r.enabled=1
                AND c.published=1 AND c.global_step>0 ORDER BY r.sort_order,r.name''',(ident,)).fetchone()
            if row: return row[0]
    return None


def load_default_model(family='HaiKu', *, path=None, device='cpu'):
    """Developer/training loader, including explicitly labelled initialization weights."""
    model = load_checkpoint(_db_path(path,family),device=device)[0]
    if model_family(model.config.architecture) != model_family(family):
        raise ValueError('默认参数与请求的模型系列不一致')
    return model


def load_version(family, version, *, path=None, device='cpu', resume=False):
    """Resolve an explicit version name or immutable checkpoint ID once, including initial weights."""
    family = model_family(family)
    if isinstance(version,int) or (isinstance(version,str) and version.upper().startswith('ID') and version[2:].isdigit()):
        row=get_parameter_config(family,version,path=path)
        return load_checkpoint(row['model_db'],row['checkpoint_id'],device,resume)
    if not isinstance(version,str) or not version.strip(): raise ValueError('必须明确指定模型版本名称或参数 ID')
    path = _db_path(path,family)
    matches = [r for r in list_versions(path)
               if model_family(r['architecture'])==family and
               (r['checkpoint_id']==version or r['version_name']==version)]
    if len(matches)!=1: raise ValueError(f'{family} 中没有唯一的版本 {version}；请先 list-versions')
    return load_checkpoint(path,matches[0]['checkpoint_id'],device,resume)


def parameter_number(value):
    text=str(value).strip()
    if text.upper().startswith('ID'): text=text[2:]
    if not text.isascii() or not text.isdecimal() or int(text)<1: raise ValueError('参数 ID 应为正整数或 ID1 这样的编号')
    return int(text)


def list_parameter_configs(family=None, *, path=None):
    """Backend inventory. IDs are stable within a family, never across families."""
    family=model_family(family) if family is not None else None
    selected=_db_path(path,family) if family is not None else (path or MODEL_CATALOG)
    rows=[]
    for row in list_versions(selected):
        f=model_family(row['architecture'])
        if family is not None and f!=family: continue
        rows.append({'family':f,'parameter_id':row['parameter_id'],
                     'id':f'ID{row["parameter_id"]}' if row['parameter_id'] is not None else None,
                     'checkpoint_id':row['checkpoint_id'],'version_name':row['version_name'],
                     'completed_epochs':row['completed_epochs'],'epoch_in_run':row['epoch'],
                     'parent_checkpoint_id':row['parent_checkpoint_id'],
                     'is_default':row['is_default'],'description':row['description'],
                     'model_config':json.loads(row['config_json']),'sha256':row['sha256'],
                     'model_db':row['model_db'],'created_at':row['created_at']})
    return sorted(rows,key=lambda r:(r['family'],r['parameter_id'] or 0))


def get_parameter_config(family,parameter_id=None,*,path=None):
    rows=list_parameter_configs(family,path=path)
    matches=([r for r in rows if r['is_default']] if parameter_id is None else
             [r for r in rows if r['parameter_id']==parameter_number(parameter_id)])
    if len(matches)!=1: raise ValueError('没有唯一的默认参数或指定 ID 不存在，请先查询 list-parameters')
    return matches[0]


def get_default_parameter_id(family,*,path=None):
    return get_parameter_config(family,path=path)['parameter_id']


def get_latest_parameter_id(family,*,path=None):
    """Most actual completed epochs; equal counts choose the higher stable ID."""
    rows=[r for r in list_parameter_configs(family,path=path) if r['parameter_id'] is not None]
    if not rows: raise ValueError('没有已编号的参数，请先 migrate 或初始化该模型')
    return max(rows,key=lambda r:(r['completed_epochs'],r['parameter_id']))['parameter_id']


def set_default_parameter_id(family,parameter_id,*,path=None):
    row=get_parameter_config(family,parameter_id,path=path)
    set_default_checkpoint(row['checkpoint_id'],path=row['model_db'])


def load_parameter_model(family,parameter_id=None,*,path=None,device='cpu',resume=False):
    row=get_parameter_config(family,parameter_id,path=path)
    model,meta,state=load_checkpoint(row['model_db'],row['checkpoint_id'],device,resume)
    model.display_name=row['version_name']
    return model,meta,state


def load_value_model(*,path=MODEL_CATALOG,device='cpu'):
    """Use the configured Opus default for analysis, independent of the opponent."""
    model,meta,_=load_parameter_model('Opus',path=path,device=device)
    if not meta['published'] or not getattr(model,'value_trained',False):
        raise ValueError('暂无已训练的价值模型')
    model.display_name=default_model_name(meta.get('model_db',_db_path(path,'Opus'))) or 'Opus-5'
    return model


def describe_version(checkpoint_id, *, path=MODEL_CATALOG, version_name=None, description=None):
    path = _checkpoint_path(path,checkpoint_id)
    init_store(path)
    fields,values = [],[]
    if version_name is not None: fields.append('version_name=?'); values.append(_version_text(version_name))
    if description is not None:
        if not isinstance(description,str) or not description.strip(): raise ValueError('版本说明不能为空')
        fields.append('description=?'); values.append(description.strip())
    if not fields: raise ValueError('请提供版本名称或说明')
    with closing(sqlite3.connect(path)) as conn, conn:
        if not conn.execute('UPDATE checkpoints SET '+','.join(fields)+' WHERE checkpoint_id=?',(*values,checkpoint_id)).rowcount:
            raise ValueError('模型版本不存在')


def list_versions(path=MODEL_CATALOG):
    rows = []
    for candidate in _catalog_paths(path):
        ident = default_checkpoint_id(candidate)
        for row in list_models(candidate,include_unpublished=True):
            row['is_default'] = row['checkpoint_id']==ident
            row['model_db'] = str(candidate)
            rows.append(row)
    return rows


def migrate_legacy_store(source=LEGACY_MODEL_DB, destination=HAIKU_MODEL_DB):
    """Consistent SQLite backup; keep the original database untouched for recovery."""
    source,destination = Path(source),Path(destination)
    if destination.exists(): return False
    if not source.exists(): return False
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(f'.migration-{uuid4().hex}.sqlite3')
    try:
        with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)) as old:
            with closing(sqlite3.connect(temporary)) as new: old.backup(new)
        init_store(temporary,'Haiku-5')
        with closing(sqlite3.connect(temporary)) as conn:
            if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('迁移参数库完整性校验失败')
            if conn.execute('PRAGMA foreign_key_check').fetchall(): raise ValueError('迁移参数库外键校验失败')
        # rename refuses an existing destination on Windows; never replace a live store.
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def initialize_model(family, *, path=None, version_name=None, description=None, seed=42):
    import torch
    from models.registry import ARCHITECTURES,build_model
    family = model_family(family)
    architecture = {'HaiKu':'Haiku-5','Opus':'Opus-5'}[family]
    path = _db_path(path,architecture)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_model(ARCHITECTURES[architecture]().to_dict())
    return save_checkpoint(model,path=path,run_id=f'{family.lower()}-initial',epoch=0,global_step=0,
        metrics={'initialized_only':True,'seed':seed},published=False,stage='initial',
        version_name=version_name or f'{architecture}.initial',
        description=description or f'{architecture} 初始化参数，尚未训练；种子 {seed}',
        set_default=default_checkpoint_id(path) is None)


def main(argv=None):
    """Backend administration only; training metadata never enters the public catalog."""
    import argparse
    p = argparse.ArgumentParser(description='后台模型发布：管理公开名称对应的训练检查点')
    p.add_argument('--db',type=Path)
    p.add_argument('--family',choices=['HaiKu','Opus'],help='限定模型系列，省略时列表汇总两个参数库')
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('list-checkpoints',help='后台查看全部参数 ID、训练轮次和指标（含测试检查点）')
    sub.add_parser('list-releases',help='后台查看公开名称与检查点的绑定')
    sub.add_parser('list-versions',help='版本名称、说明及各系列的默认参数')
    sub.add_parser('list-parameters',help='按模型查询稳定 ID、实际完成轮数、网络配置和默认 ID')
    sub.add_parser('get-default',help='查询指定 --family 的默认参数 ID 和配置')
    sub.add_parser('get-latest',help='查询实际训练轮数最多的参数 ID；同轮数取编号较大者')
    best=sub.add_parser('get-best',help='查询一次训练运行按独立古谱验证集选出的最佳参数 ID')
    best.add_argument('--run-id',required=True)
    sub.add_parser('migrate',help='复制旧集中参数库到 HaiKu 目录，保留原库')
    initial = sub.add_parser('initialize',help='保存尚未训练的初始化参数，供开发与训练加载')
    initial.add_argument('--version')
    initial.add_argument('--description')
    initial.add_argument('--seed',type=int,default=42)
    default = sub.add_parser('set-default',help='设置本系列默认加载的参数版本，不自动发布到游戏')
    selector=default.add_mutually_exclusive_group(required=True)
    selector.add_argument('--checkpoint')
    selector.add_argument('--parameter-id','--id',dest='parameter_id',help='使用 ID1 或 1，需要 --family')
    describe = sub.add_parser('describe',help='修改版本名称与说明，不修改权重')
    describe.add_argument('--checkpoint',required=True)
    describe.add_argument('--version')
    describe.add_argument('--description')
    release = sub.add_parser('publish',help='发布名称或更新同名模型对应的参数')
    release.add_argument('--name',required=True)
    selector = release.add_mutually_exclusive_group(required=True)
    selector.add_argument('--checkpoint')
    selector.add_argument('--run-id')
    release.add_argument('--alias',choices=['best','latest'],default='best')
    release.add_argument('--order',type=int)
    hide = sub.add_parser('unpublish',help='从客户端列表下架名称，保留训练参数')
    hide.add_argument('--name',required=True)
    args = p.parse_args(argv)
    args.db = args.db or (family_model_db(args.family) if args.family else MODEL_CATALOG)
    try:
        if args.command == 'migrate':
            print('已迁移，原库保留' if migrate_legacy_store() else '无需迁移')
            for path in _catalog_paths(args.db):
                if path.exists(): init_store(path)
        elif args.command == 'initialize':
            if not args.family: raise ValueError('initialize 需要 --family HaiKu 或 Opus')
            print(initialize_model(args.family,path=args.db,version_name=args.version,
                                   description=args.description,seed=args.seed))
        elif args.command == 'set-default':
            if args.parameter_id is not None:
                if not args.family: raise ValueError('编号 ID 需要 --family，以区分模型系列')
                set_default_parameter_id(args.family,args.parameter_id,path=args.db)
                print(f'已设置默认参数：{args.family}/ID{parameter_number(args.parameter_id)}')
            else:
                set_default_checkpoint(args.checkpoint,path=args.db)
                print(f'已设置默认参数：{args.checkpoint}')
        elif args.command=='get-best':
            if not args.family: raise ValueError('get-best 需要 --family')
            print(json.dumps(get_best_parameter_config(args.family,args.run_id,path=args.db),ensure_ascii=False,indent=2))
        elif args.command in ('list-parameters','get-default','get-latest'):
            for path in _catalog_paths(args.db):
                if path.exists(): init_store(path)
            if args.command in ('get-default','get-latest'):
                if not args.family: raise ValueError(f'{args.command} 需要 --family')
                ident=(get_latest_parameter_id(args.family,path=args.db) if args.command=='get-latest' else None)
                rows=get_parameter_config(args.family,ident,path=args.db)
            else: rows=list_parameter_configs(args.family,path=args.db)
            print(json.dumps(rows,ensure_ascii=False,indent=2))
        elif args.command == 'describe':
            describe_version(args.checkpoint,path=args.db,version_name=args.version,description=args.description)
            print('已更新版本说明')
        elif args.command == 'list-versions':
            rows = list_versions(args.db)
            for row in rows: row['metrics'] = json.loads(row.pop('metrics_json'))
            print(json.dumps(rows,ensure_ascii=False,indent=2))
        elif args.command == 'list-checkpoints':
            rows = list_models(args.db,include_unpublished=True)
            for row in rows:
                row['metrics'] = json.loads(row.pop('metrics_json'))
                row['can_publish'] = bool(row.pop('published') and row['global_step']>0)
            print(json.dumps(rows,ensure_ascii=False,indent=2))
        elif args.command == 'list-releases':
            rows = []
            for path in _catalog_paths(args.db):
                init_store(path)
                with closing(_read_connection(path)) as conn:
                    rows.extend(dict(r) for r in conn.execute('SELECT * FROM model_releases ORDER BY sort_order,name'))
            print(json.dumps(rows,ensure_ascii=False,indent=2))
        elif args.command == 'publish':
            checkpoint_id = args.checkpoint or resolve_alias(args.db,args.run_id,args.alias)
            publish_model(args.name,checkpoint_id,path=args.db,sort_order=args.order)
            print(f'已发布模型：{args.name.strip()}')
        else:
            unpublish_model(args.name,path=args.db)
            print(f'已下架模型：{args.name}')
    except (ValueError,sqlite3.Error) as error:
        p.exit(1,f'{error}\n')


def load_checkpoint(path=MODEL_DB, checkpoint_id=None, device='cpu', resume=False):
    import torch
    from models.registry import build_model
    path = _checkpoint_path(path,checkpoint_id)
    if checkpoint_id is None:
        checkpoint_id = default_checkpoint_id(path)
    if checkpoint_id is None:
        models = list_models(path)
        if not models: raise ValueError('还没有可加载的默认模型参数，请先初始化或训练')
        checkpoint_id = models[0]['checkpoint_id']
    conn = _read_connection(path)
    try:
        row = conn.execute('SELECT * FROM checkpoints WHERE checkpoint_id=?',(checkpoint_id,)).fetchone()
        if row is None: raise ValueError('模型版本不存在')
        payload = row['weights_blob']
        if hashlib.sha256(payload).hexdigest() != row['sha256']: raise ValueError('模型参数校验失败')
        model = build_model(json.loads(row['config_json']))
        model.load_state_dict(torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True),strict=True)
        model.to(device).eval()
        metadata = {k:row[k] for k in row.keys() if k not in ('weights_blob','training_state_blob')}
        metrics=json.loads(metadata['metrics_json'])
        model.value_trained=bool(metrics.get('value_trained',
            metrics.get('last_batch',{}).get('terms',{}).get('value',0)>0))
        state = (torch.load(io.BytesIO(row['training_state_blob']),map_location=device,weights_only=True)
                 if resume and row['training_state_blob'] else {})
        return model,metadata,state
    finally: conn.close()


if __name__ == '__main__': main()

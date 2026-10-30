from contextlib import closing
import json
import sqlite3
import subprocess
import sys

import pytest

from selfplay.runs import delete_run,generation_runs,main,resolve_run_id
from selfplay.stats import count_games
from selfplay.store import SCHEMA,connect,init_store,set_run_log_directory


FEN='rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1'


def add_run(path,ident,n,*,family='Opus',status='complete',unknown=False):
    init_store(family,path)
    with closing(connect(path)) as conn,conn:
        conn.execute('''INSERT INTO runs(run_id,family,architecture,checkpoint_id,version_name,
            weights_sha256,model_config_json,exploration_json,rules_json,has_value_head,seed,
            requested_games,saved_games,completed_games,status,parameter_id,completed_epochs,
            save_every,micro_batch,device,elapsed_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (ident,family,'Opus-5' if family=='Opus' else 'Haiku-5','checkpoint','version','sha',
             '{"dropout":0.1}','{"temperature":1.2,"epsilon":0.02}','{"max_plies":512,"quiet_plies":120}',
             1,42,n,9999,n,status,5,4,32,32,'cuda',12.5))
        for i in range(n):
            game=f'{ident}-{i}'; result='*' if unknown else ('1-0','0-1','1/2-1/2')[i%3]
            conn.execute('''INSERT INTO games(game_id,run_id,start_fen,prefix_json,moves_json,
                result,termination,final_fen,ply_count) VALUES (?,?,?,?,?,?,?,?,?)''',
                (game,ident,FEN,'[]','[4905]',result,'test',FEN,1))
            conn.execute('INSERT INTO steps VALUES (?,?,?,?,?,?,?,?)',(game,0,'r',4905,-1.,0.,1.2,.02))


@pytest.fixture
def store(tmp_path):
    path=tmp_path/'games.sqlite3'
    add_run(path,'first-128',128); add_run(path,'second-1000',1000)
    return path


def test_queries_count_only_each_runs_actual_games_and_results(store,tmp_path,capsys):
    before=store.read_bytes()
    rows=generation_runs('Opus',path=store,logdir=tmp_path/'none')
    assert {r['run_id']:r['saved_games'] for r in rows}=={'first-128':128,'second-1000':1000}
    row=main(['--family','Opus','--data-db',str(store),'--run-id','second-1000'])
    assert row['saved_outcomes']=={'red_wins':334,'black_wins':333,'draws':333}
    assert row['requested_games']==1000 and row['saved_steps']==1000
    assert row['parameter_id']==5 and row['exploration']['temperature']==1.2
    output=capsys.readouterr().out
    assert '红胜 334 局' in output and '黑胜 333 局' in output and '和棋 333 局' in output
    assert '实际已保存 1000 局' in output and '并行微批量=32' in output
    main(['--family','Opus','--data-db',str(store),'--run-id','first-128','--json'])
    assert json.loads(capsys.readouterr().out)['saved_games']==128
    assert store.read_bytes()==before


def test_training_reader_keeps_128_and_1000_runs_separate(store):
    pytest.importorskip('torch')
    from training.data import SelfPlayDataset
    first=SelfPlayDataset(store,'first-128',family='Opus',C=32)
    second=SelfPlayDataset(store,'second-1000',family='Opus',C=32)
    assert len(first.ids)==len(first)==128 and len(second.ids)==len(second)==1000
    assert not set(first.ids)&set(second.ids)
    assert first[0].length==1 and second[0].length==1
    with pytest.raises(ValueError,match='必须指定'): SelfPlayDataset(store,None)
    with pytest.raises(ValueError,match='不存在'): SelfPlayDataset(store,'missing')


def test_legacy_log_folder_resolves_uuid_without_guessing_prefixes(store,tmp_path):
    legacy='opus-selfplay-20261030-033934-fcdd8519'; directory=tmp_path/'logs'/legacy
    directory.mkdir(parents=True)
    (directory/'config.json').write_text(json.dumps(
        {'kind':'selfplay','family':'Opus','run_id':'first-128'}),encoding='utf-8')
    row=generation_runs('Opus',path=store,logdir=tmp_path/'logs',run_id=legacy)[0]
    assert row['run_id']=='first-128' and row['saved_games']==128
    assert resolve_run_id(store,legacy,family='Opus',logdir=directory)=='first-128'
    set_run_log_directory(store,'first-128',directory)
    assert resolve_run_id(store,legacy,family='Opus',logdir=tmp_path/'missing')=='first-128'
    with pytest.raises(ValueError,match='不存在'): resolve_run_id(store,'first',family='Opus')
    delete_run('Opus',legacy,path=store,logdir=directory)
    assert generation_runs('Opus',path=store,logdir=directory,run_id=legacy)==[]
    assert (directory/'config.json').exists()  # Historical logs cannot resurrect deleted games.


def test_deletion_preview_is_readonly_and_deletion_is_scoped_and_cascades(store,tmp_path):
    before=store.read_bytes()
    preview=delete_run('Opus','first-128',path=store,dry_run=True)
    assert preview['deleted_games']==preview['deleted_steps']==128 and store.read_bytes()==before
    with closing(connect(store,True)) as conn:
        other=[tuple(r) for r in conn.execute("SELECT * FROM games WHERE run_id='second-1000'")]
    result=delete_run('Opus','first-128',path=store)
    assert result['saved_outcomes']=={'red_wins':43,'black_wins':43,'draws':42}
    assert count_games('Opus',path=store)==1000
    with closing(connect(store,True)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE run_id='first-128'").fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM steps').fetchone()[0]==1000
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()
        assert [tuple(r) for r in conn.execute("SELECT * FROM games WHERE run_id='second-1000'")]==other
    assert [r['run_id'] for r in generation_runs('Opus',path=store)]==['second-1000']


def test_failed_delete_rolls_back_steps_and_games(store):
    with closing(connect(store)) as conn:
        conn.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT,'test abort'); END")
    with pytest.raises(sqlite3.IntegrityError,match='test abort'):
        delete_run('Opus','first-128',path=store)
    with closing(connect(store,True)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]==1128
        assert conn.execute('SELECT COUNT(*) FROM steps').fetchone()[0]==1128
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==2


def test_running_family_missing_and_cli_guardrails(store,tmp_path,capsys):
    add_run(store,'active',2,status='running')
    before=store.read_bytes()
    with pytest.raises(ValueError,match='生成中'): delete_run('Opus','active',path=store)
    with pytest.raises(ValueError,match='系列'): delete_run('HaiKu','first-128',path=store)
    with pytest.raises(ValueError,match='不存在'): delete_run('Opus','absent',path=store)
    assert store.read_bytes()==before
    result=main(['--family','Opus','--data-db',str(store),'--run-id','active','--delete','--force-running','--json'])
    assert result['deleted_games']==2 and json.loads(capsys.readouterr().out)['deleted_steps']==2
    for args in (['--delete'],['--delete','--family','Opus'],['--dry-run'],['--force-running'],
                 ['--data-db',str(store)]):
        with pytest.raises(SystemExit) as error: main(args)
        assert error.value.code==1
    assert generation_runs('Opus',path=tmp_path/'missing.sqlite3')==[]
    assert not (tmp_path/'missing.sqlite3').exists()
    model_db=tmp_path/'wrong.sqlite3'
    with sqlite3.connect(model_db) as conn: conn.execute('CREATE TABLE checkpoints (id TEXT)')
    before=model_db.read_bytes()
    with pytest.raises(ValueError,match='不是自弈'): delete_run('Opus','anything',path=model_db)
    assert model_db.read_bytes()==before


def test_unknown_results_and_legacy_schema_without_new_columns(tmp_path):
    path=tmp_path/'old.sqlite3'
    with closing(sqlite3.connect(path)) as conn,conn:
        conn.executescript(SCHEMA);conn.execute("INSERT INTO metadata VALUES ('family','HaiKu')")
        conn.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            ('legacy','HaiKu','Haiku-5','id','old','sha','{}','{}','{}',0,42,'2026-10-01'))
    before=path.read_bytes()
    row=generation_runs('HaiKu',path=path)[0]
    assert row['saved_games']==0 and row.get('elapsed_seconds') is None
    assert path.read_bytes()==before
    assert delete_run('HaiKu','legacy',path=path)['deleted_games']==0
    add_run(path,'unknown',1,family='HaiKu',unknown=True)
    row=generation_runs('HaiKu',path=path)[0]
    assert row['unknown_results']==1 and sum(row['saved_outcomes'].values())==0


def test_query_command_never_imports_torch_or_tensorboard(store):
    code='''import sys,json
from selfplay.__main__ import main
main(['runs','--family','Opus','--data-db',sys.argv[1],'--run-id','first-128','--json'])
assert 'torch' not in sys.modules and 'tensorboard' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-X','utf8','-c',code,str(store)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['saved_games']==128


@pytest.mark.parametrize('source',['selfplay','mixed'])
def test_training_requires_one_run_before_loading_model(source,tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from training.__main__ import main as train
    def unexpected_load(*args,**kwargs): raise AssertionError('must reject missing run before loading weights')
    monkeypatch.setattr('training.__main__.load_parameter_model',unexpected_load)
    with pytest.raises(SystemExit) as error:
        train(['--family','Opus','--source',source,'--output',str(tmp_path/'runs')])
    assert error.value.code==1 and not (tmp_path/'runs').exists()

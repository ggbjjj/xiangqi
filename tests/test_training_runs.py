from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from model_store import SCHEMA
from training.runs import main,training_runs


RUN='opus-mixed-example'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def make_store(path,family,run_id=RUN):
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as conn,conn:
        conn.executescript(SCHEMA)
        for field in ('parameter_id INTEGER','completed_epochs INTEGER','parent_checkpoint_id TEXT',
                      'version_name TEXT','description TEXT'):
            conn.execute('ALTER TABLE checkpoints ADD COLUMN '+field)
        architecture='Haiku-5' if family=='HaiKu' else 'Opus-5'
        conn.execute('INSERT INTO architectures VALUES (?,?)',(architecture,'{}'))
        for number in (1,2,3):
            metrics={'train':{'loss':4-number,'terms':{'book':3-number/2}},
                     'validation':{'loss':1.5,'perplexity':8-number,'L':32,'n':16}}
            conn.execute('''INSERT INTO checkpoints(checkpoint_id,architecture,run_id,stage,epoch,global_step,
                config_json,weights_blob,metrics_json,sha256,published,parameter_id,completed_epochs,
                version_name,description,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (f'{family}-{number}',architecture,'initial' if number==1 else run_id,
                 'initial' if number==1 else 'pretrain',number-1,(number-1)*10,'{}',
                 b'not a valid weight blob: querying must never deserialize this',json.dumps(metrics),
                 'test-sha',1,number,number,f'{family}-version-{number}','test',f'2026-10-30T0{number}:00:00Z'))
        conn.execute('INSERT INTO model_defaults VALUES (1,?)',(f'{family}-2',))
        conn.execute('INSERT INTO aliases VALUES (?,?,?)',(run_id,'best',f'{family}-3'))


@pytest.fixture
def run_files(tmp_path):
    catalog=tmp_path/'models'; root=tmp_path/'runs'; directory=root/'mixed'/RUN
    make_store(catalog/'Opus'/'models.sqlite3','Opus')
    make_store(catalog/'HaiKu'/'models.sqlite3','HaiKu','haiku-example')
    config={'family':'Opus','source':'books','epochs':2,'lr':3e-4,'batch_size':128,'micro_batch':16,
            'T':32,'validation_length':32,'ngram':16,'book_source':'both',
            'weights':{'book':1,'value':1,'ppo':0,'entropy':0},'base_checkpoint_id':'Opus-1',
            'training_samples':84853,'book_games':29181,'book_segments':84853,
            'validation_games':1687,'validation_segments':4830,'model':{'architecture':'Opus-5','dropout':.1}}
    write_json(directory/'config.json',config)
    write_json(directory/'report.json',{'run_id':RUN,'completed_epochs_this_run':2,'completed_epochs':3,
                                       'interrupted':False,'checkpoints':['Opus-2','Opus-3']})
    write_json(directory/'last_completed.json',{'parameter_id':3,'completed_epochs':3,'completed_epochs_this_run':2,
                                              'validation':{'perplexity':5,'L':32,'n':16}})
    write_json(directory/'best.json',{'parameter_id':3,'validation':{'perplexity':5}})
    (directory/'metrics.jsonl').write_text(json.dumps({'epoch':2,'global_step':20,'loss':1.2})+'\n',encoding='utf-8')
    return root,catalog,directory


def test_inventory_merges_logs_and_metadata_without_loading_or_writing_weights(run_files):
    root,catalog,directory=run_files
    files=[p for parent in (root,catalog) for p in parent.rglob('*') if p.is_file()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    rows=training_runs(logdir=root,model_db=catalog)
    assert len(rows)==2 and {r['run_id'] for r in rows}=={RUN,'haiku-example'}
    row=next(r for r in rows if r['run_id']==RUN)
    assert row['status']=='complete' and row['completed_epochs_this_run']==2 and row['completed_epochs']==3
    assert row['last_parameter_id']==row['best_parameter_id']==3 and row['base_parameter_id']==1
    assert row['latest_validation']['perplexity']==5 and row['config']['lr']==3e-4
    assert [p['parameter_id'] for p in row['parameters']]==[2,3]
    assert row['parameters'][0]['is_default'] and not row['parameters'][0]['is_best']
    assert row['parameters'][1]['is_best'] and not row['parameters'][1]['is_default']
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==before
    assert set(p for parent in (root,catalog) for p in parent.rglob('*') if p.is_file())==set(files)


def test_cli_lists_and_queries_human_and_json_formats(run_files,capsys):
    root,catalog,_=run_files
    args=['--logdir',str(root),'--model-db',str(catalog),'--family','Opus']
    result=main(args)
    assert len(result)==1 and RUN in capsys.readouterr().out
    result=main(args+['--run-id',RUN])
    output=capsys.readouterr().out
    assert '学习率：0.0003' in output and 'L=32；n=16' in output and 'ID3' in output
    assert '价值=1' in output and '总训练片段：84853' in output
    main(args+['--run-id',RUN,'--json'])
    assert json.loads(capsys.readouterr().out)['parameters'][1]['checkpoint_id']=='Opus-3'
    with pytest.raises(SystemExit) as error: main(args+['--run-id','not-found'])
    assert error.value.code==1


def test_in_progress_partial_records_are_not_reported_as_completed(run_files):
    root,catalog,directory=run_files
    (directory/'report.json').write_text('{"run_id":',encoding='utf-8')
    with (directory/'metrics.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'epoch':3,'global_step':23,'loss':.9})+'\n{"epoch":3,')
    row=training_runs(logdir=root,model_db=catalog,family='Opus')[0]
    assert row['status']=='unknown' and row['completed_epochs_this_run']==2
    assert row['latest_metrics']['global_step']==23
    assert row['latest_metrics']['epoch']==3 and row['completed_epochs']==3
    assert any('没有结束报告' in w for w in row['warnings'])


def test_interrupted_and_no_validation_runs_keep_unknown_metrics(run_files):
    root,catalog,directory=run_files
    write_json(directory/'report.json',{'run_id':RUN,'interrupted':True,'completed_epochs_this_run':2})
    with closing(sqlite3.connect(catalog/'Opus'/'models.sqlite3')) as conn,conn:
        conn.execute('UPDATE checkpoints SET metrics_json=?',('{"train_loss":1.1}',))
        conn.execute('DELETE FROM aliases')
    (directory/'last_completed.json').unlink();(directory/'best.json').unlink()
    row=training_runs(logdir=root,model_db=catalog,family='Opus')[0]
    assert row['status']=='interrupted' and row['best_parameter_id'] is None
    assert row['latest_validation'] is None and row['parameters'][0]['train_loss']==1.1


def test_log_only_and_checkpoint_only_queries(run_files,tmp_path):
    root,catalog,directory=run_files
    row=training_runs(logdir=root,model_db=tmp_path/'missing-models',family='Opus')[0]
    assert row['status']=='complete' and row['last_parameter_id']==3 and row['best_parameter_id']==3
    assert row['latest_validation']['perplexity']==5 and row['parameters']==[]
    assert not (tmp_path/'missing-models').exists()
    rows=training_runs(logdir=tmp_path/'missing-logs',model_db=catalog)
    assert len(rows)==2 and all(row['status']=='unknown' for row in rows)
    assert not (tmp_path/'missing-logs').exists()


def test_generation_and_standalone_evaluation_not_listed_as_training(tmp_path):
    root=tmp_path/'logs'
    write_json(root/'selfplay'/'config.json',{'kind':'selfplay','family':'Opus','epochs':3})
    write_json(root/'evaluation'/'config.json',{'family':'Opus','validation_length':32})
    write_json(root/'legacy-eval'/'config.json',{'family':'Opus','epochs':3,'evaluate':'some-id'})
    write_json(root/'broken'/'config.json',[])
    assert training_runs(logdir=root,model_db=tmp_path/'missing')==[]


def test_same_run_id_in_two_families_requires_family(run_files,capsys):
    root,catalog,_=run_files
    with closing(sqlite3.connect(catalog/'HaiKu'/'models.sqlite3')) as conn,conn:
        conn.execute('UPDATE checkpoints SET run_id=? WHERE stage<>?',(RUN,'initial'))
    args=['--logdir',str(root),'--model-db',str(catalog),'--run-id',RUN]
    with pytest.raises(SystemExit) as error: main(args)
    assert error.value.code==1 and '同名 run_id' in capsys.readouterr().err
    assert main(args+['--family','HaiKu'])['family']=='HaiKu'


def test_old_resume_log_directories_share_run_id_and_keep_configs(tmp_path):
    root=tmp_path/'logs'; catalog=tmp_path/'models'
    make_store(catalog/'HaiKu'/'models.sqlite3','HaiKu','old-haiku-run')
    for index in (1,2):
        directory=root/f'resume-{index}'/'old-haiku-run'
        write_json(directory/'config.json',{'epochs':2,'lr':index*1e-4,'resume':'some-checkpoint',
                                           'model':{'architecture':'Haiku-5'}})
        write_json(directory/'evaluation.json',{'run_id':'old-haiku-run','test':{}})
    row=training_runs(logdir=root,model_db=catalog)[0]
    assert len(row['log_dirs'])==len(row['config_history'])==2 and row['status']=='complete'
    assert row['config']['lr']==2e-4 and row['latest_validation']['perplexity']==5


def test_run_query_imports_no_torch_or_tensorboard():
    code="import sys; import training.runs; assert 'torch' not in sys.modules; assert 'tensorboard' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(__file__).resolve().parents[1],capture_output=True)

import copy
import json
import math
import signal
import sqlite3
import threading
from contextlib import closing

import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(2)
from model_store import (save_checkpoint,load_checkpoint,load_parameter_model,list_parameter_configs,
    get_parameter_config,get_default_parameter_id,set_default_parameter_id,get_latest_parameter_id,
    list_versions,init_store,main as admin)
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from data_pipeline.database import connect
from data_pipeline.notation import STANDARD_FEN
from encoder import move_to_action
from training.data import BookDataset,SelfPlayDataset,collate_examples
from training.control import StopController
from selfplay.generator import SelfPlayGenerator,SelfPlayRules
from selfplay.store import list_runs,load_game,save_games,committed_game_ids


@pytest.fixture
def saved(tmp_path):
    net=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    db=tmp_path/'models.sqlite3'; ids=[]
    for n in (1,2):
        with torch.no_grad(): net.policy_head.bias.fill_(n)
        ids.append(save_checkpoint(net,path=db,run_id='original',epoch=n,global_step=n,metrics={},
                         version_name=f'HaiKu-test.{n}',set_default=n==1))
    books=tmp_path/'books.sqlite3'; conn=connect(books)
    for i in range(2):
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
              (str(i),STANDARD_FEN,json.dumps([move_to_action((6,0,5,0))]),1,'*','human_master','train',str(i),'{}'))
    conn.commit(); conn.close()
    return net,db,ids,books


def test_stable_ids_actual_epochs_latest_and_default_are_independent(saved,capsys):
    net,db,ids,books=saved
    rows=list_parameter_configs('HaiKu',path=db)
    assert [(r['parameter_id'],r['completed_epochs']) for r in rows]==[(1,1),(2,2)]
    assert get_default_parameter_id('HaiKu',path=db)==1
    assert get_latest_parameter_id('HaiKu',path=db)==2
    model,meta,_=load_parameter_model('HaiKu','ID2',path=db)
    assert torch.all(model.policy_head.bias==2) and meta['checkpoint_id']==ids[1]
    ident=save_checkpoint(net,path=db,run_id='new',epoch=3,global_step=5,metrics={},
                           completed_epochs=5,parent_checkpoint_id=ids[1])
    third=get_parameter_config('HaiKu','ID3',path=db)
    assert third['completed_epochs']==5 and third['epoch_in_run']==3 and third['parent_checkpoint_id']==ids[1]
    assert third['model_config']==net.config.to_dict()
    assert get_latest_parameter_id('HaiKu',path=db)==3 and get_default_parameter_id('HaiKu',path=db)==1
    set_default_parameter_id('HaiKu','ID2',path=db)
    assert get_default_parameter_id('HaiKu',path=db)==2
    assert load_parameter_model('HaiKu',path=db)[1]['checkpoint_id']==ids[1]
    before=db.read_bytes(); init_store(db); assert get_parameter_config('HaiKu',3,path=db)['checkpoint_id']==ident
    assert [r['parameter_id'] for r in list_parameter_configs('HaiKu',path=db)]==[1,2,3]
    admin(['--db',str(db),'--family','HaiKu','get-latest'])
    assert json.loads(capsys.readouterr().out)['parameter_id']==3
    admin(['--db',str(db),'--family','HaiKu','set-default','--id','ID3'])
    capsys.readouterr()
    admin(['--db',str(db),'--family','HaiKu','get-default'])
    assert json.loads(capsys.readouterr().out)['completed_epochs']==5


def _train_args(db,books,out):
    return ['--family','HaiKu','--model-db',str(db),'--source','books','--book-db',str(books),
            '--epochs','3','--T','4','--output',str(out),'--batch-size','2','--micro-batch','1',
            '--device','cpu','--cpu-threads','2']


def test_training_auto_loads_most_trained_id_and_counts_two_plus_three_as_five(saved,tmp_path):
    from training.__main__ import main
    net,db,ids,books=saved
    original=load_checkpoint(db,ids[1])[1]['sha256']
    report=main(_train_args(db,books,tmp_path/'runs'))
    assert not report['interrupted'] and report['completed_epochs']==5 and report['completed_epochs_this_run']==3
    assert [r['completed_epochs'] for r in list_parameter_configs('HaiKu',path=db)]==[1,2,3,4,5]
    assert get_latest_parameter_id('HaiKu',path=db)==5 and get_default_parameter_id('HaiKu',path=db)==1
    assert load_checkpoint(db,ids[1])[1]['sha256']==original
    final=get_parameter_config('HaiKu','ID5',path=db)
    assert '实际训练 5 轮' in final['description'] and '-e5-' in final['version_name']
    config=json.loads(next((tmp_path/'runs').glob('*/config.json')).read_text(encoding='utf-8'))
    assert config['base_checkpoint_id']==ids[1]


def test_explicit_training_id_overrides_latest(saved,tmp_path):
    from training.__main__ import main
    net,db,ids,books=saved
    args=_train_args(db,books,tmp_path/'chosen')+['--id','ID1','--epochs','1']
    report=main(args)
    assert report['completed_epochs']==2
    row=list_parameter_configs('HaiKu',path=db)[-1]
    assert row['completed_epochs']==2 and row['parent_checkpoint_id']==ids[0]


@pytest.mark.parametrize('interrupt_after',[0,1])
def test_interrupt_during_training_keeps_only_complete_epochs(saved,tmp_path,monkeypatch,interrupt_after):
    import training.__main__ as cli
    net,db,ids,books=saved
    original=cli.train_logical_batch; calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        if calls==interrupt_after: raise KeyboardInterrupt
        calls+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(cli,'train_logical_batch',interrupt)
    report=cli.main(_train_args(db,books,tmp_path/'interrupted'))
    assert report['interrupted'] and report['completed_epochs_this_run']==interrupt_after
    assert report['completed_epochs']==2+interrupt_after
    assert len(list_parameter_configs('HaiKu',path=db))==2+interrupt_after
    assert get_latest_parameter_id('HaiKu',path=db)==2+interrupt_after


def test_stop_signal_handler_is_cooperative_and_restored():
    previous=signal.getsignal(signal.SIGTERM)
    with StopController() as stop:
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGTERM)==previous


def test_interrupt_inside_second_epoch_does_not_save_partial_updates(saved,tmp_path,monkeypatch):
    import training.__main__ as cli
    net,db,ids,books=saved
    original=cli.train_logical_batch; calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        if calls==3: raise KeyboardInterrupt
        calls+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(cli,'train_logical_batch',interrupt)
    report=cli.main(_train_args(db,books,tmp_path/'partial')+['--batch-size','1'])
    assert report['interrupted'] and report['completed_epochs']==3 and report['global_step']==4
    assert len(list_parameter_configs('HaiKu',path=db))==3
    model,meta,state=load_parameter_model('HaiKu','ID3',path=db,resume=True)
    assert meta['completed_epochs']==3
    assert all(int(s['step'])==2 for s in state['optimizer']['state'].values())


@pytest.mark.parametrize('T,expected_length',[(1,1),(4,1)])
def test_books_auto_crop_or_pad_to_exact_T(saved,T,expected_length):
    net,db,ids,books=saved
    sample=BookDataset(T,db=books)[0]; batch=collate_examples([sample])
    assert batch.x.shape==(T,1,15,10,9) and batch.loss.lengths.tolist()==[expected_length]
    if T>1:
        assert not batch.x[1:].count_nonzero() and (batch.loss.actions[1:]==-100).all()


def _generator(saved):
    net,db,ids,books=saved
    return SelfPlayGenerator.from_version('HaiKu','ID2',model_db=db,rules=SelfPlayRules(max_plies=1))


@pytest.mark.parametrize('stop_after,expected_saved',[(1,0),(2,2),(3,2)])
def test_generator_interrupt_discards_only_uncommitted_K_group(saved,tmp_path,stop_after,expected_saved):
    generator=_generator(saved); stop=threading.Event()
    path=tmp_path/'rollout.sqlite3'
    def progress(row):
        if row['game']==stop_after: stop.set()
    report=generator.generate(5,path=path,save_every=2,cancel_event=stop,progress=progress)
    assert report['interrupted'] and report['saved_games']==expected_saved
    assert report['discarded_games']==stop_after-expected_saved
    assert len(committed_game_ids(path,report['run_id']))==expected_saved
    assert list_runs(path)[0]['status']=='interrupted'
    assert generator.snapshot.parameter_id==2 and generator.snapshot.completed_epochs==2


def test_normal_end_commits_final_short_group_and_default_K_is_16(saved,tmp_path):
    generator=_generator(saved); observed=[]
    path=tmp_path/'normal.sqlite3'
    report=generator.generate(3,path=path,progress=lambda row:observed.append(row['saved_games']))
    assert observed==[0,0,0] and report['saved_games']==3 and report['save_every']==16
    assert not report['interrupted'] and not report['discarded_games']
    assert len(SelfPlayDataset(path,report['run_id']))==3


def test_selfplay_reader_applies_T_including_prefix_without_losing_ppo_context(saved,tmp_path):
    generator=_generator(saved)
    generator.rules=SelfPlayRules(max_plies=3)
    prefix=[move_to_action((6,0,5,0))]
    report=generator.generate(1,path=tmp_path/'reader.sqlite3',prefix=prefix)
    for T in (2,6):
        sample=SelfPlayDataset(report['data_db'],report['run_id'],T=T)[0]
        batch=collate_examples([sample])
        length=min(T,4)
        assert batch.x.shape==(T,1,15,10,9) and batch.loss.lengths.tolist()==[length]
        assert batch.loss.counts().ppo==length-1 and not batch.loss.selfplay_mask[0,0]
        if T>length: assert not batch.x[length:].count_nonzero()
    with pytest.raises(ValueError,match='暖启动'):
        SelfPlayDataset(report['data_db'],report['run_id'],T=1)[0]


def test_bulk_save_rolls_back_entire_group(saved,tmp_path):
    generator=_generator(saved)
    report=generator.generate(1,path=tmp_path/'atomic.sqlite3')
    game,run,steps=load_game(report['data_db'],report['game_ids'][0])
    valid={'start_fen':game['start_fen'],'prefix':[],'steps':steps,'result':game['result'],
           'termination':game['termination'],'final_fen':game['final_fen']}
    invalid=copy.deepcopy(valid); invalid['steps'][0]['old_log_mu']=float('nan')
    with pytest.raises(ValueError): save_games(report['data_db'],report['run_id'],[valid,invalid])
    assert len(committed_game_ids(report['data_db'],report['run_id']))==1


def test_game_loads_by_family_id_and_falls_back_to_its_default(saved,monkeypatch):
    from ai import load_network
    import main
    import ui
    net,db,ids,books=saved
    assert load_network(db,family='HaiKu',parameter_id='ID2').parameter_id==2
    assert load_network(db,family='HaiKu').parameter_id==1
    calls=[]
    monkeypatch.setattr(ui,'run_app',lambda **kwargs:calls.append(kwargs))
    main.main(['gui','--family','HaiKu','--id','ID2','--model-db',str(db)])
    main.main(['gui','--family','HaiKu','--model-db',str(db)])
    assert [c['parameter_id'] for c in calls]==[2,1]

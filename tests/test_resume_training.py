import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

torch=pytest.importorskip('torch')
from data_pipeline.database import connect
from data_pipeline.notation import STANDARD_FEN
from encoder import move_to_action
from model_store import load_checkpoint,save_checkpoint
from models.haiku5.config import Haiku5Config
from models.haiku5.network import Haiku5
from resume_train import main,validate_record


@pytest.fixture
def saved_training(tmp_path):
    data=tmp_path/'data.sqlite3'
    conn=connect(data)
    ids=[]
    for split in ('train','validation','test'):
        ident=split+'-game'; ids.append(ident)
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                     (ident,STANDARD_FEN,json.dumps([move_to_action((6,0,5,0))]),1,'*',
                      'human_master',split,ident,'{}'))
    conn.commit(); conn.close()
    fingerprint=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    torch.set_num_threads(2)
    net=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8,context_plies=1))
    optimizer=torch.optim.AdamW(net.parameters(),lr=0.0003,weight_decay=0.0001)
    net(torch.randn(1,15,10,9)).square().mean().backward(); optimizer.step()
    db=tmp_path/'models.sqlite3'
    ident=save_checkpoint(net,path=db,run_id='fixture',epoch=1,global_step=1,
                          metrics={'train_loss':4.,'validation':{'loss':4.,'top1':0.,'top3':0.}},
                          optimizer=optimizer,best=True,training_state={
                              'T':1,'seed':42,'dataset_fingerprint':fingerprint,'best_loss':4.,
                              'rng_state':torch.get_rng_state(),'cuda_rng_state':[],'scaler':{},'micro_step':1})
    _,meta,_=load_checkpoint(db,ident)
    config={'model_db':str(db),'data':str(data),'checkpoint_id':ident,'run_id':'fixture',
            'completed_epoch':1,'default_total_epochs':2,'weights_sha256':meta['sha256'],
            'dataset_fingerprint':fingerprint,'source_history':str(tmp_path/'missing-history.csv'),
            'output_base':str(tmp_path/'resumed'),'public_model_name':'HaiKu-5',
            'T':1,'seed':42,'batch_size':128,'micro_batch':16,'workers':0,'cpu_threads':2,
            'lr':0.0003,'weight_decay':0.0001,'clip_grad':1.,'limit_games':0,
            'device':'cuda','precision':'auto'}
    record=tmp_path/'resume.json'
    record.write_text(json.dumps(config),encoding='utf-8')
    return config,record


def test_check_only_preserves_parameters_and_does_not_start_training(saved_training):
    config,record=saved_training
    db=Path(config['model_db']); before=db.read_bytes()
    args=main(['--record',str(record),'--check-only'])
    options=dict(zip(args[::2],args[1::2]))
    assert options['--resume']==config['checkpoint_id'] and options['--epochs']=='2'
    assert options['--micro-batch']=='16'
    assert before==db.read_bytes() and not Path(config['output_base']).exists()
    bad=dict(config,seed=43)
    with pytest.raises(ValueError,match='seed'): validate_record(bad)
    bad=dict(config,dataset_fingerprint='wrong')
    with pytest.raises(ValueError,match='数据划分'): validate_record(bad)


def test_resume_continues_epoch_and_optimizer_keeps_original_checkpoint(saved_training):
    config,record=saved_training
    main(['--record',str(record),'--device','cpu','--precision','none','--epochs','2'])
    with sqlite3.connect(config['model_db']) as conn:
        rows=conn.execute('SELECT checkpoint_id,epoch,global_step FROM checkpoints ORDER BY epoch').fetchall()
    assert len(rows)==2 and rows[0]==(config['checkpoint_id'],1,1) and rows[1][1:]==(2,2)
    _,original,_=load_checkpoint(config['model_db'],config['checkpoint_id'])
    _,_,state=load_checkpoint(config['model_db'],rows[1][0],resume=True)
    assert original['sha256']==config['weights_sha256']
    resumed,_,_=load_checkpoint(config['model_db'],rows[1][0])
    assert resumed.config.context_plies==1
    assert state['T']==1 and state['context_plies']==1
    assert all(int(item['step'])==2 for item in state['optimizer']['state'].values())
    outputs=list(Path(config['output_base']).glob('*/fixture'))
    assert len(outputs)==1 and (outputs[0]/'resume_origin.json').exists()
    assert (outputs[0]/'evaluation.json').exists()
    lines=(outputs[0]/'history.csv').read_text().splitlines()
    assert len(lines)==3 and lines[1].startswith('1,') and lines[2].startswith('2,')


def test_resume_rejects_total_epochs_already_completed(saved_training):
    config,record=saved_training
    with pytest.raises(SystemExit) as error:
        main(['--record',str(record),'--epochs','1','--check-only'])
    assert error.value.code==1 and not Path(config['output_base']).exists()


@pytest.mark.parametrize('sequence_length,window',[(3,2),(1,4),(1,None)])
def test_fresh_training_saves_independent_sequence_length_and_window(saved_training,tmp_path,monkeypatch,
                                                                  sequence_length,window):
    import train
    config,_=saved_training
    monkeypatch.setattr(train,'Haiku5',lambda cfg: Haiku5(replace(cfg,dim=16,ffn_dim=32,cnn_channels=8)))
    db=tmp_path/'fresh-models.sqlite3'
    output=tmp_path/'fresh-runs'
    args=['--data',config['data'],'--model-db',str(db),'--output',str(output),
          '--epochs','1','--T',str(sequence_length),'--batch-size','1','--micro-batch','1',
          '--workers','0','--cpu-threads','2','--device','cpu','--precision','none',
          '--run-name','independent-window','--smoke']
    if window is not None:
        args+=['--context-plies',str(window)]
    train.main(args)
    report=json.loads((output/'independent-window'/'evaluation.json').read_text(encoding='utf-8'))
    net,_,state=load_checkpoint(db,report['checkpoint_id'],resume=True)
    expected=120 if window is None else window
    assert net.config.context_plies==expected
    assert all(block.attn.limit==expected for core in net.cores for block in [*core.history,core.cross])
    assert state['T']==sequence_length and state['context_plies']==expected
    saved=json.loads((output/'independent-window'/'config.json').read_text(encoding='utf-8'))
    assert saved['T']==sequence_length and saved['context_plies']==saved['model']['context_plies']==expected


def test_resume_window_override_selects_best_with_new_context(saved_training,monkeypatch):
    import train
    config,record=saved_training
    # A worse score must still replace the previous window's best checkpoint.
    monkeypatch.setattr(train,'evaluate',lambda *a,**kw: {'loss':10.,'top1':0.,'top3':0.,'games':1,'steps':1})
    main(['--record',str(record),'--device','cpu','--precision','none',
          '--epochs','2','--context-plies','2'])
    evaluation=next(Path(config['output_base']).glob('*/fixture/evaluation.json'))
    report=json.loads(evaluation.read_text(encoding='utf-8'))
    net,meta,state=load_checkpoint(config['model_db'],report['checkpoint_id'],resume=True)
    assert meta['epoch']==2 and meta['global_step']==2
    assert net.config.context_plies==state['context_plies']==2 and state['T']==1
    assert all(block.attn.limit==2 for core in net.cores for block in [*core.history,core.cross])
    assert all(int(item['step'])==2 for item in state['optimizer']['state'].values())
    original,original_meta,_=load_checkpoint(config['model_db'],config['checkpoint_id'])
    assert original.config.context_plies==1 and original_meta['sha256']==config['weights_sha256']
    assert 'context_plies' not in json.loads(record.read_text(encoding='utf-8'))


def test_resume_record_and_cli_forward_context_independently(saved_training):
    config,record=saved_training
    record.write_text(json.dumps(dict(config,context_plies=3)),encoding='utf-8')
    args=main(['--record',str(record),'--check-only'])
    options=dict(zip(args[::2],args[1::2]))
    assert options['--T']=='1' and options['--context-plies']=='3'
    args=main(['--record',str(record),'--check-only','--context-plies','2'])
    options=dict(zip(args[::2],args[1::2]))
    assert options['--T']=='1' and options['--context-plies']=='2'


@pytest.mark.parametrize('window',[None,2])
def test_evaluation_inherits_or_overrides_saved_window_without_writes(saved_training,monkeypatch,window):
    import train
    config,_=saved_training
    before=Path(config['model_db']).read_bytes()
    seen=[]
    def evaluate(model,loader,*args,**kwargs):
        seen.append((model.config.context_plies,loader.dataset.T))
        assert all(block.attn.limit==model.config.context_plies
                   for core in model.cores for block in [*core.history,core.cross])
        return {'loss':1.,'top1':0.,'top3':0.,'games':1,'steps':1}
    monkeypatch.setattr(train,'evaluate',evaluate)
    args=['--evaluate',config['checkpoint_id'],'--model-db',config['model_db'],
          '--data',config['data'],'--T','3','--device','cpu','--precision','none','--workers','0']
    if window is not None:
        args+=['--context-plies',str(window)]
    train.main(args)
    assert seen==[(1 if window is None else window,3)]
    assert before==Path(config['model_db']).read_bytes()


@pytest.mark.parametrize('window',['0','-1'])
def test_training_and_resume_reject_nonpositive_windows(window):
    from train import main as train_main
    for entry in (train_main,main):
        with pytest.raises(SystemExit) as error:
            entry(['--context-plies',window])
        assert error.value.code==2

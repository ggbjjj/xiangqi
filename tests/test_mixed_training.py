import copy
from dataclasses import replace
import json
import math
import random
import sqlite3

import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(2)
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from models.HaiKu.loss import masked_policy_ce
from encoder import move_to_action
from model_store import save_checkpoint,load_checkpoint,load_version,list_versions
from selfplay.generator import SelfPlayGenerator,SelfPlayRules,Position
from selfplay.store import init_store,list_runs,load_game
from training.exploration import Exploration,behavior_log_probs,behavior_entropy
from training.loss import LossBatch,LossWeights,LossCounts,mixed_loss
from training.data import SelfPlayDataset,BookDataset,collate_examples,result_target
from training.engine import ModelOutput,iter_chunks,train_logical_batch
from data_pipeline.notation import STANDARD_FEN


def tiny_opus():
    return Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
                            ffn_dim=32,cnn_channels=8,value_hidden=8,checkpoint_attention=True))


def fake_batch(t=4,b=2):
    legal=torch.zeros(t,b,8100,dtype=torch.bool); legal[...,:3]=True
    actions=torch.zeros(t,b,dtype=torch.long); lengths=torch.full((b,),t)
    book=torch.zeros(t,b,dtype=torch.bool); book[:,0]=True
    own=~book; values=torch.full((t,b),-100,dtype=torch.long); values[own]=0
    return LossBatch(actions,legal,lengths,book,own,values,
                     old_log_mu=torch.full((t,b),-math.log(3)),old_value=torch.full((t,b),.2),
                     temperature=torch.ones(t,b),epsilon=torch.full((t,b),.02))


@pytest.mark.parametrize('epsilon',[0.,.1,1.])
def test_behavior_distribution_endpoints_and_finite_backward(epsilon):
    logits=torch.zeros(8100,requires_grad=True)
    with torch.no_grad(): logits[:3]=torch.tensor([.7,.2,.1]).log()
    legal=torch.arange(8100)<3
    log_mu=behavior_log_probs(logits,legal,1.,epsilon)
    expected=(1-epsilon)*torch.tensor([.7,.2,.1])+epsilon/3
    assert torch.allclose(log_mu[:3].exp(),expected,atol=1e-7)
    assert log_mu[3:].exp().sum()==0 and torch.allclose(log_mu.exp().sum(),torch.tensor(1.))
    loss=-log_mu[0]-.01*behavior_entropy(log_mu,legal)
    loss.backward(); assert torch.isfinite(logits.grad).all()
    if epsilon==1: assert not logits.grad.count_nonzero()


def test_temperature_uses_logits_and_legal_mask():
    z=torch.zeros(2,8100); z[0,:3]=torch.tensor([4.,1.,0.]); z[1,:3]=torch.tensor([2.,3.,4.])
    legal=torch.arange(8100).expand(2,-1)<3
    mu=behavior_log_probs(z,legal,torch.tensor([2.,.5]),torch.tensor([0.,.1])).exp()
    assert torch.allclose(mu[0,:3],(z[0,:3]/2).softmax(-1))
    assert torch.allclose(mu[1,:3],.9*(z[1,:3]/.5).softmax(-1)+.1/3)
    with pytest.raises(ValueError,match='合法'): behavior_log_probs(z,torch.zeros_like(legal))


@pytest.mark.parametrize('ratio,advantage,expected',[(1.5,.8,-.96),(.7,-.8,.64)])
def test_ppo_clip_examples_and_constant_old_measurements(ratio,advantage,expected):
    logits=torch.zeros(1,1,8100,requires_grad=True)
    legal=torch.zeros_like(logits,dtype=torch.bool); legal[...,:2]=True
    old_log=torch.tensor([[math.log(.5/ratio)]],requires_grad=True)
    old_value=torch.tensor([[1-advantage if advantage>0 else -1-advantage]],requires_grad=True)
    target=torch.tensor([[0 if advantage>0 else 2]])
    data=LossBatch(torch.zeros(1,1,dtype=torch.long),legal,torch.tensor([1]),
                   selfplay_mask=torch.ones(1,1,dtype=torch.bool),value_targets=target,
                   old_log_mu=old_log,old_value=old_value)
    result=mixed_loss(logits,data,LossWeights(0,1,0,0))
    assert float(result.total.detach())==pytest.approx(expected,abs=1e-6)
    result.total.backward()
    assert old_log.grad is None and old_value.grad is None
    assert torch.isfinite(logits.grad).all() and not logits.grad.count_nonzero()


def test_disabled_branches_skip_nan_inputs_and_match_legacy_book_ce():
    logits=torch.randn(3,1,8100,requires_grad=True)
    legal=torch.ones_like(logits,dtype=torch.bool); y=torch.tensor([[2],[7],[-100]])
    lengths=torch.tensor([2])
    data=LossBatch.books(y,legal,lengths,old_log_mu=torch.full((3,1),torch.nan),old_value=torch.full((3,1),torch.nan))
    out=ModelOutput(logits,torch.full((3,1,3),torch.nan))
    actual=mixed_loss(out,data,LossWeights()).total
    expected=masked_policy_ce(logits,y,legal,lengths)[0]
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward(); assert torch.isfinite(logits.grad).all() and not logits.grad[2].count_nonzero()
    data.legal[0,0,2]=False
    with pytest.raises(ValueError,match='非法'): mixed_loss(out,data)


def test_source_specific_whole_batch_denominators_match_split_gradients():
    torch.manual_seed(7)
    data=fake_batch(4,2); data.book_mask[-1,0]=False; data.lengths[0]=3
    data.value_targets[0,1]=-100  # unknown outcome, excluded from PPO/value/entropy
    policy=torch.randn(4,2,8100,requires_grad=True); value=torch.randn(4,2,3,requires_grad=True)
    split_policy=policy.detach().clone().requires_grad_(); split_value=value.detach().clone().requires_grad_()
    weights=LossWeights.mixed(); counts=data.counts()
    assert counts==LossCounts(3,3,3,6)
    result=mixed_loss(ModelOutput(policy,value),data,weights); result.total.backward()
    total=0.
    for start,stop in ((0,2),(2,4)):
        r=mixed_loss(ModelOutput(split_policy[start:stop],split_value[start:stop]),data.slice(start,stop),
                     weights,denominators=counts)
        total+=float(r.total.detach()); r.total.backward()
    assert total==pytest.approx(float(result.total.detach()),abs=1e-6)
    assert torch.allclose(policy.grad,split_policy.grad,atol=1e-7)
    assert torch.allclose(value.grad,split_value.grad,atol=1e-7)


def test_wdl_targets_and_model_without_value_head():
    assert [result_target('1-0',t) for t in ('r','b')]==[0,2]
    assert [result_target('0-1',t) for t in ('r','b')]==[2,0]
    assert result_target('1/2-1/2','b')==1 and result_target('*','r')==-100
    data=fake_batch(1,2)
    logits=torch.zeros(1,2,8100,requires_grad=True)
    with pytest.raises(ValueError,match='没有价值头'): mixed_loss(logits,data,LossWeights.selfplay())
    assert torch.isfinite(mixed_loss(logits,data,LossWeights.selfplay(False)).total)


@pytest.fixture
def rollout(tmp_path):
    torch.manual_seed(27)
    model=tiny_opus()
    db=tmp_path/'parameters.sqlite3'
    ident=save_checkpoint(model,path=db,run_id='init',epoch=0,global_step=0,metrics={},
                         version_name='Opus-test.initial',description='test initial')
    generator=SelfPlayGenerator.from_version('Opus','Opus-test.initial',model_db=db,
                   rules=SelfPlayRules(max_plies=5),exploration=Exploration(1.2,.02))
    data=tmp_path/'selfplay.sqlite3'
    report=generator.generate(2,path=data)
    return model,db,ident,data,report


def test_generator_freezes_snapshot_pins_version_and_replays_prefix_exactly(rollout,assert_rollout_replay):
    model,db,ident,path,report=rollout
    generator=SelfPlayGenerator.from_version('Opus',ident,model_db=db,
                   rules=SelfPlayRules(max_plies=5),exploration=Exploration(1.2,.02))
    report=generator.generate(2,path=path,prefix=[move_to_action((6,0,5,0))])
    assert len(report['game_ids'])==2 and model.training and all(p.requires_grad for p in model.parameters())
    ds=SelfPlayDataset(path,report['run_id'],family='Opus')
    sample=ds[0]; batch=collate_examples([sample])
    assert batch.loss.counts()==LossCounts(0,5,5,5)
    assert batch.loss.selfplay_mask[0,0] and sample.length==5 and len(sample.x)==128
    loaded,meta,_=load_checkpoint(db,ident); loaded.eval()
    assert_rollout_replay(loaded,path,sample)
    game,run,steps=load_game(path,ds.ids[0])
    assert game['result']=='1/2-1/2' and game['termination']=='max_plies'
    assert run['checkpoint_id']==ident and run['version_name']=='Opus-test.initial'
    assert run['weights_sha256']==meta['sha256'] and run['has_value_head']==1
    assert all(s['old_log_mu']<=0 for s in steps)
    with pytest.raises(ValueError,match='版本'): load_version('Opus','missing',path=db)


def test_selfplay_stores_separate_families_but_share_versions(rollout):
    model,db,ident,path,report=rollout
    second=save_checkpoint(model,path=db,run_id='second',epoch=1,global_step=1,metrics={},version_name='Opus-test.1')
    other=SelfPlayGenerator.from_version('Opus',second,model_db=db,rules=SelfPlayRules(max_plies=1))
    other.generate(1,path=path)
    assert {r['checkpoint_id'] for r in list_runs(path)}=={ident,second}
    before=path.read_bytes()
    with pytest.raises(ValueError,match='系列'): init_store('HaiKu',path)
    assert before==path.read_bytes()
    before=db.read_bytes()
    with pytest.raises(ValueError,match='不是自弈'): init_store('Opus',db)
    assert before==db.read_bytes()


def test_haiKu_zero_baseline_and_resource_interruption_is_unknown(tmp_path):
    net=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    db=tmp_path/'params.sqlite3'
    ident=save_checkpoint(net,path=db,run_id='h',epoch=1,global_step=1,metrics={},version_name='HaiKu-test')
    generator=SelfPlayGenerator.from_version('HaiKu',ident,model_db=db,rules=SelfPlayRules(max_plies=8))
    class Cancel:
        calls=0
        def is_set(self):
            self.calls+=1; return self.calls>3
    path=tmp_path/'rollouts.sqlite3'
    report=generator.generate(1,path=path,cancel_event=Cancel())
    assert report['game_ids']==[] and report['interrupted'] and report['unfinished_game_discarded']
    assert list_runs(path)[0]['saved_games']==0 and not list_runs(path)[0]['has_value_head']
    complete=generator.generate(1,path=tmp_path/'complete.sqlite3')
    game,run,steps=load_game(complete['data_db'],complete['game_ids'][0])
    assert all(s['old_value']==0. for s in steps)
    # Unknown imported/audited outcomes must still be excluded from terminal-return losses.
    with sqlite3.connect(complete['data_db']) as conn:
        conn.execute("UPDATE games SET result='*' WHERE game_id=?",(complete['game_ids'][0],))
    data=SelfPlayDataset(complete['data_db'],complete['run_id'])[0]
    batch=collate_examples([data]); assert batch.loss.counts().ppo==batch.loss.counts().value==0
    optimizer=torch.optim.AdamW(net.parameters())
    before=copy.deepcopy(net.state_dict())
    report=train_logical_batch(net,optimizer,[batch],LossWeights.selfplay(False))
    assert not report['updated'] and all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    # A complete HaiKu game can train policy with the same interface and zero baseline.
    complete=generator.generate(1,path=tmp_path/'complete-known.sqlite3')
    example=SelfPlayDataset(complete['data_db'],complete['run_id'])[0]
    meta=load_checkpoint(db,ident)[1]
    learned=train_logical_batch(net,optimizer,[collate_examples([example])],LossWeights.selfplay(False),
                                chunk_size=2,behavior_checkpoint_id=ident,behavior_sha256=meta['sha256'])
    assert learned['updated'] and math.isfinite(learned['loss'])


def test_logical_batch_updates_once_and_rejects_other_behavior_version(rollout):
    model,db,ident,path,report=rollout
    ds=SelfPlayDataset(path,report['run_id']); batches=[collate_examples([ds[i]]) for i in range(2)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    meta=load_checkpoint(db,ident)[1]
    with pytest.raises(ValueError,match='采样参数版本'):
        train_logical_batch(model,optimizer,batches,LossWeights.selfplay(),behavior_checkpoint_id='wrong',behavior_sha256=meta['sha256'])
    old_value=copy.deepcopy(model.value_head.state_dict())
    result=train_logical_batch(model,optimizer,batches,LossWeights.selfplay(),chunk_size=2,
                               behavior_checkpoint_id=ident,behavior_sha256=meta['sha256'])
    assert result['updated'] and result['counts']=={'book':0,'ppo':10,'value':10,'entropy':10}
    assert result['ratio_mean']==pytest.approx(1.,abs=2e-5)
    assert all(int(s['step'])==1 for s in optimizer.state.values())
    assert any(not torch.equal(v,model.value_head.state_dict()[k]) for k,v in old_value.items())


def test_rules_distinguish_repetition_change_quiet_limit_and_step_cap():
    state=Position(STANDARD_FEN,SelfPlayRules(max_plies=1))
    state.move(move_to_action((6,0,5,0)),state.terminal(0)[1])
    assert state.terminal(1)[0]==('1/2-1/2','max_plies')
    state=Position(STANDARD_FEN,SelfPlayRules(100,1))
    state.move(move_to_action((6,0,5,0)),state.terminal(0)[1])
    assert state.terminal(1)[0]==('1/2-1/2','quiet_limit')
    state=Position(STANDARD_FEN,SelfPlayRules())
    moves=((9,1,7,2),(0,1,2,2),(7,2,9,1),(2,2,0,1))*2
    for move in moves: state.move(move_to_action(move),state.terminal(0)[1])
    terminal,legal=state.terminal(8)
    assert terminal is None and not legal[move_to_action(moves[0])]


@pytest.mark.parametrize('C',[2,128])
def test_mixed_cli_roundtrip_saves_new_version_without_overwriting_base(rollout,tmp_path,C):
    from training.__main__ import main
    from data_pipeline.database import connect
    model,db,ident,path,report=rollout
    book=tmp_path/'books.sqlite3'; conn=connect(book)
    conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                 ('book',STANDARD_FEN,json.dumps([move_to_action((6,0,5,0))]),1,'*','human_master','train','book','{}'))
    conn.commit(); conn.close()
    base_sha=load_checkpoint(db,ident)[1]['sha256']
    result=main(['--family','Opus','--version','Opus-test.initial','--model-db',str(db),
                 '--source','mixed','--book-db',str(book),'--selfplay-db',str(path),
                 '--rollout-run',report['run_id'],'--T','2','--C',str(C),'--epochs','1','--batch-size','3','--micro-batch','1',
                 '--tbptt-steps','2','--dropout','0.25','--output',str(tmp_path/'runs'),'--device','cpu','--cpu-threads','2','--smoke'])
    updates=math.ceil((1+2*math.ceil(5/C))/3)
    assert result['global_step']==updates and len(list_versions(db))==2
    assert load_checkpoint(db,ident)[1]['sha256']==base_sha
    trained,meta,state=load_checkpoint(db,result['checkpoints'][0],resume=True)
    assert meta['stage']=='mixed' and not meta['published'] and trained.config.tbptt_steps==2
    assert trained.config.dropout==.25
    assert state['behavior_checkpoint_id']==ident and state['rollout_run']==report['run_id']
    assert state['C']==C
    config=json.loads((tmp_path/'runs'/result['run_id']/'config.json').read_text(encoding='utf-8'))
    assert config['T']==2 and config['C']==C and config['selfplay_games']==2
    assert config['dropout']==config['model']['dropout']==.25
    assert config['selfplay_segments']==2*math.ceil(5/C)
    metrics=[json.loads(line) for line in (tmp_path/'runs'/result['run_id']/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(m['counts']['ppo'] for m in metrics)==10
    assert sum(m['counts']['book'] for m in metrics)==1
    indices=list(range(config['training_samples']));random.Random(43).shuffle(indices)
    value_updates=sum(any(i>0 for i in indices[start:start+3]) for start in range(0,len(indices),3))
    # A book-only batch with an unknown outcome updates the policy path only.
    parameter_ids=state['optimizer']['param_groups'][0]['params']
    for (name,_),parameter_id in zip(trained.named_parameters(),parameter_ids):
        expected=value_updates if name.startswith(('cores.3.','cores.4.','bridge_','value_head.')) else updates
        assert int(state['optimizer']['state'][parameter_id]['step'])==expected

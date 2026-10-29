import copy
from dataclasses import replace
import json
import sqlite3

import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(2)
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5, SingleHeadCross, AttentionPolicyHead
from models.Opus.loss import masked_value_ce
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from model_store import (save_checkpoint,load_checkpoint,list_versions,load_default_model,
    set_default_checkpoint,default_model_name,publish_model,list_hint_models,migrate_legacy_store)
from ai import NeuralAI,load_network
from board import Board
from encoder import legal_action_mask,move_to_action


def tiny(recompute=False):
    return Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(1,2,3,2,3),
                           ffn_dim=32,cnn_channels=8,value_hidden=8,tbptt_steps=2,
                           checkpoint_attention=recompute))


def test_default_spec_and_exact_branch_wiring():
    cfg=OpusConfig()
    assert cfg.heads==(16,8,4,8,4) and cfg.context_windows==(10,20,30,20,30)
    assert cfg.tbptt_steps==16
    net=tiny().eval()
    inputs,outputs={},{}
    hooks=[]
    for i,core in enumerate(net.cores):
        def capture(module,args,out,index=i):
            inputs[index]=args[0]; outputs[index]=out
        hooks.append(core.register_forward_hook(capture))
    bridges={}
    for name in ('bridge_a','bridge_b'):
        def capture_bridge(module,args,out,key=name): bridges[key]=(args,out)
        hooks.append(getattr(net,name).register_forward_hook(capture_bridge))
    with torch.no_grad(): out,cache=net.forward_step(torch.randn(1,15,10,9))
    for hook in hooks: hook.remove()
    assert inputs[0] is inputs[3]
    assert inputs[1] is outputs[0] and inputs[2] is outputs[1]
    assert bridges['bridge_a'][0][0] is outputs[3]
    assert bridges['bridge_a'][0][1] is outputs[1]
    assert inputs[4] is bridges['bridge_a'][1]
    assert bridges['bridge_b'][0][0] is outputs[4]
    assert bridges['bridge_b'][0][1] is outputs[2]
    assert torch.equal(out.policy_logits,net.policy_head(outputs[2]))
    assert torch.equal(out.value_logits,net.value_head(bridges['bridge_b'][1].mean(1)))
    assert sum(len(c.history)+1 for c in cache.cores)==15
    assert not any('cache' in key for key in net.state_dict())


def test_incremental_padding_causality_and_independent_windows():
    torch.manual_seed(72)
    net=tiny().eval(); x=torch.randn(5,2,15,10,9); lengths=torch.tensor([5,1])
    with torch.inference_mode():
        seq=net(x,lengths); cache=None; steps=[]
        for t in range(5):
            out,cache=net.forward_step(x[t],cache,t<lengths); steps.append(out)
            assert cache.time==t+1
            for i,core in enumerate(cache.cores):
                for kv in [*core.history,core.cross]:
                    assert len(kv.keys)==min(t+1,net.config.context_windows[i])
                    assert kv.keys[-1].shape==(2,net.config.heads[i],90,16//net.config.heads[i])
        assert torch.allclose(seq.policy_logits,torch.stack([s.policy_logits for s in steps]),atol=1e-6)
        assert torch.allclose(seq.value_logits,torch.stack([s.value_logits for s in steps]),atol=1e-6)
        assert torch.allclose(seq.value_probabilities[:,0].sum(-1),torch.ones(5))
        assert not seq.value_probabilities[1:,1].count_nonzero()
        changed=x.clone(); changed[-1,0]+=20; changed[1:,1]*=100
        other=net(changed,lengths)
        assert torch.allclose(seq.policy_logits[:-1,0],other.policy_logits[:-1,0],atol=1e-6)
        assert torch.allclose(seq.value_logits[:,1],other.value_logits[:,1],atol=1e-6)
        weights=copy.deepcopy(net.state_dict())
        net.set_context_windows((1,1,1,1,1))
        _,cache=net.forward_step(x[0],cache)
        assert cache.time==6 and all(len(c.cross.keys)==1 for c in cache.cores)
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in weights.items())


def test_single_head_bridges_and_policy_use_specified_equations():
    x,y=torch.randn(2,90,16),torch.randn(2,90,16)
    cross=SingleHeadCross(16)
    expected=(cross.q(x)@cross.k(y).transpose(-2,-1)/4).softmax(-1)@cross.v(y)
    actual=cross(x,y,torch.tensor([True,False]))
    assert torch.allclose(actual[0],expected[0],atol=2e-6)
    assert not actual[1].count_nonzero()
    policy=AttentionPolicyHead(16)
    assert torch.equal(policy(x),(policy.q(x)@policy.k(x).transpose(-2,-1)/4).flatten(1))


def test_value_loss_reaches_both_branches_policy_only_reaches_policy_branch():
    net=tiny().train(); x=torch.randn(2,1,15,10,9)
    out=net(x)
    loss,_=masked_value_ce(out.value_logits,torch.tensor([[0],[2]]),torch.tensor([2]))
    loss.backward()
    for core in net.cores: assert core.history[0].attn.q.weight.grad.abs().sum()>0
    assert net.bridge_a.v.weight.grad.abs().sum()>0 and net.bridge_b.v.weight.grad.abs().sum()>0
    assert net.policy_head.q.weight.grad is None
    net.zero_grad(set_to_none=True)
    net(x).policy_logits[1,0,42].backward()
    assert all(net.cores[i].cross.attn.q.weight.grad is not None for i in range(3))
    assert all(p.grad is None for core in net.cores[3:] for p in core.parameters())
    assert all(p.grad is None for p in net.value_head.parameters())


def test_chunk_detach_keeps_forward_memory_but_stops_cross_chunk_gradients():
    torch.manual_seed(89)
    net=tiny().train(); net.set_context_windows((4,4,4,4,4))
    x=torch.randn(4,1,15,10,9,requires_grad=True); lengths=torch.tensor([4])
    torch.manual_seed(1234)
    full=net(x)
    full.value_logits[-1,0,0].backward()
    assert x.grad[0].abs().sum()>0
    net.zero_grad(set_to_none=True); x.grad=None
    torch.manual_seed(1234)
    chunks=net.forward_chunks(x,lengths)
    first=next(chunks); second=next(chunks)
    assert first[:2]==(0,2) and second[:2]==(2,4)
    assert torch.allclose(full.policy_logits,torch.cat([first[2].policy_logits,second[2].policy_logits]),atol=1e-6)
    assert torch.allclose(full.value_logits,torch.cat([first[2].value_logits,second[2].value_logits]),atol=1e-6)
    second[2].value_logits[-1,0,0].backward()
    assert not x.grad[:2].count_nonzero() and x.grad[2:].abs().sum()>0
    # Detach retained data: replacing the early boards still changes later outputs.
    changed=x.detach().clone(); changed[:2]+=3
    assert not torch.allclose(full.value_logits[-1],net(changed).value_logits[-1])


def test_recompute_and_chunk_backward_accumulation_match():
    torch.manual_seed(93)
    a,b=tiny(True).train(),tiny(False).train(); b.load_state_dict(a.state_dict())
    x=torch.randn(5,1,15,10,9); lengths=torch.tensor([5])
    for net in (a,b):
        torch.manual_seed(1234)
        for start,stop,out in net.forward_chunks(x,lengths):
            loss=out.policy_logits[...,17].mean()+out.value_logits[...,0].mean()
            loss.backward()
    for p,q in zip(a.parameters(),b.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.allclose(p.grad,q.grad,atol=3e-5,rtol=2e-4)


def test_value_loss_ignores_unknown_outcomes_and_padding():
    logits=torch.randn(3,2,3,requires_grad=True)
    target=torch.tensor([[0,-100],[2,1],[1,100]])
    loss,tb=masked_value_ce(logits,target,torch.tensor([3,2]))
    expected=torch.nn.functional.cross_entropy(logits[torch.tensor([[1,0],[1,1],[1,0]],dtype=torch.bool)],
                                               torch.tensor([0,2,1,1]))
    assert torch.allclose(loss,expected) and tb[0,1]==tb[2,1]==0
    loss.backward(); assert not logits.grad[0,1].count_nonzero()
    unknown=torch.full((3,2),-100)
    zero,_=masked_value_ce(logits,unknown,torch.tensor([3,2]))
    assert zero==0 and torch.isfinite(zero)


def test_family_stores_metadata_defaults_and_legacy_migration(tmp_path):
    root=tmp_path/'parameters'; root.mkdir()
    haiku=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8))
    opus=tiny()
    hid=save_checkpoint(haiku,path=root,run_id='h',epoch=2,global_step=2,metrics={},
                         version_name='HaiKu-5.1',description='第二轮权重',set_default=True)
    oid=save_checkpoint(opus,path=root,run_id='o',epoch=0,global_step=0,metrics={},
                         version_name='Opus-5.initial',description='未训练初始化',set_default=True)
    assert (root/'HaiKu/models.sqlite3').exists() and (root/'Opus/models.sqlite3').exists()
    loaded=load_default_model('Opus',path=root)
    assert loaded.config==opus.config
    assert all(torch.equal(p,q) for p,q in zip(loaded.parameters(),opus.parameters()))
    versions={r['checkpoint_id']:r for r in list_versions(root)}
    assert versions[oid]['description']=='未训练初始化' and versions[oid]['is_default']
    assert versions[hid]['version_name']=='HaiKu-5.1'
    publish_model('HaiKu-5.1',hid,path=root)
    assert list_hint_models(root)==[{'name':'HaiKu-5.1'}]
    assert default_model_name(root)=='HaiKu-5.1' and load_network(root).display_name=='HaiKu-5.1'
    with pytest.raises(ValueError,match='未训练'): publish_model('Opus-5',oid,path=root)
    before=(root/'HaiKu/models.sqlite3').read_bytes()
    with pytest.raises(ValueError,match='系列'):
        save_checkpoint(opus,path=root/'HaiKu/models.sqlite3',run_id='bad',epoch=1,global_step=1,metrics={})
    assert before==(root/'HaiKu/models.sqlite3').read_bytes()
    destination=tmp_path/'copy/HaiKu/models.sqlite3'
    assert migrate_legacy_store(root/'HaiKu/models.sqlite3',destination)
    assert before==(root/'HaiKu/models.sqlite3').read_bytes()
    _,oldmeta,_=load_checkpoint(root,hid); _,newmeta,_=load_checkpoint(destination,hid)
    assert oldmeta['sha256']==newmeta['sha256']
    assert list_hint_models(destination)==[{'name':'HaiKu-5.1'}]
    assert not migrate_legacy_store(root/'HaiKu/models.sqlite3',destination)
    newer=save_checkpoint(haiku,path=root,run_id='h',epoch=3,global_step=3,metrics={})
    publish_model('HaiKu-5.2',newer,path=root); set_default_checkpoint(newer,path=root)
    assert load_network(root).display_name=='HaiKu-5.2'
    assert load_default_model('Opus',path=root).config.architecture=='Opus-5'
    with pytest.raises(ValueError,match='系列'):
        load_default_model('Opus',path=root/'HaiKu/models.sqlite3')


def test_opus_neural_ai_returns_legal_moves():
    board=Board(); ai=NeuralAI(tiny())
    ranked=ai.rank_moves(board,'r',topk=3)
    mask=legal_action_mask(board,'r')
    assert len(ranked)==3 and all(mask[move_to_action(move)] for move,p in ranked)
    assert all(0<p<=1 for move,p in ranked)


def test_opus_training_chunks_and_saved_configuration(tmp_path,monkeypatch):
    import train
    from data_pipeline.database import connect
    from data_pipeline.notation import STANDARD_FEN
    data=tmp_path/'data.sqlite3'; conn=connect(data)
    actions=[move_to_action(m) for m in ((6,0,5,0),(3,0,4,0),(6,2,5,2))]
    for split in ('train','validation','test'):
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                     (split,STANDARD_FEN,json.dumps(actions),3,'*','human_master',split,split,'{}'))
    conn.commit(); conn.close()
    monkeypatch.setattr(train,'Opus5',lambda cfg: Opus5(replace(cfg,dim=16,ffn_dim=32,cnn_channels=8,value_hidden=8)))
    root=tmp_path/'parameters'; root.mkdir(); output=tmp_path/'runs'
    train.main(['--architecture','Opus','--model-db',str(root),'--data',str(data),
                '--output',str(output),'--run-name','opus-smoke','--epochs','1','--T','5',
                '--tbptt-steps','2','--context-windows','1','2','3','2','3',
                '--dropout','0.3','--batch-size','1','--micro-batch','1','--device','cpu','--precision','none','--smoke'])
    saved=next(r for r in list_versions(root) if r['architecture']=='Opus-5')
    model,meta,state=load_checkpoint(root,saved['checkpoint_id'],resume=True)
    assert model.config.tbptt_steps==2 and model.config.context_windows==(1,2,3,2,3)
    assert model.config.dropout==.3
    assert state['T']==5 and state['tbptt_steps']==2 and meta['global_step']==1
    # Only one optimizer update after both graph chunks; no value labels in books.
    assert all(int(s['step'])==1 for s in state['optimizer']['state'].values())
    assert list_hint_models(root)==[]
    assert (output/'opus-smoke/evaluation.json').exists()

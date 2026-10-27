import copy
import sqlite3
import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(2)
from ai import NeuralAI
from board import Board
from encoder import encode_state, legal_action_mask, move_to_action
from game import Game
from models.haiku5.config import Haiku5Config
from models.haiku5.network import Haiku5
from models.haiku5.loss import masked_policy_ce
from model_store import save_checkpoint, load_checkpoint, list_models, resolve_alias


def tiny(recompute=True):
    return Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,
                               context_plies=4,checkpoint_attention=recompute))


def test_sequence_matches_incremental_no_future_and_no_batch_leakage():
    torch.manual_seed(13)
    net = tiny().eval()
    x = torch.randn(3,2,15,10,9)
    lengths = torch.tensor([3,1])
    with torch.no_grad():
        out = net(x,lengths)
        cache,steps = None,[]
        for t in range(3):
            y,cache = net.forward_step(x[t],cache,t<lengths); steps.append(y)
        assert torch.allclose(out,torch.stack(steps),atol=1e-6)
        changed = x.clone(); changed[2,0] += 10; changed[1:,1] = torch.randn_like(changed[1:,1])*100
        out2 = net(changed,lengths)
        assert torch.allclose(out[:2,0],out2[:2,0],atol=1e-6)
        assert torch.allclose(out[:,1],out2[:,1],atol=1e-6)
        assert not out[1:,1].count_nonzero()
        assert torch.allclose(out,net(x,lengths),atol=1e-6)
        assert len(cache.cores)==3 and all(len(c.history)==2 for c in cache.cores)
        assert cache.cores[0].history[0].keys[-1].shape==(2,4,90,4)
        assert sum(k.shape[2] for k in cache.cores[0].cross.keys)==270
        assert not any('cache' in key for key in net.state_dict())


@pytest.mark.parametrize('window',[2,4])
def test_recompute_gradients_match_and_late_loss_reaches_history(window):
    torch.manual_seed(17)
    a,b = tiny(True).train(),tiny(False).train()
    a.set_context_plies(window); b.set_context_plies(window)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3,1,15,10,9,requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    a(x)[-1,0,123].backward(); b(z)[-1,0,123].backward()
    assert x.grad[0].abs().sum()>0
    assert torch.allclose(x.grad,z.grad,atol=2e-6,rtol=2e-5)
    for p,q in zip(a.parameters(),b.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.allclose(p.grad,q.grad,atol=2e-5,rtol=1e-4)


@pytest.mark.parametrize('window',[1,2,8])
def test_kv_window_is_independent_of_sequence_length(window):
    torch.manual_seed(31)
    net=tiny().eval()
    net.set_context_plies(window)
    x=torch.randn(6,1,15,10,9)
    with torch.inference_mode():
        expected=net(x)
        cache,outputs=None,[]
        for t in range(len(x)):
            logits,cache=net.forward_step(x[t],cache)
            outputs.append(logits)
            assert cache.time==t+1
            for core in cache.cores:
                for kv in [*core.history,core.cross]:
                    assert len(kv.keys)==len(kv.values)==len(kv.valid)==min(t+1,window)
        assert expected.shape==(6,1,8100)
        assert torch.allclose(expected,torch.stack(outputs),atol=1e-6)


def test_changing_window_trims_existing_caches_without_resetting_time_or_weights():
    net=tiny().eval()
    weights={key:value.clone() for key,value in net.state_dict().items()}
    with torch.inference_mode():
        cache=None
        for _ in range(4):
            _,cache=net.forward_step(torch.randn(1,15,10,9),cache)
        net.set_context_plies(1)
        _,cache=net.forward_step(torch.randn(1,15,10,9),cache)
    assert net.config.context_plies==1 and cache.time==5
    for core in cache.cores:
        assert all(len(kv.keys)==len(kv.values)==len(kv.valid)==1 for kv in [*core.history,core.cross])
    assert all(torch.equal(value,net.state_dict()[key]) for key,value in weights.items())
    with pytest.raises(ValueError,match='正数'):
        net.set_context_plies(0)
    assert net.config.context_plies==1


def test_masked_loss_padding_dense_labels_and_invalid_targets():
    logits = torch.randn(3,2,8100,requires_grad=True)
    legal = torch.zeros_like(logits,dtype=torch.bool)
    legal[0,:,10:13]=True; legal[1,0,10:13]=True
    lengths=torch.tensor([2,1]); y=torch.tensor([[10,11],[12,-100],[-100,-100]])
    loss,tb,p = masked_policy_ce(logits,y,legal,lengths)
    assert tb.shape==(3,2) and tb[1,1]==0 and not tb[2].count_nonzero()
    assert torch.allclose(loss,tb.sum()/3)
    assert torch.allclose(p.sum(-1),torch.tensor([[1.,1.],[1.,0.],[0.,0.]]))
    assert not p[~legal].count_nonzero()
    dense=torch.zeros_like(logits); dense.scatter_(-1,y.clamp_min(0)[...,None],1)
    dense[y<0]=0
    assert torch.allclose(loss,masked_policy_ce(logits,dense,legal,lengths)[0])
    loss.backward(); assert torch.isfinite(logits.grad).all()
    assert not logits.grad[~legal].count_nonzero()
    bad=y.clone(); bad[0,0]=25
    with pytest.raises(ValueError,match='非法动作'): masked_policy_ce(logits,bad,legal,lengths)
    zero,_,p=masked_policy_ce(logits,torch.full_like(y,-100),torch.zeros_like(legal),torch.zeros(2,dtype=torch.long))
    assert zero==0 and not p.count_nonzero()


def test_sqlite_roundtrip_resume_aliases_and_hash(tmp_path):
    torch.manual_seed(4)
    net=tiny(); optimizer=torch.optim.AdamW(net.parameters())
    x=torch.randn(1,15,10,9)
    net(x).sum().backward(); optimizer.step()
    path=tmp_path/'models.sqlite3'
    ident=save_checkpoint(net,path=path,run_id='run',epoch=1,global_step=1,metrics={'loss':1.2},
                          optimizer=optimizer,best=True)
    loaded,meta,state=load_checkpoint(path,ident,resume=True)
    assert torch.allclose(net.eval()(x),loaded(x),atol=1e-6)
    assert state['optimizer']['state'] and meta['architecture']=='Haiku-5'
    assert resolve_alias(path,'run')==ident and len(list_models(path))==1
    save_checkpoint(net,path=path,run_id='smoke',epoch=1,global_step=1,metrics={},published=False)
    assert len(list_models(path))==1
    with sqlite3.connect(path) as c: c.execute('UPDATE checkpoints SET sha256=? WHERE checkpoint_id=?',('bad',ident))
    with pytest.raises(ValueError,match='校验失败'): load_checkpoint(path,ident)


def test_hint_top_three_matches_full_softmax_and_undo_rebuilds():
    torch.manual_seed(8); net=tiny().eval(); ai=NeuralAI(net)
    game=Game()
    assert game.move(6,0,5,0)=='ok'; assert game.move(3,0,4,0)=='ok'
    ranked=ai.rank_moves(game.board,game.turn,history=game.history)
    states=[encode_state(Board(h['board_state']),h['prev_turn']) for h in game.history]
    states.append(encode_state(game.board,game.turn))
    with torch.no_grad(): logits=net(torch.from_numpy(np.stack(states))[:,None])[-1,0]
    mask=torch.from_numpy(legal_action_mask(game.board,game.turn))
    probs=logits.masked_fill(~mask,-torch.inf).softmax(-1)
    assert [move_to_action(m) for m,_ in ranked]==probs.topk(3).indices.tolist()
    assert np.allclose([v for _,v in ranked],probs.topk(3).values.numpy())
    assert ai._cache.time==3
    assert ai.rank_moves(game.board,game.turn,game.history)==ranked and ai._cache.time==3
    game.undo()
    reset_rank=NeuralAI(net).rank_moves(game.board,game.turn,game.history)
    assert ai.rank_moves(game.board,game.turn,game.history)==reset_rank


def test_microbatch_gradients_equal_joint_valid_step_mean():
    torch.manual_seed(22)
    together,separate=tiny(False),tiny(False)
    separate.load_state_dict(together.state_dict())
    x=torch.randn(2,2,15,10,9)
    lengths=torch.tensor([2,1]); y=torch.tensor([[3,5],[4,-100]])
    legal=torch.ones(2,2,8100,dtype=torch.bool)
    masked_policy_ce(together(x,lengths),y,legal,lengths)[0].backward()
    for b in range(2):
        z=separate(x[:,b:b+1],lengths[b:b+1])
        loss=masked_policy_ce(z,y[:,b:b+1],legal[:,b:b+1],lengths[b:b+1],'sum')[0]
        (loss/4).backward()
    for p,q in zip(together.parameters(),separate.parameters()):
        q.grad.mul_(4/int(lengths.sum()))
        assert torch.allclose(p.grad,q.grad,atol=2e-5,rtol=2e-4)

"""Residual dropout placement, training modes and old checkpoint compatibility."""
import copy
import json
import sqlite3

import pytest

torch = pytest.importorskip('torch')
from torch import nn
from torch.nn import functional as F
from models.common import Block
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from models.registry import build_model
from model_store import save_checkpoint, load_checkpoint
from training.data import TrainingBatch
from training.engine import train_logical_batch
from training.loss import LossBatch, LossWeights, split_output


def tiny(family, **options):
    common = dict(dim=16, ffn_dim=32, cnn_channels=8, checkpoint_attention=False, **options)
    if family == 'HaiKu':
        return Haiku5(Haiku5Config(heads=4, context_plies=2, **common))
    return Opus5(OpusConfig(heads=(4,2,1,2,1), context_windows=(2,2,2,2,2),
                           value_hidden=8, **common))


@pytest.mark.parametrize('family,total', [('HaiKu',30),('Opus',50)])
def test_layer_counts_modes_and_parameterless_probability_updates(family,total):
    model=tiny(family)
    layers=[m for m in model.modules() if isinstance(m,nn.Dropout)]
    assert len(layers)==total and all(m.p==.1 for m in layers)
    for core in model.cores:
        assert sum(isinstance(m,nn.Dropout) for m in core.modules())==10
        for block in (*core.history,*core.spatial,core.cross):
            assert isinstance(block.attn_dropout,nn.Dropout) and isinstance(block.ffn_dropout,nn.Dropout)
    parameters=list(model.named_parameters()); state=copy.deepcopy(model.state_dict())
    model.set_dropout(.4)
    assert model.config.dropout==.4 and all(m.p==.4 for m in layers)
    assert [(name,id(p)) for name,p in parameters]==[(name,id(p)) for name,p in model.named_parameters()]
    assert not any('dropout' in name for name in state)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in state.items())
    x=torch.randn(1,15,10,9)
    model.train()
    a=split_output(model(x))[0]; b=split_output(model(x))[0]
    assert not torch.equal(a,b)
    model.eval()
    with torch.inference_mode():
        assert torch.equal(split_output(model(x))[0],split_output(model(x))[0])
    model.set_dropout(0); model.train()
    assert torch.equal(split_output(model(x))[0],split_output(model(x))[0])


def test_dropout_is_after_attention_activation_and_ffn_before_each_residual():
    block=Block(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8,dropout=.4)).train()
    captured={}
    def remember(name):
        def hook(module,args,output): captured[name]=(args[0],output)
        return hook
    hooks=[getattr(block,name).register_forward_hook(remember(name))
           for name in ('attn','attn_dropout','norm1','ffn','ffn_dropout','norm2')]
    x=torch.randn(2,90,16); valid=torch.tensor([True,False])
    actual=block(x,valid)
    for hook in hooks: hook.remove()
    assert torch.equal(captured['attn_dropout'][0],F.gelu(captured['attn'][1]))
    assert torch.equal(captured['norm1'][0],x+captured['attn_dropout'][1])
    assert torch.equal(captured['ffn_dropout'][0],captured['ffn'][1])
    assert torch.equal(captured['norm2'][0],captured['norm1'][1]+captured['ffn_dropout'][1])
    assert torch.equal(actual[0],captured['norm2'][1][0]) and not actual[1].count_nonzero()
    block.attn_dropout.p=block.ffn_dropout.p=1
    assert torch.equal(block(x,valid)[0],block.norm2(block.norm1(x))[0])


@pytest.mark.parametrize('family',['HaiKu','Opus'])
def test_old_config_strict_weights_optimizer_load_and_eval_equivalence(family,tmp_path,monkeypatch):
    model=tiny(family).train(); x=torch.randn(1,15,10,9)
    optimizer=torch.optim.AdamW(model.parameters())
    split_output(model(x))[0].square().mean().backward(); optimizer.step()
    database=tmp_path/'parameters.sqlite3'
    ident=save_checkpoint(model,path=database,run_id='old',epoch=1,global_step=1,
                          metrics={},optimizer=optimizer)
    old_config=model.config.to_dict(); old_config.pop('dropout')
    with sqlite3.connect(database) as conn:
        conn.execute('UPDATE checkpoints SET config_json=? WHERE checkpoint_id=?',
                     (json.dumps(old_config),ident))
    loaded,_,state=load_checkpoint(database,ident,resume=True)
    assert loaded.config.dropout==.1 and not loaded.training
    assert list(model.state_dict())==list(loaded.state_dict())
    # The loader already uses strict=True; also verify the registry accepts old configs.
    restored=build_model(old_config)
    assert not restored.load_state_dict(model.state_dict(),strict=True).missing_keys
    resumed=torch.optim.AdamW(loaded.parameters()); resumed.load_state_dict(state['optimizer'])
    for original,current in zip(optimizer.param_groups[0]['params'],resumed.param_groups[0]['params']):
        if not optimizer.state[original]:
            assert not resumed.state[current]
            continue
        assert torch.equal(optimizer.state[original]['exp_avg'],resumed.state[current]['exp_avg'])
        assert torch.equal(optimizer.state[original]['exp_avg_sq'],resumed.state[current]['exp_avg_sq'])
    sequence=torch.randn(2,1,15,10,9)
    with torch.inference_mode(): actual=split_output(loaded(sequence))
    def legacy_forward(self,x,valid,cache=None,query=None,source=None):
        a=self.attn(x if query is None else query,x if source is None else source,valid,cache)
        h=self.norm1(x+F.gelu(a))
        return self.norm2(h+self.ffn(h)).masked_fill(~valid[:,None,None],0)
    monkeypatch.setattr(Block,'forward',legacy_forward)
    with torch.inference_mode(): expected=split_output(loaded(sequence))
    for before,after in zip(expected,actual):
        if before is not None: assert torch.equal(before,after)


@pytest.mark.parametrize('config',[Haiku5Config,OpusConfig])
@pytest.mark.parametrize('probability',[-.1,1.1,float('nan'),float('inf'),True,'0.1'])
def test_reject_invalid_probability(config,probability):
    with pytest.raises(ValueError,match='dropout'): config(dropout=probability)


@pytest.mark.parametrize('ppo',[False,True])
def test_supervised_training_enables_dropout_but_ppo_preserves_rollout_probabilities(ppo):
    model=tiny('HaiKu').eval(); length=torch.tensor([1])
    legal=torch.ones(1,1,8100,dtype=torch.bool); action=torch.zeros(1,1,dtype=torch.long)
    x=torch.randn(1,1,15,10,9)
    with torch.inference_mode(): old=model(x).log_softmax(-1)[...,0]
    if ppo:
        loss=LossBatch(action,legal,length,selfplay_mask=torch.ones(1,1,dtype=torch.bool),
                       value_targets=torch.zeros(1,1,dtype=torch.long),old_log_mu=old,
                       old_value=torch.zeros(1,1),temperature=torch.ones(1,1),epsilon=torch.zeros(1,1))
        weights=LossWeights.selfplay(False); provenance=frozenset({('id','sha','Haiku-5')})
    else:
        loss=LossBatch.books(action,legal,length); weights=LossWeights.books(False); provenance=frozenset()
    modes=[]
    hook=model.cores[0].cross.attn_dropout.register_forward_hook(lambda module,args,out:modes.append(module.training))
    report=train_logical_batch(model,torch.optim.AdamW(model.parameters()),[TrainingBatch(x,loss,provenance)],
                               weights,behavior_checkpoint_id='id',behavior_sha256='sha')
    hook.remove()
    assert report['updated'] and modes==[not ppo] and model.training is not ppo
    if ppo: assert report['ratio_mean']==pytest.approx(1.,abs=2e-5)

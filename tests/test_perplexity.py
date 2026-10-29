import copy
import json
import math
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')

from training.evaluation import evaluate_books,validate_evaluation
from training.perplexity import perplexity_histogram,save_perplexity_report


class Sequences:
    def __init__(self,T,lengths,choices=2):
        self.T=T; self.lengths=lengths; self.choices=choices
        self.ids=[f'g-{i}' for i in range(len(lengths))]
        self.segments=[(key,7,7+length) for key,length in zip(self.ids,lengths)]
        self.summary={'mode':'main','label':'测试主库'}
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        x=torch.zeros(self.T,15,10,9); y=torch.full((self.T,),-100,dtype=torch.long)
        legal=torch.zeros(self.T,8100,dtype=torch.bool); n=self.lengths[i]
        x[:n,0,0,0]=torch.arange(1,n+1); y[:n]=0; legal[:n,:self.choices]=True
        return x,y,legal,n


class Policy(torch.nn.Module):
    def __init__(self,history=False,perfect=False):
        super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(8100))
        self.history=history; self.perfect=perfect; self.calls=0
    def forward_step(self,x,cache,valid):
        self.calls+=1; values=x[:,0,0,0]
        assert (values[valid]>0).all()  # Padding is never evaluated as a real board.
        total=values if cache is None else values+cache.total
        logits=self.weight.expand(len(x),-1).clone()
        if self.history: logits[:,1]=total/10
        if self.perfect: logits[:,0]=1000
        state=SimpleNamespace(total=total,detach_=lambda:None)
        return logits,state


def test_uniform_policy_and_half_probability_padding_use_sequence_likelihood():
    data=Sequences(8,[8,4,1,0],choices=4); model=Policy()
    result=evaluate_books(model,data,micro_batch=3,ngram=8,chunk_size=3)
    expected=[4,math.sqrt(8),math.exp((math.log(4)+7*math.log(2))/8),2]
    assert [s['perplexity'] for s in result['samples']]==pytest.approx(expected,rel=1e-6)
    assert result['perplexity']==pytest.approx(sum(expected)/4)
    assert result['corpus_perplexity']==pytest.approx(math.prod(expected)**.25)
    assert result['steps']==13 and result['padding_steps']==19 and result['segments']==4
    assert 'top1' not in result and 'top3' not in result
    assert [(s['source_game_id'],s['start_ply']) for s in result['samples']]==[(f'g-{i}',7) for i in range(4)]


def test_four_pads_in_eight_steps_contribute_four_times_log_two():
    result=evaluate_books(Policy(perfect=True),Sequences(8,[4]),ngram=8)
    assert result['samples'][0]['log_perplexity']==pytest.approx(math.log(2)/2)
    assert result['perplexity']==pytest.approx(math.sqrt(2))


@pytest.mark.parametrize('ngram',[1,2,3,6])
def test_sliding_context_has_no_older_state_and_counts_each_action_once(ngram):
    data=Sequences(6,[6,4,2]); model=Policy(history=True)
    state=copy.deepcopy(model.state_dict()); model.train()
    parallel=evaluate_books(model,data,micro_batch=3,ngram=ngram,chunk_size=2)
    serial=evaluate_books(model,data,micro_batch=1,ngram=ngram,chunk_size=1)
    expected=[]
    for length in data.lengths:
        total=sum(math.log1p(math.exp(sum(range(max(1,t-ngram+1),t+1))/10)) for t in range(1,length+1))
        expected.append(math.exp((total+(6-length)*math.log(2))/6))
    assert [s['perplexity'] for s in parallel['samples']]==pytest.approx(expected,rel=1e-6)
    assert parallel['perplexity']==pytest.approx(serial['perplexity'],rel=1e-7)
    assert model.training and model.weight.grad is None
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


def test_all_padding_slice_still_has_exactly_one_perplexity():
    model=Policy(); result=evaluate_books(model,Sequences(32,[0,0]))
    assert result['perplexity']==2 and result['segments']==2 and model.calls==0


@pytest.mark.parametrize('length,ngram',[(0,1),(32,0),(32,33),(8,16),(32,True)])
def test_invalid_context_or_slice_length_is_rejected(length,ngram):
    with pytest.raises(ValueError): validate_evaluation(length,ngram)


def test_nonfinite_probability_restores_original_mode():
    model=Policy(); model.train()
    with torch.no_grad(): model.weight[0]=torch.nan
    with pytest.raises(FloatingPointError): evaluate_books(model,Sequences(32,[1]))
    assert model.training


def test_histogram_boundaries_and_overflow_bin_count_every_slice(tmp_path):
    values=[1,1.5,2,4,8,1024,10000]
    hist=perplexity_histogram(values)
    assert hist['total']==7 and sum(b['count'] for b in hist['bins'])==7
    assert hist['bins'][0]['count']==2 and hist['bins'][1]['count']==1
    assert hist['bins'][-1]['upper'] is None and hist['bins'][-1]['count']==2
    result=evaluate_books(Policy(),Sequences(32,[1,17,32]))
    latest=tmp_path/'validation'/'latest.json'
    compact=save_perplexity_report(result,tmp_path/'validation'/'epoch-0001',latest=latest)
    assert 'samples' not in compact and compact['histogram']['total']==3
    assert sum(b['count'] for b in compact['histogram']['bins'])==3
    rows=[json.loads(s) for s in open(compact['artifacts']['samples'],encoding='utf-8')]
    assert len(rows)==3
    assert json.loads(latest.read_text(encoding='utf-8'))['perplexity']==result['perplexity']


def test_dashboard_uses_histogram_without_accuracy(tmp_path):
    from training.curves import TrainingCurves
    from training.dashboard_data import CurveReader
    from training import dashboard
    from pathlib import Path
    run=tmp_path/'trial'; curves=TrainingCurves(run,{})
    result=evaluate_books(Policy(),Sequences(32,[1,17,32]))
    report=save_perplexity_report(result,run/'validation'/'epoch-0001',latest=run/'validation'/'latest.json')
    curves.epoch(1,{'loss':1.,'counts':{'book':0}},report); curves.close()
    data=CurveReader(tmp_path).read('trial/tensorboard')
    assert data['validation_perplexity']['histogram']['total']==3
    assert 'validation/perplexity' in data['series'] and 'validation/top1' not in data['series']
    html=Path(dashboard.__file__).with_name('dashboard.html').read_text(encoding='utf-8')
    assert '困惑度区间' in html and 'validation/top1' not in html and '动作命中率' not in html

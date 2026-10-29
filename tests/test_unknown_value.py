import copy
from dataclasses import replace
import json

import pytest
torch=pytest.importorskip('torch')

from data_pipeline.database import connect
from data_pipeline.notation import STANDARD_FEN
from encoder import move_to_action
from models.Opus.config import OpusConfig
from models.Opus.loss import masked_value_ce
from models.Opus.network import Opus5
from training.data import BookDataset,TrainingBatch
from training.engine import ModelOutput,train_logical_batch
from training.loss import LossBatch,LossWeights,mixed_loss


@pytest.mark.parametrize('result',[None,'',' ','\t\r\n','*',' * '])
def test_empty_database_outcomes_keep_policy_but_no_value_supervision(tmp_path,result):
    path=tmp_path/'books.sqlite3'; conn=connect(path)
    moves=[move_to_action((6,0,5,0))]
    conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
        ('unknown',STANDARD_FEN,json.dumps(moves),1,result,'book_study','train','unknown','{}'))
    conn.commit(); conn.close()
    sample=BookDataset(T=4,db=path)[0]
    assert sample.length==1 and sample.book_mask.tolist()==[True,False,False,False]
    assert sample.value_targets.tolist()==[-100]*4
    from training.data import collate_examples
    batch=collate_examples([sample])
    output=ModelOutput(torch.zeros(4,1,8100,requires_grad=True),torch.full((4,1,3),torch.nan))
    result=mixed_loss(output,batch.loss,LossWeights(book=1,value=2.5))
    assert result.counts.book==1 and result.counts.value==0
    assert result.sums['value'].item()==0 and torch.isfinite(result.total)


@pytest.mark.parametrize('targets',[None,torch.full((2,1),-100)])
def test_missing_value_loss_is_exact_zero_even_for_nonfinite_unused_logits(targets):
    logits=torch.full((2,1,3),torch.nan,requires_grad=True)
    loss,tb=masked_value_ce(logits,targets,torch.tensor([2]))
    assert loss.item()==0 and torch.equal(tb,torch.zeros(2,1))
    loss.backward()
    assert torch.equal(logits.grad,torch.zeros_like(logits))


def test_mixed_known_unknown_value_labels_only_normalize_by_known_steps():
    policy=torch.randn(3,2,8100,requires_grad=True)
    values=torch.randn(3,2,3)
    labels=torch.tensor([[0,-100],[2,-100],[-100,-100]])
    values[labels==-100]=torch.nan
    values.requires_grad_()
    legal=torch.zeros_like(policy,dtype=torch.bool); legal[...,:3]=True
    batch=LossBatch.books(torch.zeros(3,2,dtype=torch.long),legal,torch.tensor([3,3]),value_targets=labels)
    result=mixed_loss(ModelOutput(policy,values),batch,LossWeights(book=0,value=2.5))
    expected=torch.nn.functional.cross_entropy(values[:2,0],torch.tensor([0,2]))
    assert result.counts.value==2 and torch.allclose(result.total,2.5*expected)
    # An unknown-only TBPTT chunk still contributes zero when the whole batch
    # has a nonzero value denominator from other chunks.
    tail=mixed_loss(ModelOutput(policy[2:],values[2:]),batch.slice(2,3),
                    LossWeights(book=0,value=2.5),denominators=result.counts)
    assert tail.counts.value==0 and tail.sums['value'].item()==tail.total.item()==0
    result.total.backward()
    assert torch.equal(values.grad[labels==-100],torch.zeros_like(values.grad[labels==-100]))
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize('book_weight',[0.,1.])
def test_unknown_batch_never_updates_value_branch_with_nonzero_coefficient(book_weight):
    torch.set_num_threads(2); torch.manual_seed(17)
    model=Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
                          ffn_dim=32,cnn_channels=8,value_hidden=8))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    x=torch.randn(2,1,15,10,9)
    legal=torch.zeros(2,1,8100,dtype=torch.bool); legal[...,:3]=True
    known=LossBatch.books(torch.zeros(2,1,dtype=torch.long),legal,torch.tensor([2]),
                          value_targets=torch.tensor([[0],[2]]))
    weights=LossWeights(book=book_weight,value=2.5)
    # Initialize Adam's momentum first; zero supervision must also leave these
    # previously trained value parameters and their optimizer state unchanged.
    assert train_logical_batch(model,optimizer,[TrainingBatch(x,known)],weights,chunk_size=1)['updated']
    before=copy.deepcopy(model.state_dict())
    protected={name:(parameter,copy.deepcopy(optimizer.state[parameter]))
               for name,parameter in model.named_parameters()
               if name.startswith(('cores.3.','cores.4.','bridge_','value_head.'))}
    assert protected
    report=train_logical_batch(model,optimizer,[TrainingBatch(x,replace(known,value_targets=None))],weights,chunk_size=1)
    assert report['terms']['value']==0 and report['counts']['value']==0
    assert report['updated']==bool(book_weight)
    for name,(parameter,state) in protected.items():
        assert torch.equal(parameter,before[name])
        for key,value in state.items():
            assert torch.equal(optimizer.state[parameter][key],value)
    if not book_weight:
        assert report['loss']==0 and all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())

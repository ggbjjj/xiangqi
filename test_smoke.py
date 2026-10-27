"""Full Haiku-5 configuration forward/backward smoke test."""
import pytest
torch=pytest.importorskip('torch')
from models.haiku5.network import Haiku5
from models.haiku5.loss import masked_policy_ce
from encoder import encode_state,legal_action_mask,move_to_action
from board import Board


def test_full_haiku5_training_step():
    torch.set_num_threads(2)
    net=Haiku5()
    board=Board.fresh()
    x=torch.from_numpy(encode_state(board,'r'))[None,None]
    mask=torch.from_numpy(legal_action_mask(board,'r'))[None,None]
    y=torch.tensor([[move_to_action((6,0,5,0))]])
    loss,_,p=masked_policy_ce(net(x),y,mask,torch.tensor([1]))
    loss.backward()
    assert p.shape==(1,1,8100) and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert sum(p.numel() for p in net.parameters())==3043614

import json
import math
import sqlite3
from dataclasses import asdict

import pytest

torch=pytest.importorskip('torch')
from data_pipeline.notation import STANDARD_FEN,from_fen,to_fen
from encoder import action_to_move,encode_state,legal_action_mask,move_to_action
from repetition import RepetitionTracker
from selfplay.generator import SelfPlayRules
from selfplay.store import Snapshot,create_run,save_game
from training.data import SelfPlayDataset,collate_examples,result_target
from training.engine import ModelOutput
from training.exploration import Exploration
from training.loss import LossCounts,LossWeights,mixed_loss


CYCLE=((9,1,7,2),(0,1,2,2),(7,2,9,1),(2,2,0,1))


def store_game(tmp_path,moves,*,result='1-0',prefix=(),start_fen=STANDARD_FEN,legacy=False):
    path,run=create_run(Snapshot('Opus','Opus-5','checkpoint','test','sha',{}),
        Exploration(),SelfPlayRules(),True,42,tmp_path/'selfplay.sqlite3')
    board,turn=from_fen(start_fen)
    tracker=None if legacy else RepetitionTracker(board,turn)
    actions=[];steps=[]
    for i,move in enumerate((*prefix,*moves)):
        action=move_to_action(move)
        assert legal_action_mask(board,turn,repetition=tracker)[action]
        if i<len(prefix): actions.append(action)
        else:
            j=i-len(prefix)
            steps.append(dict(turn=turn,action=action,old_log_mu=-1-j/1000,
                old_value=(j%9-4)/10,temperature=1+(j%3)/10,epsilon=(j%5)/100))
        board=board.apply_move(*move)
        if tracker is not None: tracker.push(move,board)
        turn='b' if turn=='r' else 'r'
    game=save_game(path,run,start_fen=start_fen,prefix=actions,steps=steps,result=result,
        termination='test',final_fen=to_fen(board,turn))
    if legacy:
        rules=asdict(SelfPlayRules());rules.pop('repetition_policy')
        with sqlite3.connect(path) as conn:
            conn.execute('UPDATE runs SET rules_json=? WHERE run_id=?',(json.dumps(rules),run))
    return path,run,game,steps


@pytest.mark.parametrize('plies',[1,127,128,129,256,257,512])
def test_default_C_keeps_every_move_once_and_only_pads_tail(tmp_path,plies):
    moves=(CYCLE*math.ceil(plies/4))[:plies]
    path,run,game,steps=store_game(tmp_path,moves,legacy=True)
    before=path.read_bytes()
    dataset=SelfPlayDataset(path,run)
    samples=[dataset[i] for i in range(len(dataset))]
    assert dataset.C==128 and dataset.ids==[game]
    assert len(samples)==math.ceil(plies/128)
    assert [s.length for s in samples]==[min(128,plies-start) for start in range(0,plies,128)]
    assert torch.cat([s.actions[:s.length] for s in samples]).tolist()==[s['action'] for s in steps]
    assert sum(s.length for s in samples)==plies
    for s in samples:
        assert s.x.shape==(128,15,10,9) and s.result=='1-0' and s.source_game_id==game
        assert not s.x[s.length:].count_nonzero() and not s.legal[s.length:].any()
        assert (s.actions[s.length:]==-100).all() and (s.value_targets[s.length:]==-100).all()
        assert s.selfplay_mask.sum()==s.length and not s.book_mask.any()
    assert path.read_bytes()==before


@pytest.mark.parametrize('result',['1-0','0-1','1/2-1/2','*'])
def test_odd_C_preserves_positions_turn_results_and_ppo_measurements(tmp_path,result):
    moves=(*CYCLE,*CYCLE,(6,0,5,0),(3,0,4,0),(6,2,5,2))
    path,run,game,steps=store_game(tmp_path,moves,result=result)
    dataset=SelfPlayDataset(path,run,C=3)
    board,turn=from_fen(STANDARD_FEN);tracker=RepetitionTracker(board,turn)
    for i in range(len(dataset)):
        sample=dataset[i]
        assert sample.start_ply==i*3 and sample.start_fen==to_fen(board,turn)
        assert sample.result==result and sample.provenance==('checkpoint','sha','Opus-5')
        for t in range(sample.length):
            step=steps[i*3+t]
            assert torch.equal(sample.x[t],torch.from_numpy(encode_state(board,turn)))
            assert torch.equal(sample.legal[t],torch.from_numpy(legal_action_mask(board,turn,repetition=tracker)))
            assert sample.value_targets[t]==result_target(result,turn)
            for field in ('old_log_mu','old_value','temperature','epsilon'):
                assert float(getattr(sample,field)[t])==pytest.approx(step[field])
            move=action_to_move(step['action']);board=board.apply_move(*move);tracker.push(move,board)
            turn='b' if turn=='r' else 'r'
        if i: assert sample.start_fen!=STANDARD_FEN


def test_crop_boundary_keeps_repetition_history(tmp_path):
    moves=(*CYCLE,*CYCLE,(6,0,5,0),(3,0,4,0))
    path,run,_,_=store_game(tmp_path,moves)
    sample=SelfPlayDataset(path,run,C=4)[2]
    assert sample.start_ply==8 and sample.length==2
    board,turn=from_fen(sample.start_fen)
    repeated=move_to_action(CYCLE[0])
    assert legal_action_mask(board,turn)[repeated]
    assert not sample.legal[0,repeated]


def test_long_prefix_and_nonstandard_start_do_not_consume_C(tmp_path):
    board,turn=from_fen(STANDARD_FEN)
    board=board.apply_move(6,0,5,0)
    start=to_fen(board,'b')
    prefix=((3,0,4,0),(6,2,5,2),(3,2,4,2))
    moves=((6,4,5,4),(3,4,4,4),(6,6,5,6))
    path,run,_,_=store_game(tmp_path,moves,start_fen=start,prefix=prefix)
    dataset=SelfPlayDataset(path,run,C=2)
    assert len(dataset)==2
    for move in prefix: board=board.apply_move(*move)
    first,last=dataset[0],dataset[1]
    assert first.start_fen==to_fen(board,'r') and first.length==2 and last.length==1
    assert first.selfplay_mask.tolist()==[True,True]
    assert last.selfplay_mask.tolist()==[True,False]
    assert first.actions.tolist()==[move_to_action(m) for m in moves[:2]]
    assert last.actions.tolist()==[move_to_action(moves[2]),-100]


def test_optional_read_limit_applies_before_splitting(tmp_path):
    moves=(*CYCLE,*CYCLE,(6,0,5,0),(3,0,4,0))
    path,run,_,_=store_game(tmp_path,moves,result='0-1')
    dataset=SelfPlayDataset(path,run,C=3,max_generated_plies=5)
    assert len(dataset)==2 and [dataset[i].length for i in range(2)]==[3,2]
    assert all(dataset[i].result=='0-1' for i in range(2))


@pytest.mark.parametrize('result,valid',[('1-0',5),('*',0)])
def test_padding_and_unknown_results_never_enter_losses(tmp_path,result,valid):
    path,run,_,_=store_game(tmp_path,(*CYCLE,(6,0,5,0)),result=result)
    dataset=SelfPlayDataset(path,run,C=3)
    batch=collate_examples([dataset[0],dataset[1]])
    assert batch.loss.lengths.tolist()==[3,2] and batch.x.shape==(3,2,15,10,9)
    assert batch.loss.counts()==LossCounts(0,valid,valid,valid)
    policy=torch.zeros(3,2,8100,requires_grad=True)
    value=torch.zeros(3,2,3,requires_grad=True)
    loss=mixed_loss(ModelOutput(policy,value),batch.loss,LossWeights.selfplay())
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert not policy.grad[2,1].count_nonzero()
    if valid: assert not value.grad[2,1].count_nonzero()
    else: assert value.grad is None


@pytest.mark.parametrize('C',[0,-1,1.5,True,None])
def test_invalid_C_rejected_before_opening_database(C):
    with pytest.raises(ValueError,match='C 必须为正整数'):
        SelfPlayDataset('missing.sqlite3','run',C=C)

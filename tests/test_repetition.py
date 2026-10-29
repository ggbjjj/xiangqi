import json
import math
import sqlite3

import numpy as np
import pytest

from ai import SearchAI
from board import Board
from data_pipeline.notation import STANDARD_FEN
from encoder import legal_action_mask,move_to_action
from game import Game
from repetition import RepetitionTracker,REPETITION_POLICY


LEFT=((9,1,7,2),(0,1,2,2),(7,2,9,1),(2,2,0,1))
RIGHT=((9,7,7,6),(0,7,2,6),(7,6,9,7),(2,6,0,7))


def play(game,moves):
    for move in moves:
        assert game.move(*move)=='ok'


@pytest.mark.parametrize('turn',('r','b'))
def test_third_consecutive_return_masks_exactly_one_current_player_move(turn):
    game=Game();game.turn=turn
    cycle=LEFT if turn=='r' else (LEFT[1],LEFT[0],LEFT[3],LEFT[2])
    play(game,cycle)
    assert not game.must_change_move()  # starting position plus one return: two
    play(game,cycle)
    physical=set(game.board.all_legal_moves(turn))
    legal=set(game.all_legal_moves())
    assert not game.is_over() and game.turn==turn
    assert physical-legal=={cycle[0]}
    assert len(legal)==len(physical)-1
    assert cycle[0][2:] not in game.legal_moves_at(*cycle[0][:2])
    mask=legal_action_mask(game.board,turn,game.history)
    assert set(np.flatnonzero(mask))=={move_to_action(m) for m in legal}
    before=game.board.signature(),len(game.history),game.quiet_moves
    assert game.move(*cycle[0])=='illegal'
    assert (game.board.signature(),len(game.history),game.quiet_moves)==before
    # A genuine different move is allowed; the opponent has no blanket ban.
    alternate=next(m for m in legal if m[:2]!=cycle[0][:2])
    assert game.move(*alternate)=='ok'
    assert not game.must_change_move()


def test_scattered_returns_do_not_accumulate_and_changed_cycle_restarts():
    game=Game()
    play(game,LEFT+RIGHT)
    assert not game.must_change_move()  # board seen three times, different loops
    play(game,RIGHT)
    assert game.repetition_tracker().forbidden_moves()=={RIGHT[0]}
    assert LEFT[0] in game.all_legal_moves()
    play(game,LEFT)
    assert not game.must_change_move()
    play(game,LEFT)
    assert game.repetition_tracker().forbidden_moves()=={LEFT[0]}


def test_longer_identical_cycle_can_contain_incidental_returns():
    game=Game()
    play(game,(LEFT+RIGHT)*2)
    assert game.repetition_tracker().forbidden_moves()=={LEFT[0]}
    assert not game.is_over()


def test_rules_history_is_rebuildable_and_undo_restores_the_ban():
    game=Game();play(game,LEFT*2)
    rebuilt=RepetitionTracker.from_history(game.board,game.turn,game.history)
    assert rebuilt.forbidden_moves()==game.repetition_tracker().forbidden_moves()=={LEFT[0]}
    with pytest.raises(ValueError,match='历史'):
        legal_action_mask(game.board,'b',repetition=rebuilt)
    assert game.undo() and not game.must_change_move()
    assert game.move(*LEFT[-1])=='ok' and game.must_change_move()
    assert game.move(6,0,5,0)=='ok' and not game.must_change_move()
    assert game.undo() and game.must_change_move()
    game.reset()
    assert not game.history and not game.must_change_move()


def test_twofold_override_and_tracker_pop():
    board=Board.fresh();tracker=RepetitionTracker(board,'r',limit=2)
    for move in LEFT:
        board=board.apply_move(*move);tracker.push(move,board)
    assert tracker.forbidden_moves()=={LEFT[0]}
    tracker.pop()
    assert not tracker.forbidden_moves()
    for bad in (0,1,2.5,True):
        with pytest.raises(ValueError):RepetitionTracker(Board.fresh(),'r',bad)


def test_search_ai_uses_the_same_exclusion_without_mutating_game():
    game=Game();play(game,LEFT*2)
    before=game.board.signature(),list(game.history)
    move=SearchAI(max_depth=2,time_limit=5).choose_move(game.board,game.turn,history=game.history)
    assert move in game.all_legal_moves() and move!=LEFT[0]
    assert (game.board.signature(),game.history)==before


def test_neural_ai_cannot_rank_or_choose_a_masked_top_logit():
    torch=pytest.importorskip('torch')
    from ai import NeuralAI
    class PreferLoop(torch.nn.Module):
        def forward_step(self,x,cache=None):
            logits=torch.zeros(len(x),8100,device=x.device)
            logits[:,move_to_action(LEFT[0])]=100
            return logits,cache
    game=Game();play(game,LEFT*2)
    ai=NeuralAI(PreferLoop())
    ranked=ai.rank_moves(game.board,game.turn,game.history,topk=8100)
    assert len(ranked)==len(game.all_legal_moves())
    assert LEFT[0] not in {move for move,prob in ranked}
    assert ai.choose_move(game.board,game.turn,history=game.history) in game.all_legal_moves()


def test_no_legal_alternative_is_a_loss_in_game_and_generator(monkeypatch):
    pytest.importorskip('torch')
    from selfplay.generator import Position,SelfPlayRules
    game=Game();play(game,LEFT*2)
    state=Position(STANDARD_FEN,SelfPlayRules())
    for move in LEFT*2:state.move(move_to_action(move),state.legal_mask())
    original=Board.all_legal_moves
    key=game.board.signature()
    def only_cycle(board,turn):
        return [LEFT[0]] if turn=='r' and board.signature()==key else original(board,turn)
    monkeypatch.setattr(Board,'all_legal_moves',only_cycle)
    game._evaluate_end()
    assert game.winner()=='b' and '无合法变招' in game.status
    terminal,mask=state.terminal(8)
    assert terminal==('0-1','no_legal_alternative') and not mask.any()


def test_generator_refuses_bypassing_the_mask():
    pytest.importorskip('torch')
    from selfplay.generator import Position,SelfPlayRules
    state=Position(STANDARD_FEN,SelfPlayRules())
    for move in LEFT*2:state.move(move_to_action(move),state.legal_mask())
    physical=legal_action_mask(state.board,state.turn)
    before=state.board.signature(),state.turn,state.quiet,len(state.repetition.moves)
    with pytest.raises(ValueError,match='变招'):
        state.move(move_to_action(LEFT[0]),physical)
    assert (state.board.signature(),state.turn,state.quiet,len(state.repetition.moves))==before


def test_new_selfplay_replays_masked_sampling_probability_past_kv_window(tmp_path,assert_rollout_replay):
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    from models.Opus.config import OpusConfig
    from models.Opus.network import Opus5
    from model_store import save_checkpoint
    from selfplay.generator import SelfPlayGenerator,SelfPlayRules
    from selfplay.store import load_game
    from training.data import SelfPlayDataset
    torch.manual_seed(123)
    model=Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(1,2,2,2,2),
                          ffn_dim=32,cnn_channels=8,value_hidden=8)).eval()
    params=tmp_path/'params.sqlite3'
    ident=save_checkpoint(model,path=params,run_id='base',epoch=0,global_step=0,metrics={})
    gen=SelfPlayGenerator.from_version('Opus',ident,model_db=params,rules=SelfPlayRules(max_plies=2))
    prefix=[move_to_action(m) for m in LEFT*2]
    result=gen.generate(1,path=tmp_path/'rollouts.sqlite3',prefix=prefix,save_every=1,micro_batch=1)
    dataset=SelfPlayDataset(result['data_db'],result['run_id'])
    sample=dataset[0]
    assert not sample.legal[0,move_to_action(LEFT[0])]
    assert sample.actions[0]!=move_to_action(LEFT[0])
    assert sample.length==2
    assert_rollout_replay(gen.model,result['data_db'],sample)
    game,run,steps=load_game(result['data_db'],result['game_ids'][0])
    assert json.loads(run['rules_json'])['repetition_policy']==REPETITION_POLICY
    assert game['termination']=='max_plies'
    path=tmp_path/'invalid-prefix.sqlite3'
    with pytest.raises(ValueError,match='非法'):
        gen.generate(1,path=path,prefix=prefix+[move_to_action(LEFT[0])],save_every=1,micro_batch=1)
    assert not path.exists()


def test_legacy_rollout_retains_original_mask_and_unknown_version_is_rejected(tmp_path):
    pytest.importorskip('torch')
    from dataclasses import asdict
    from selfplay.generator import SelfPlayRules
    from selfplay.store import Snapshot,create_run,save_game
    from training.data import SelfPlayDataset
    from training.exploration import Exploration
    board=Board.fresh();turn='r';steps=[]
    for move in LEFT*2:
        mask=legal_action_mask(board,turn)
        steps.append({'turn':turn,'action':move_to_action(move),'old_log_mu':-math.log(mask.sum()),
                      'old_value':0.,'temperature':1.,'epsilon':0.})
        board=board.apply_move(*move);turn='b' if turn=='r' else 'r'
    path,run=create_run(Snapshot('Opus','opus5','checkpoint','v','sha',{}),Exploration(),
                        SelfPlayRules(),True,42,tmp_path/'legacy.sqlite3')
    save_game(path,run,start_fen=STANDARD_FEN,prefix=[],steps=steps,result='1/2-1/2',
              termination='repetition',final_fen=STANDARD_FEN)
    rules=asdict(SelfPlayRules());rules.pop('repetition_policy')
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE runs SET rules_json=? WHERE run_id=?',(json.dumps(rules),run))
    before=path.read_bytes()
    sample=SelfPlayDataset(path,run)[0]
    assert sample.length==8 and sample.value_targets[:8].tolist()==[1]*8
    assert (sample.value_targets[8:]==-100).all()
    assert path.read_bytes()==before
    rules['repetition_policy']='unknown-future-rule'
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE runs SET rules_json=? WHERE run_id=?',(json.dumps(rules),run))
    with pytest.raises(ValueError,match='规则版本'):SelfPlayDataset(path,run)


def test_book_loader_keeps_valid_prefix_and_drops_incompatible_outcome(tmp_path):
    torch=pytest.importorskip('torch')
    from data_pipeline.database import connect
    from data_pipeline.loader import GameDataset
    from models.HaiKu.loss import masked_policy_ce
    path=tmp_path/'books.sqlite3';conn=connect(path)
    samples=[('continued',LEFT*2+(LEFT[0],),'1-0'),
             ('escape',LEFT*2+((6,0,5,0),(3,0,4,0)),'1-0'),
             ('old-draw',LEFT*2,'1/2-1/2')]
    for name,moves,result in samples:
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                     (name,STANDARD_FEN,json.dumps([move_to_action(m) for m in moves]),len(moves),
                      result,'human_master','train',name,'{}'))
    conn.commit();conn.close();before=path.read_bytes()
    ds=GameDataset(12,db=path,include_value=True)
    x,y,legal,n,value=ds[0]
    assert n==8 and y[8:].tolist()==[-100]*4 and value.tolist()==[-100]*12
    loss=masked_policy_ce(torch.zeros(12,1,8100),y[:,None],legal[:,None],torch.tensor([n]))[0]
    assert torch.isfinite(loss)
    x,y,legal,n,value=ds[1]
    assert n==10 and not legal[8,move_to_action(LEFT[0])]
    assert value[:10].tolist()==[0,2]*5
    assert ds[2][-1].tolist()==[-100]*12
    cropped=GameDataset(2,db=path,include_value=True)[0]
    assert cropped[-1].tolist()==[-100]*2  # incompatibility was beyond T
    assert path.read_bytes()==before

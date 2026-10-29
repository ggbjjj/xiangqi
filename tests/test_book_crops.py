import json
import math
import random

import pytest
torch=pytest.importorskip('torch')

from board import Board
from data_pipeline.database import connect
from data_pipeline.loader import GameDataset,iter_game_batches,BOOK_CROP_VERSION
from data_pipeline.notation import STANDARD_FEN,from_fen,to_fen
from encoder import encode_state,legal_action_mask,move_to_action,action_to_move
from repetition import RepetitionTracker
from training.data import BookDataset,collate_examples
from training.engine import ModelOutput
from training.loss import LossCounts,LossWeights,mixed_loss


def legal_game(plies):
    board=Board.fresh();turn='r';tracker=RepetitionTracker(board,turn);rng=random.Random(23)
    actions=[]
    for _ in range(plies):
        move=rng.choice(tracker.legal_moves(board,turn))
        actions.append(move_to_action(move));board=board.apply_move(*move);tracker.push(move,board)
        turn='b' if turn=='r' else 'r'
    return actions


def store(tmp_path,actions,*,fen=STANDARD_FEN,result='1-0',extra=()):
    path=tmp_path/'books.sqlite3';conn=connect(path)
    for name,split,moves in [('game','train',actions),*extra]:
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
            (name,fen,json.dumps(moves),len(moves),result,'book_study',split,name,'{}'))
    conn.commit();conn.close();return path


@pytest.mark.parametrize('plies',[1,31,32,33,64,100,128])
def test_full_book_splits_cover_all_actions_and_only_zero_pad_tail(tmp_path,plies):
    actions=legal_game(plies);path=store(tmp_path,actions);before=path.read_bytes()
    dataset=GameDataset(32,db=path)
    samples=[dataset[i] for i in range(len(dataset))]
    assert dataset.ids==['game'] and len(dataset)==math.ceil(plies/32)
    assert [s[3] for s in samples]==[min(32,plies-start) for start in range(0,plies,32)]
    assert torch.cat([s[1][:s[3]] for s in samples]).tolist()==actions
    for x,y,legal,n in samples:
        assert x.shape==(32,15,10,9) and not x[n:].any() and not legal[n:].any()
        assert (y[n:]==-100).all()
    batches=list(iter_game_batches(32,4,db=path,shuffle=False))
    assert sum(int(lengths.sum()) for _,_,lengths in batches)==plies
    assert sum(len(lengths) for _,_,lengths in batches)==len(dataset)
    assert path.read_bytes()==before


@pytest.mark.parametrize('result',['1-0','0-1','1/2-1/2','*'])
def test_crops_restore_nonstandard_boards_black_turn_and_original_outcome(tmp_path,result):
    all_actions=legal_game(13)
    board,turn=from_fen(STANDARD_FEN)
    board=board.apply_move(*action_to_move(all_actions[0]));turn='b'
    fen=to_fen(board,turn);actions=all_actions[1:]
    path=store(tmp_path,actions,fen=fen,result=result)
    dataset=BookDataset(3,db=path)
    tracker=RepetitionTracker(board,turn)
    from data_pipeline.loader import result_target
    for i in range(len(dataset)):
        sample=dataset[i]
        assert sample.start_ply==3*i and sample.source_game_id=='game'
        assert sample.start_fen==to_fen(board,turn) and sample.result==result
        for t in range(sample.length):
            assert torch.equal(sample.x[t],torch.from_numpy(encode_state(board,turn)))
            assert torch.equal(sample.legal[t],torch.from_numpy(legal_action_mask(board,turn,repetition=tracker)))
            assert sample.value_targets[t]==result_target(result,turn)
            move=action_to_move(actions[i*3+t]);board=board.apply_move(*move);tracker.push(move,board)
            turn='r' if turn=='b' else 'b'


def test_100_ply_game_has_four_lengths_and_loss_ignores_tail_padding(tmp_path):
    actions=legal_game(100);path=store(tmp_path,actions)
    dataset=BookDataset(32,db=path)
    batch=collate_examples([dataset[i] for i in range(4)])
    assert batch.loss.lengths.tolist()==[32,32,32,4]
    assert batch.loss.counts()==LossCounts(book=100,value=100,entropy=100)
    assert not batch.x[4:,3].any() and not batch.loss.book_mask[4:,3].any()
    policy=torch.randn(32,4,8100,requires_grad=True)
    value=torch.randn(32,4,3,requires_grad=True)
    output=ModelOutput(policy,value)
    weights=LossWeights.books()
    result=mixed_loss(output,batch.loss,weights)
    # Poison every padded position: masking must precede CE/softmax operations.
    valid=torch.arange(32)[:,None]<batch.loss.lengths[None,:]
    poisoned=ModelOutput(policy.detach().clone(),value.detach().clone())
    poisoned.policy_logits[~valid]=torch.nan;poisoned.value_logits[~valid]=torch.nan
    other=mixed_loss(poisoned,batch.loss,weights)
    assert torch.isfinite(other.total) and other.total==pytest.approx(float(result.total.detach()),abs=1e-6)
    result.total.backward()
    assert not policy.grad[~valid].any() and not value.grad[~valid].any()
    sums=0.
    for start,stop in ((0,7),(7,21),(21,32)):
        chunk=ModelOutput(policy.detach()[start:stop],value.detach()[start:stop])
        sums+=float(mixed_loss(chunk,batch.loss.slice(start,stop),weights,denominators=result.counts).total)
    assert sums==pytest.approx(float(result.total.detach()),abs=1e-6)


def test_original_game_limits_splits_and_independent_loader_lengths(tmp_path):
    actions=legal_game(100)
    path=store(tmp_path,actions,extra=[('next','train',actions[:33]),('held-out','validation',actions[:65])])
    train=GameDataset(32,db=path,limit=1)
    validation=GameDataset(32,'validation',path)
    assert train.ids==['game'] and len(train)==4
    assert validation.ids==['held-out'] and len(validation)==3
    assert not set(train.ids)&set(validation.ids)
    for T in (True,0,-1,2.5):
        with pytest.raises(ValueError,match='正整数'): GameDataset(T,db=path)


def test_repetition_and_outcome_checks_span_crop_boundaries(tmp_path):
    cycle=((9,1,7,2),(0,1,2,2),(7,2,9,1),(2,2,0,1))
    escaped=cycle*2+((6,0,5,0),(3,0,4,0))
    actions=[move_to_action(m) for m in escaped]
    path=store(tmp_path,actions)
    dataset=BookDataset(4,db=path)
    sample=dataset[2]
    assert sample.start_ply==8 and sample.length==2
    assert not sample.legal[0,move_to_action(cycle[0])]
    assert sample.value_targets.tolist()==[0,2,-100,-100]
    # An invalid continuation after a crop invalidates every crop's value label.
    with connect(path) as conn:
        moves=[move_to_action(m) for m in cycle*3]
        conn.execute('UPDATE games SET moves_json=?,ply_count=?',(json.dumps(moves),len(moves)))
    dataset=BookDataset(3,db=path)
    samples=[dataset[i] for i in range(len(dataset))]
    assert [s.length for s in samples]==[3,3,2,0]
    assert all((s.value_targets==-100).all() for s in samples)
    assert not samples[-1].x.any() and not samples[-1].legal.any()
    assert collate_examples(samples).loss.counts().book==8
    # A whole microbatch beyond the first forbidden move still advances
    # progress, with no forward pass, optimizer update or loss denominator.
    from training.engine import train_logical_batch
    class NeverRun(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def forward_step(self,*args): raise AssertionError('empty crop reached model')
    model=NeverRun();events=[]
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    result=train_logical_batch(model,optimizer,[collate_examples(samples[-1:])],progress=events.append)
    assert not result['updated'] and result['counts']==dict.fromkeys(('book','ppo','value','entropy'),0)
    assert model.weight.item()==1 and events[0]['advance']==1 and events[0]['steps']==0


def test_cli_counts_segments_and_trains_every_original_action(tmp_path,capsys):
    from model_store import save_checkpoint,load_checkpoint
    from models.Opus.config import OpusConfig
    from models.Opus.network import Opus5
    from training.__main__ import main
    torch.set_num_threads(2)
    model=Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
        ffn_dim=32,cnn_channels=8,value_hidden=8))
    db=tmp_path/'models.sqlite3'
    save_checkpoint(model,path=db,run_id='initial',epoch=0,global_step=0,metrics={})
    books=store(tmp_path,legal_game(10),extra=[('held-out','validation',legal_game(7))])
    result=main(['--family','Opus','--source','books','--model-db',str(db),'--book-db',str(books),
        '--T','3','--epochs','1','--batch-size','4','--micro-batch','2','--device','cpu',
        '--output',str(tmp_path/'runs'),'--no-progress'])
    folder=tmp_path/'runs'/result['run_id']
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert config['book_games']==1 and config['book_segments']==config['training_samples']==4
    assert config['validation_games']==1 and config['validation_segments']==1
    assert config['validation_length']==32 and config['ngram']==16
    assert config['book_crop_version']==BOOK_CROP_VERSION
    metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(r['counts']['book'] for r in metrics)==10
    assert sum(r['counts']['value'] for r in metrics)==10
    assert result['best']['validation']['steps']==7 and result['best']['validation']['segments']==1
    assert result['best']['validation']['padding_steps']==25
    state=load_checkpoint(db,result['checkpoints'][0],resume=True)[2]
    assert state['book_crop_version']==BOOK_CROP_VERSION and state['T']==3
    assert '本次训练总局数（切段后）：4' in capsys.readouterr().out


def test_legacy_training_entry_counts_crops_and_skips_zero_length_tail(tmp_path,capsys):
    from model_store import save_checkpoint,load_checkpoint
    from models.HaiKu.config import Haiku5Config
    from models.HaiKu.network import Haiku5
    import train
    torch.set_num_threads(2)
    model=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    db=tmp_path/'models.sqlite3'
    save_checkpoint(model,path=db,run_id='init',epoch=0,global_step=0,metrics={})
    cycle=((9,1,7,2),(0,1,2,2),(7,2,9,1),(2,2,0,1))
    actions=[move_to_action(m) for m in cycle*3]
    books=store(tmp_path,actions,extra=[('val','validation',legal_game(4)),('test','test',legal_game(4))])
    train.main(['--data',str(books),'--model-db',str(db),'--T','3','--epochs','1',
        '--batch-size','1','--micro-batch','1','--device','cpu','--precision','none',
        '--output',str(tmp_path/'runs'),'--run-name','legacy','--no-progress','--smoke'])
    directory=tmp_path/'runs'/'legacy'
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    assert config['train_games']==1 and config['train_segments']==config['training_samples']==4
    report=json.loads((directory/'evaluation.json').read_text(encoding='utf-8'))
    _,meta,state=load_checkpoint(db,report['checkpoint_id'],resume=True)
    assert meta['global_step']==3 and state['book_crop_version']==BOOK_CROP_VERSION
    assert '本次训练总局数（切段后）：4' in capsys.readouterr().out

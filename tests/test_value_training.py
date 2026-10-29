import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')

from data_pipeline.database import connect
from data_pipeline.loader import GameDataset
from data_pipeline.notation import STANDARD_FEN
from encoder import move_to_action
from model_store import save_checkpoint,load_checkpoint,load_value_model
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from training.curves import TrainingCurves
from training.dashboard_data import CurveReader
from training.data import BookDataset,collate_examples
from training.loss import LossBatch,LossWeights,mixed_loss


def tiny_opus():
    torch.set_num_threads(2)
    return Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
        ffn_dim=32,cnn_channels=8,value_hidden=8))


@pytest.fixture
def books(tmp_path):
    path=tmp_path/'books.sqlite3'
    moves=[move_to_action(m) for m in ((6,0,5,0),(3,0,4,0),(6,2,5,2))]
    conn=connect(path)
    for name,result in [('a-red','1-0'),('b-black','0-1'),('c-draw','1/2-1/2'),('d-unknown','*'),('e-null',None)]:
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
            (name,STANDARD_FEN,json.dumps(moves),len(moves),result,'human_master','train',name,'{}'))
    conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
        ('f-black-start',STANDARD_FEN.replace(' w ',' b '),json.dumps(moves[1:2]),1,'1-0',
         'human_master','train','f-black-start','{}'))
    conn.commit(); conn.close()
    return path


def test_book_results_follow_actual_turn_and_survive_crop_padding(books):
    dataset=BookDataset(5,'train',books)
    expected=[[0,2,0,-100,-100],[2,0,2,-100,-100],[1,1,1,-100,-100],
              [-100]*5,[-100]*5,[2,-100,-100,-100,-100]]
    assert [dataset[i].value_targets.tolist() for i in range(len(dataset))]==expected
    batch=collate_examples([dataset[i] for i in range(len(dataset))])
    assert batch.loss.counts().book==16 and batch.loss.counts().value==10
    cropped=BookDataset(2,'train',books)
    assert len(cropped)==11 and len(cropped.ids)==6
    assert cropped[0].value_targets.tolist()==[0,2]
    assert cropped[1].value_targets.tolist()==[0,-100]
    assert cropped[2].value_targets.tolist()==[2,0]
    assert len(GameDataset(2,'train',books)[0])==4  # legacy callers retain their tuple contract


@pytest.mark.parametrize('source',['selfplay','mixed'])
def test_selfplay_and_mixed_cli_defaults_train_real_rollouts(books,tmp_path,source):
    from selfplay.generator import SelfPlayGenerator,SelfPlayRules
    from training.__main__ import main
    model=tiny_opus()
    db=tmp_path/'Opus/models.sqlite3'
    ident=save_checkpoint(model,path=db,run_id='initial',epoch=0,global_step=0,metrics={})
    generator=SelfPlayGenerator.from_version('Opus',ident,model_db=db,rules=SelfPlayRules(max_plies=2))
    rollout=generator.generate(2,path=tmp_path/'selfplay.sqlite3')
    report=main(['--family','Opus','--model-db',str(db),'--source',source,'--book-db',str(books),
        '--book-limit','2','--selfplay-db',rollout['data_db'],'--rollout-run',rollout['run_id'],
        '--validation','none','--T','2','--epochs','1','--batch-size','4','--micro-batch','1',
        '--device','cpu','--output',str(tmp_path/'runs'),'--no-progress'])
    folder=tmp_path/'runs'/report['run_id']
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert config['weights']==dict.fromkeys(('book','value','ppo','entropy'),1.)
    metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(r['updated'] for r in metrics)
    assert sum(r['counts']['ppo'] for r in metrics)==4
    assert sum(r['counts']['value'] for r in metrics)==(10 if source=='mixed' else 4)
    assert sum(r['counts']['book'] for r in metrics)==(6 if source=='mixed' else 0)
    assert load_checkpoint(db,report['checkpoints'][0])[0].value_trained


@pytest.mark.parametrize('overrides,expected',[
    ([],{'book':1.,'value':1.,'ppo':0.,'entropy':0.}),
    (['--lambda-policy','.3','--lambda-value','.7','--lambda-ppo','0','--lambda-entropy','0'],
     {'book':.3,'value':.7,'ppo':0.,'entropy':0.}),
])
def test_book_cli_trains_value_branch_and_records_coefficients(books,tmp_path,overrides,expected):
    from training.__main__ import main
    torch.manual_seed(75)
    model=tiny_opus()
    db=tmp_path/'Opus/models.sqlite3'
    base=save_checkpoint(model,path=db,run_id='init',epoch=0,global_step=0,metrics={})
    before=copy.deepcopy(model.value_head.state_dict())
    report=main(['--family','Opus','--model-db',str(db),'--source','books','--book-db',str(books),
        '--book-limit','2','--validation','none','--T','2','--epochs','1','--batch-size','2',
        '--micro-batch','1','--tbptt-steps','1','--device','cpu','--output',str(tmp_path/'runs'),
        '--no-progress','--set-default',*overrides])
    config=json.loads((tmp_path/'runs'/report['run_id']/'config.json').read_text(encoding='utf-8'))
    assert config['weights']==expected
    learned,meta,state=load_checkpoint(db,report['checkpoints'][0],resume=True)
    assert learned.value_trained and state['loss_weights']==expected
    assert not load_checkpoint(db,base)[0].value_trained
    assert any(not torch.equal(value,learned.value_head.state_dict()[key]) for key,value in before.items())
    assert config['book_games']==2 and config['book_segments']==4
    assert all(int(item['step'])==2 for item in state['optimizer']['state'].values())
    assert load_value_model(path=db).value_trained


def test_zero_coefficient_components_hidden_on_all_curve_axes(tmp_path):
    weights={'book':1.,'value':.5,'ppo':0.,'entropy':0.}
    curves=TrainingCurves(tmp_path/'run',{'weights':weights})
    terms={'book':2.,'value':4.,'ppo':7.,'entropy':3.}
    counts={key:2 for key in terms}
    event={'chunk_loss':4.,'chunk_terms':terms,'chunk_counts':counts,'loss':4.,
           'terms':terms,'counts':counts,'finished_micro':True}
    curves.segment(event,epoch=1,global_step=0)
    curves.logical_batch({'loss':4.,'terms':terms,'counts':counts,'global_step':1})
    curves.close()
    payload=CurveReader(tmp_path).read('run/tensorboard')
    assert payload['weights']==weights
    for level in ('segment','micro','logical'):
        assert payload['series'][f'loss/{level}_total'][0][1]==4.
        assert payload['series'][f'loss/{level}_value'][0][1]==4.  # viewer multiplies by .5
        assert f'loss/{level}_ppo' not in payload['series']
        assert f'loss/{level}_entropy' not in payload['series']


def test_explicit_book_entropy_has_finite_policy_gradient_without_outcome():
    logits=torch.zeros(1,1,8100,requires_grad=True)
    with torch.no_grad(): logits[0,0,0]=2.
    legal=torch.zeros_like(logits,dtype=torch.bool); legal[...,:3]=True
    batch=LossBatch.books(torch.zeros(1,1,dtype=torch.long),legal,torch.tensor([1]))
    result=mixed_loss(logits,batch,LossWeights(book=0,entropy=1))
    assert result.counts.entropy==1 and result.total<0
    result.total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0,0,0]>0


def test_value_prediction_maps_current_player_and_reuses_history():
    from ai import NeuralAI
    from game import Game
    class FixedValue(torch.nn.Module):
        value_head=True
        def __init__(self):
            super().__init__(); self.calls=0
        def forward_step(self,x,cache):
            self.calls+=1
            return SimpleNamespace(policy_logits=torch.zeros(1,8100),value_logits=torch.tensor([[.6,.3,.1]]).log()),None
    net=FixedValue(); ai=NeuralAI(net); game=Game()
    first=ai.evaluate_position(game.board,game.turn,game.history)
    assert first['red_win']==pytest.approx(.6) and first['black_win']==pytest.approx(.1)
    assert sum(first[k] for k in ('red_win','draw','black_win'))==pytest.approx(1.)
    ai.rank_moves(game.board,game.turn,game.history)
    assert net.calls==1
    game.move(6,0,5,0)
    second=ai.evaluate_position(game.board,game.turn,game.history)
    assert second['red_win']==pytest.approx(.1) and second['black_win']==pytest.approx(.6)
    assert net.calls==2
    game.undo()
    ai.evaluate_position(game.board,game.turn,game.history)
    assert net.calls==3


def test_client_uses_independent_value_model_and_discards_stale_results(monkeypatch,tmp_path):
    import pygame
    import ui
    net=tiny_opus(); net.value_trained=True; net.display_name='Opus-5'
    calls=[]
    monkeypatch.setattr(ui,'load_value_model',lambda **kwargs:(calls.append(kwargs) or net))
    app=ui.XiangqiUI('pvp',model_db=tmp_path/'catalog')
    try:
        app._request_value()
        deadline=time.monotonic()+8
        while app.value_busy and time.monotonic()<deadline:
            app._poll_value(); time.sleep(.01)
        assert app.value_prediction is not None and app.value_model_name=='Opus-5'
        assert len(calls)==1 and app.ai is None and app._match_net is None
        app._draw()
        token=app._value_generation
        app.game.move(6,0,5,0); app._after_move()
        app._value_results.put((token,{'red_win':1.,'draw':0.,'black_win':0.},'stale',None))
        app._poll_value()
        assert app.value_prediction is None
        app._request_value()
        deadline=time.monotonic()+8
        while app.value_busy and time.monotonic()<deadline:
            app._poll_value(); time.sleep(.01)
        assert app.value_prediction['turn']=='b' and len(calls)==1
        app.action_undo()
        assert app.value_prediction is None
    finally:
        app._invalidate_ai(); pygame.quit()


def test_untrained_value_head_is_not_presented_as_game_forecast(tmp_path):
    import pygame
    from ui import XiangqiUI
    net=tiny_opus(); net.value_trained=False
    app=XiangqiUI('pve_red',net,model_db=tmp_path/'none.sqlite3')
    try:
        app._request_value()
        deadline=time.monotonic()+8
        while app.value_busy and time.monotonic()<deadline:
            app._poll_value(); time.sleep(.01)
        assert app.value_prediction is None
        assert '暂无' in app.value_message
    finally:
        app._invalidate_ai(); pygame.quit()

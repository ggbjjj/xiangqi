from contextlib import closing
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest

torch=pytest.importorskip('torch')
from model_store import save_checkpoint
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from selfplay.curves import SelfPlayCurves
from selfplay.generator import Position,SelfPlayGenerator,SelfPlayRules,generation_micro_batch
from selfplay.lengths import PlyHistogram
from selfplay.store import connect,committed_outcomes,committed_ply_histogram
from training.dashboard import make_server
from training.dashboard_data import CurveReader


@pytest.fixture(params=['HaiKu','Opus'])
def generator(request,tmp_path,monkeypatch):
    torch.set_num_threads(2)
    if request.param=='HaiKu':
        model=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    else:
        model=Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,2,2,2,2),
                              ffn_dim=32,cnn_channels=8,value_hidden=8))
    db=tmp_path/'models.sqlite3'
    ident=save_checkpoint(model,path=db,run_id='initial',epoch=0,global_step=0,metrics={})
    gen=SelfPlayGenerator.from_version(request.param,ident,model_db=db,rules=SelfPlayRules(max_plies=1))
    # Exercise every outcome with asynchronous completion/commit boundaries,
    # without relying on an untrained network to happen to deliver checkmate.
    results=iter(['1-0','0-1','1/2-1/2','1-0','0-1'])
    forward=gen._forward_batch; terminal=Position.terminal
    def scripted_forward(positions,cache,valid=None):
        for position in positions:
            if not hasattr(position,'test_outcome'): position.test_outcome=next(results)
        return forward(positions,cache,valid)
    def scripted_terminal(position,generated):
        result,legal=terminal(position,generated)
        if generated: result=(position.test_outcome,'test_outcome')
        return result,legal
    monkeypatch.setattr(gen,'_forward_batch',scripted_forward)
    monkeypatch.setattr(Position,'terminal',scripted_terminal)
    return gen,db


def test_cli_live_counts_commit_tail_and_remain_viewable(generator,tmp_path,monkeypatch,capsys):
    from selfplay.__main__ import main
    gen,db=generator; before=db.read_bytes(); observed=[]
    monkeypatch.setattr(SelfPlayGenerator,'from_version',lambda *args,**kwargs:gen)
    root=tmp_path/'runs'; reader=CurveReader(root)
    original_game=SelfPlayCurves.game
    def game(curves,row):
        original_game(curves,row)
        payload=reader.read(reader.runs()[0]); state=payload['selfplay']
        assert payload['kind']=='selfplay' and state['status']=='running'
        assert sum(state['outcomes'].values())==row['game']
        assert sum(state['saved_outcomes'].values())==row['saved_games']
        assert state['saved_outcomes']==committed_outcomes(state['data_db'],state['run_id'])
        observed.append(state)
    monkeypatch.setattr(SelfPlayCurves,'game',game)
    report=main(['--family',gen.snapshot.family,'--id','ID1','--games','5','--save-every','4',
                 '--micro-batch','4','--device','cpu','--data-db',str(tmp_path/'games.sqlite3'),
                 '--output',str(root),'--no-progress'])
    expected={'red_wins':2,'black_wins':2,'draws':1}
    assert report['outcomes']==report['saved_outcomes']==expected
    assert report['discarded_outcomes']==dict.fromkeys(expected,0)
    assert observed[0]['outcomes']=={'red_wins':1,'black_wins':0,'draws':0}
    assert observed[0]['saved_games']==0 and observed[0]['buffered_games']==1
    assert [r['saved_games'] for r in observed]==[0,0,0,4,4]
    payload=reader.read(reader.runs()[0]); state=payload['selfplay']
    assert state['status']=='complete' and state['saved_games']==5 and state['buffered_games']==0
    for key,value in expected.items():
        assert payload['series'][f'selfplay/completed_{key}'][-1][:2]==[5,value]
        assert payload['series'][f'selfplay/saved_{key}'][-1][:2]==[5,value]
    with closing(connect(report['data_db'],True)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]==5
    assert db.read_bytes()==before
    terminal=capsys.readouterr().out
    assert '本次已保存棋局胜负：红胜 2 局 / 黑胜 2 局 / 和棋 1 局' in terminal
    assert 'ply_histogram' not in terminal and 'saved_ply_histogram' not in terminal
    assert Path(report['output']).name==report['run_id']
    assert report['saved_ply_histogram']['total']==report['ply_histogram']['total']==5
    assert state['saved_ply_histogram']==committed_ply_histogram(report['data_db'],report['run_id'])
    directory=Path(report['output'])
    assert json.loads((directory/'report.json').read_text(encoding='utf-8'))['outcomes']==expected
    assert len((directory/'outcomes.jsonl').read_text(encoding='utf-8').splitlines())==7
    # The exact service used for training also serves the finished generation.
    server=make_server(root,port=0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        url=f'http://127.0.0.1:{server.server_port}'
        with urlopen(url+'/api/runs') as response: runs=json.load(response)['runs']
        with urlopen(url+'/api/scalars?run='+runs[0]) as response:
            assert json.load(response)['selfplay']['saved_outcomes']==expected
    finally: server.shutdown(); server.server_close()


def test_interruption_keeps_saved_and_discarded_results_separate(generator,tmp_path):
    gen,_=generator; stop=threading.Event(); logs=[]
    root=tmp_path/'runs'
    def start(row):
        logs.append(SelfPlayCurves(root/'interrupted',{'family':gen.snapshot.family,**row}))
    def progress(row):
        logs[0].game(row)
        if row['game']==3: stop.set()
    report=gen.generate(5,path=tmp_path/'games.sqlite3',save_every=2,micro_batch=2,
                        on_start=start,progress=progress,cancel_event=stop)
    logs[0].finish(report); logs[0].close()
    state=CurveReader(root).read('interrupted/tensorboard')['selfplay']
    assert state['status']=='interrupted' and state['completed_games']==3 and state['saved_games']==2
    assert state['outcomes']=={'red_wins':1,'black_wins':1,'draws':1}
    assert state['saved_outcomes']=={'red_wins':1,'black_wins':1,'draws':0}
    assert state['discarded_outcomes']=={'red_wins':0,'black_wins':0,'draws':1}
    assert state['buffered_games']==0 and state['discarded_games']==1 and state['unfinished_games']==1
    assert state['saved_outcomes']==committed_outcomes(report['data_db'],report['run_id'])
    assert state['ply_histogram']['total']==3 and state['saved_ply_histogram']['total']==2


def test_interruption_before_first_result_keeps_zero_counts(generator,tmp_path):
    gen,_=generator; stop=threading.Event(); stop.set(); logs=[]
    def start(row): logs.append(SelfPlayCurves(tmp_path/'empty',{'family':gen.snapshot.family,**row}))
    report=gen.generate(2,path=tmp_path/'empty.sqlite3',cancel_event=stop,on_start=start)
    logs[0].finish(report); logs[0].close()
    state=CurveReader(tmp_path).read('empty/tensorboard')['selfplay']
    assert state['status']=='interrupted' and state['completed_games']==0 and state['saved_games']==0
    assert sum(state['outcomes'].values())==sum(state['saved_outcomes'].values())==0
    assert state['ply_histogram']['total']==state['saved_ply_histogram']['total']==0
    assert 'NaN' not in (tmp_path/'empty'/'selfplay.json').read_text(encoding='utf-8')


def test_finished_lengths_not_step_cap_are_recorded_live(generator,tmp_path,monkeypatch):
    gen,_=generator; gen.rules=SelfPlayRules(max_plies=512); logs=[]; observed=[]
    def terminal(position,generated):
        legal=position.legal_mask()
        target={'1-0':1,'0-1':3,'1/2-1/2':4}.get(getattr(position,'test_outcome',None),512)
        return ((position.test_outcome,'test_early_end') if generated>=target else None),legal
    monkeypatch.setattr(Position,'terminal',terminal)
    def start(row): logs.append(SelfPlayCurves(tmp_path/'early',{'family':gen.snapshot.family,**row}))
    def game(row):
        logs[0].game(row); state=json.loads((logs[0].directory/'selfplay.json').read_text('utf-8'))
        assert state['ply_histogram']['total']==row['game']
        assert state['saved_ply_histogram']['total']==row['saved_games']
        observed.append(state)
    report=gen.generate(5,path=tmp_path/'early.sqlite3',save_every=4,micro_batch=2,on_start=start,progress=game)
    logs[0].finish(report); logs[0].close()
    histogram=report['ply_histogram']
    assert histogram['total']==5 and histogram['min']==1 and histogram['max']==4
    assert histogram['mean']==pytest.approx(12/5) and histogram['steps']==12
    assert histogram==report['saved_ply_histogram']==committed_ply_histogram(report['data_db'],report['run_id'])
    assert observed[0]['saved_ply_histogram']['total']==0 and observed[3]['saved_ply_histogram']['total']==4
    assert report['elapsed_seconds']>0


def test_histogram_boundaries_and_explicit_large_gpu_micro_batch():
    histogram=PlyHistogram()
    for length in (1,32,33,64,65,128,129,512): histogram.add(length)
    report=histogram.snapshot()
    assert report['total']==8 and report['min']==1 and report['max']==512
    assert [b['count'] for b in report['bins'][:5]]==[2,2,1,1,1]
    assert report['bins'][-1]['label']=='481–512' and report['bins'][-1]['count']==1
    assert sum(b['percent'] for b in report['bins'])==pytest.approx(100)
    assert generation_micro_batch(32,32)==32 and generation_micro_batch(64,64)==64
    assert generation_micro_batch(16)==16

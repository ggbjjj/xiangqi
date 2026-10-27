from datetime import datetime,timedelta,timezone
import io
from types import SimpleNamespace
import threading

import pytest

from training.progress import ConsoleProgress,GPUStats,duration


class FakeCUDA:
    def is_available(self): return True
    def current_device(self): return 0
    def get_device_properties(self,index): return SimpleNamespace(uuid='GPU-physical-7')
    def mem_get_info(self,index): return 6*2**30,8*2**30
    def memory_allocated(self,index): return 2**30
    def memory_reserved(self,index): return 2*2**30


def test_gpu_monitor_uses_physical_uuid_and_reports_device_memory(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    calls=[]
    def smi(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='83, 3072, 8192\n')
    stats=GPUStats('cuda:0',cuda=FakeCUDA(),runner=smi).sample()
    assert calls[0][1:3]==['-i','GPU-physical-7']
    assert stats['utilization']==83 and stats['memory_percent']==37.5
    assert stats['memory_used_mib']==3072 and stats['memory_total_mib']==8192
    assert stats['allocated_mib']==1024 and stats['reserved_mib']==2048


@pytest.mark.parametrize('response',['missing','[N/A], [N/A], [N/A]\n',''])
def test_gpu_monitor_unavailable_does_not_invent_zero_utilization(response):
    def unavailable(*args,**kwargs):
        if response=='missing': raise FileNotFoundError('nvidia-smi')
        return SimpleNamespace(stdout=response)
    stats=GPUStats('cuda',cuda=FakeCUDA(),runner=unavailable).sample()
    assert stats['utilization'] is None
    assert stats['memory_percent']==25  # CUDA memory API still works.
    assert GPUStats('cpu',runner=lambda *a,**kw:pytest.fail('CPU must not query NVIDIA')).sample()['memory_percent'] is None


def test_progress_eta_speed_losses_and_china_finish_time():
    now=[0.]; start=datetime(2026,10,27,tzinfo=timezone.utc)
    progress=ConsoleProgress('训练',10,enabled=False,clock=lambda:now[0],
                             wall_clock=lambda:start+timedelta(seconds=now[0]))
    assert progress.snapshot()['remaining_seconds'] is None
    now[0]=20.
    progress.update(2,work=100,games=32,L=.5,棋谱=.5,PPO=0.,V=0.,H=0.)
    snap=progress.snapshot()
    assert snap['units_per_second']==.1 and snap['steps_per_second']==5.
    assert snap['remaining_seconds']==80 and snap['estimated_total_seconds']==100
    assert snap['estimated_finish']=='2026-10-27T08:01:40+08:00'
    assert any('L=0.5000' in line for line in progress.lines())
    assert any('20.0%' in line for line in progress.lines())
    progress.close('已中断')
    assert progress.snapshot()['completed']==2 and progress.snapshot()['status']=='已中断'
    assert duration(None)=='统计中' and duration(3661)=='01:01:01'


def test_generation_eta_has_upper_bound_fallback_before_first_finished_game():
    now=[0.]
    progress=ConsoleProgress('生成',16,unit='局',fallback_work_total=1600,enabled=False,clock=lambda:now[0])
    now[0]=10; progress.update(work=160,batches=10,并行局数=16)
    snap=progress.snapshot()
    assert snap['remaining_seconds']==90 and snap['estimate_source']=='按局长上限估算'
    assert snap['inference_batches_per_second']==1
    progress.update(2,games=2)
    assert progress.snapshot()['remaining_seconds']==70
    assert progress.snapshot()['estimate_source']=='实测速度'
    assert not any('L=' in line for line in progress.lines())


def test_progress_samples_gpu_while_main_thread_is_idle_and_stops_cleanly():
    sampled=threading.Event(); stream=io.StringIO()
    class Provider:
        def sample(self):
            sampled.set(); return {'device':'cuda:0','utilization':75}
    with ConsoleProgress('生成',4,unit='局',provider=Provider(),stream=stream,interval=.01) as progress:
        assert sampled.wait(1)
        progress.update(1,phase='批量推理')
    assert not progress.thread.is_alive()
    assert progress.snapshot()['gpu']['utilization']==75
    assert '75%' in stream.getvalue() and '完成' in stream.getvalue()


def small_model(device='cpu'):
    import torch
    from models.Opus.config import OpusConfig
    from models.Opus.network import Opus5
    torch.set_num_threads(2)
    return Opus5(OpusConfig(dim=16,heads=(4,2,1,2,1),context_windows=(2,3,4,3,4),
                            ffn_dim=32,cnn_channels=8,value_hidden=8)).to(device)


@pytest.fixture
def generator(tmp_path):
    from model_store import save_checkpoint
    from selfplay.generator import SelfPlayGenerator,SelfPlayRules
    db=tmp_path/'parameters.sqlite3'
    ident=save_checkpoint(small_model(),path=db,run_id='init',epoch=0,global_step=0,metrics={})
    return SelfPlayGenerator.from_version('Opus',ident,model_db=db,rules=SelfPlayRules(max_plies=3))


@pytest.mark.parametrize('k,explicit,expected',[(16,None,16),(32,None,16),(8,None,8),(3,None,1),(16,4,4),(16,8,8)])
def test_generation_micro_batch_matches_commit_size(k,explicit,expected):
    from selfplay.generator import generation_micro_batch
    assert generation_micro_batch(k,explicit)==expected


@pytest.mark.parametrize('k,m',[(16,3),(16,32),(10,4),(0,4),(16,0)])
def test_invalid_generation_micro_batch_rejected(k,m):
    from selfplay.generator import generation_micro_batch
    with pytest.raises(ValueError): generation_micro_batch(k,m)


def test_batched_generation_separates_histories_handles_finished_slots_and_tail(generator,tmp_path,monkeypatch):
    import torch
    from encoder import move_to_action
    from selfplay.generator import Position
    from selfplay.store import list_runs,load_game
    from training.data import SelfPlayDataset,collate_examples
    from training.engine import iter_chunks
    from training.loss import LossWeights,mixed_loss
    calls=[]; steps=[]; games=[]
    original_forward=generator._forward_batch
    original_terminal=Position.terminal
    def forward(positions,cache,valid=None):
        for i,position in enumerate(positions): position.slot=i
        calls.append((len(positions),None if cache is None else cache.time,list(valid) if valid is not None else None))
        return original_forward(positions,cache,valid)
    def terminal(position,generated):
        result,legal=original_terminal(position,generated)
        if generated>=1 and getattr(position,'slot',-1)==0: result=('1/2-1/2','test_early_end')
        return result,legal
    monkeypatch.setattr(generator,'_forward_batch',forward)
    monkeypatch.setattr(Position,'terminal',terminal)
    result=generator.generate(5,path=tmp_path/'games.sqlite3',save_every=4,micro_batch=4,
        prefix=[move_to_action((6,0,5,0))],progress=games.append,on_step=steps.append)
    assert result['saved_games']==5 and result['completed_games']==5 and not result['interrupted']
    assert {c[0] for c in calls}=={4,1} and sum(c[1] is None for c in calls)==2
    assert any(c[2]==[False,True,True,True] for c in calls)
    assert sum(s['steps'] for s in steps)==result['generated_steps']==11
    assert len(calls)==result['inference_batches']==sum(s['batches'] for s in steps)
    assert [g['saved_games'] for g in games]==[0,0,0,4,4]
    run=list_runs(result['data_db'])[0]
    assert run['micro_batch']==4 and run['device']=='cpu'
    assert {load_game(result['data_db'],i)[0]['generation_index'] for i in result['game_ids']}==set(range(5))
    # Replay each game alone; inactive neighbors and batch changes cannot change old probabilities/value.
    dataset=SelfPlayDataset(result['data_db'],result['run_id'])
    with torch.inference_mode():
        for i in range(len(dataset)):
            batch=collate_examples([dataset[i]])
            for start,stop,output in iter_chunks(generator.model,batch.x,batch.loss.lengths,2):
                loss=mixed_loss(output,batch.loss.slice(start,stop),LossWeights.selfplay())
                if loss.counts.ppo: assert float(loss.diagnostics['ratio_sum'])/loss.counts.ppo==pytest.approx(1.,abs=2e-5)


def test_training_progress_counts_real_steps_and_preserves_optimizer_update(tmp_path):
    import torch
    from training.engine import train_logical_batch
    from training.loss import LossBatch
    from types import SimpleNamespace
    model=small_model(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    batches=[]
    for lengths in ([3,1],[2]):
        b=len(lengths); x=torch.randn(4,b,15,10,9)
        legal=torch.ones(4,b,8100,dtype=torch.bool); labels=torch.zeros(4,b,dtype=torch.long)
        batches.append(SimpleNamespace(x=x,loss=LossBatch.books(labels,legal,torch.tensor(lengths)),provenance=set()))
    events=[]
    result=train_logical_batch(model,optimizer,batches,chunk_size=2,progress=events.append)
    assert sum(e['advance'] for e in events)==pytest.approx(2.)
    assert sum(e['steps'] for e in events)==6 and sum(e['games'] for e in events)==3
    assert [e['finished_micro'] for e in events]==[False,True,True]
    assert result['micro_batches']==2 and result['duration_seconds']>0 and result['micro_batches_per_second']>0
    assert result['updated'] and all(int(s['step'])==1 for s in optimizer.state.values())


def test_selfplay_cli_batch_display_and_no_progress_roundtrip(generator,tmp_path,monkeypatch,capsys):
    from selfplay.__main__ import main
    from selfplay.generator import SelfPlayGenerator
    monkeypatch.setattr(SelfPlayGenerator,'from_version',lambda *a,**kw:generator)
    report=main(['--family','Opus','--id','ID1','--games','5','--save-every','4','--micro-batch','4',
        '--device','cpu','--data-db',str(tmp_path/'cli.sqlite3'),'--no-progress'])
    assert report['saved_games']==5 and report['micro_batch']==4
    assert report['progress']['completed']==5 and report['progress']['status']=='完成'
    assert '"saved_games": 5' in capsys.readouterr().out


@pytest.mark.skipif(not __import__('torch').cuda.is_available(),reason='requires a CUDA GPU')
def test_real_cuda_batched_generation(tmp_path):
    from model_store import save_checkpoint
    from selfplay.generator import SelfPlayGenerator,SelfPlayRules
    db=tmp_path/'parameters.sqlite3'
    ident=save_checkpoint(small_model(),path=db,run_id='cuda-init',epoch=0,global_step=0,metrics={})
    generator=SelfPlayGenerator.from_version('Opus',ident,model_db=db,device='cuda',rules=SelfPlayRules(max_plies=2))
    report=generator.generate(4,path=tmp_path/'cuda-games.sqlite3',micro_batch=4,save_every=4)
    assert report['saved_games']==4 and report['generated_steps']==8 and report['inference_batches']==2

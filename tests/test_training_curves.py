import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
torch=pytest.importorskip('torch')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from data_pipeline.database import connect
from data_pipeline.notation import STANDARD_FEN
from encoder import move_to_action
from model_store import (save_checkpoint,load_checkpoint,list_parameter_configs,
    get_best_parameter_config,get_default_parameter_id,get_latest_parameter_id)
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from training.control import WorkInterrupted
from training.curves import TrainingCurves,aggregate_reports
from training.data import BookDataset
from training.engine import train_logical_batch
from training.evaluation import evaluate_books
from training.loss import LossBatch,LossWeights


def events(path):
    accumulator=EventAccumulator(str(path),size_guidance={'scalars':0})
    accumulator.Reload(); return accumulator


@pytest.fixture
def book_run(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(46)
    model=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,context_plies=2))
    db=tmp_path/'models.sqlite3'
    ident=save_checkpoint(model,path=db,run_id='init',epoch=0,global_step=0,metrics={})
    books=tmp_path/'books.sqlite3'
    actions=[move_to_action(m) for m in [(6,0,5,0),(3,0,4,0),(6,2,5,2)]]
    conn=connect(books)
    for split in ('train','validation','test'):
        for i,n in enumerate((1,3)):
            conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                (f'{split}-{i}',STANDARD_FEN,json.dumps(actions[:n]),n,'*','human_master',split,f'{split}-{i}','{}'))
    conn.commit(); conn.close()
    return model,db,ident,books


def test_segment_curves_are_live_before_single_logical_optimizer_update(tmp_path):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(8100))
        def forward_step(self,x,cache,valid): return self.weight.expand(len(x),-1),None
    model=Policy(); optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    before=model.weight.detach().clone(); batches=[]
    for length in (35,17):
        legal=torch.zeros(length,1,8100,dtype=torch.bool); legal[...,:3]=True
        labels=torch.zeros(length,1,dtype=torch.long)
        batches.append(SimpleNamespace(x=torch.zeros(length,1,15,10,9),
            loss=LossBatch.books(labels,legal,torch.tensor([length])),provenance=set()))
    curves=TrainingCurves(tmp_path,{'weights':{'book':1}}); observed=[]
    def progress(event):
        # Backward and graph truncation have occurred, but optimizer.step has not.
        assert torch.equal(model.weight,before)
        curves.segment(event,epoch=1,global_step=0); observed.append(event)
        assert len(events(curves.directory).Scalars('loss/segment_total'))==len(observed)
    result=train_logical_batch(model,optimizer,batches,chunk_size=16,progress=progress)
    result['global_step']=1; curves.logical_batch(result); curves.close()
    assert len(observed)==5 and sum(e['steps'] for e in observed)==52
    assert not torch.equal(model.weight,before)
    log=events(curves.directory)
    assert len(log.Scalars('loss/micro_total'))==2
    assert [e.value for e in log.Scalars('optimization/parameter_updates')]==[0]*5
    assert [e.value for e in log.Scalars('optimization/updates_after_batch')]==[1]


def test_epoch_means_weight_actual_samples_in_uneven_batches():
    reports=[{'terms':{'book':1.,'ppo':2.,'value':3.,'entropy':4.},'counts':{'book':3,'ppo':1,'value':1,'entropy':1}},
             {'terms':{'book':5.,'ppo':6.,'value':7.,'entropy':8.},'counts':{'book':1,'ppo':3,'value':3,'entropy':3}}]
    report=aggregate_reports(reports,LossWeights(1,1,1,.1))
    assert report['terms']=={'book':2.,'ppo':5.,'value':6.,'entropy':7.}
    assert report['loss']==pytest.approx(12.3)


def test_heldout_evaluation_excludes_padding_and_preserves_parameters(book_run):
    model,db,ident,books=book_run
    dataset=BookDataset(5,'validation',books)
    model.train(); state=copy.deepcopy(model.state_dict())
    parallel=evaluate_books(model,dataset,micro_batch=2,ngram=3,chunk_size=2)
    serial=evaluate_books(model,dataset,micro_batch=1,ngram=3,chunk_size=2)
    assert parallel['steps']==4 and parallel['games']==2
    assert parallel['loss']==pytest.approx(serial['loss'],abs=2e-5)
    assert parallel['perplexity']==pytest.approx(serial['perplexity'],abs=2e-5)
    assert parallel['padding_steps']==6 and len(parallel['samples'])==2
    assert model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


def cli_args(db,books,tmp_path):
    return ['--family','HaiKu','--model-db',str(db),'--source','books','--book-db',str(books),
        '--epochs','3','--T','5','--batch-size','2','--micro-batch','1','--device','cpu',
        '--cpu-threads','2','--output',str(tmp_path/'runs'),'--run-name','curves-test','--no-progress']


def test_validation_selects_best_id_without_discarding_latest_epoch(book_run,tmp_path,monkeypatch,capsys):
    import training.__main__ as cli
    from model_store import main as admin
    _,db,ident,books=book_run
    base_sha=load_checkpoint(db,ident)[1]['sha256']
    scores=iter(zip((3.,2.,4.),(1.,2.,.5)))
    def score(model,dataset,**kwargs):
        result=evaluate_books(model,dataset,**kwargs)
        result['perplexity'],result['loss']=next(scores)
        return result
    monkeypatch.setattr(cli,'evaluate_books',score)
    report=cli.main(cli_args(db,books,tmp_path)+['--set-default'])
    assert report['best']['parameter_id']==3 and report['best']['completed_epochs']==2
    assert get_default_parameter_id('HaiKu',path=db)==3 and get_latest_parameter_id('HaiKu',path=db)==4
    assert len(list_parameter_configs('HaiKu',path=db))==4
    best=get_best_parameter_config('HaiKu',report['run_id'],path=db)
    assert best['parameter_id']==3 and best['checkpoint_id']==report['checkpoints'][1]
    assert load_checkpoint(db,ident)[1]['sha256']==base_sha
    directory=tmp_path/'runs'/'curves-test'
    assert json.loads((directory/'best.json').read_text(encoding='utf-8'))['parameter_id']==3
    log=events(directory/'tensorboard')
    assert [e.value for e in log.Scalars('validation/perplexity')]==[3.,2.,4.]
    assert [e.value for e in log.Scalars('validation/loss')]==[1.,2.,.5]
    assert len(log.Scalars('book_ce/train'))==3 and len(log.Scalars('loss/logical_total'))==3
    capsys.readouterr(); before=db.read_bytes()
    admin(['--db',str(db),'--family','HaiKu','get-best','--run-id',report['run_id']])
    assert json.loads(capsys.readouterr().out)['parameter_id']==3 and db.read_bytes()==before


def test_interruption_in_validation_saves_finished_training_epoch_without_partial_best(book_run,tmp_path,monkeypatch):
    import training.__main__ as cli
    _,db,ident,books=book_run; stop=threading.Event()
    def interrupt(*a,**kw):
        stop.set(); raise WorkInterrupted()
    monkeypatch.setattr(cli,'evaluate_books',interrupt)
    report=cli.main(cli_args(db,books,tmp_path),cancel_event=stop)
    assert report['interrupted'] and report['completed_epochs_this_run']==1 and report['best'] is None
    assert len(list_parameter_configs('HaiKu',path=db))==2
    _,metadata,_=load_checkpoint(db,report['checkpoints'][0])
    metrics=json.loads(metadata['metrics_json'])
    assert metrics['validation'] is None and metrics['validation_interrupted']
    assert not (tmp_path/'runs'/'curves-test'/'best.json').exists()
    with pytest.raises(ValueError): get_best_parameter_config('HaiKu',report['run_id'],path=db)


def test_explicit_validation_requires_heldout_data(book_run,tmp_path):
    import training.__main__ as cli
    _,db,ident,books=book_run
    with connect(books) as conn: conn.execute("DELETE FROM games WHERE split='validation'")
    with pytest.raises(SystemExit) as error: cli.main(cli_args(db,books,tmp_path)+['--validation','books'])
    assert error.value.code==1 and len(list_parameter_configs('HaiKu',path=db))==1


def test_standalone_evaluation_crops_all_heldout_moves_without_parameter_or_book_writes(book_run,tmp_path,capsys):
    from training.evaluate import main
    _,db,ident,books=book_run
    model_before=db.read_bytes(); books_before=books.read_bytes()
    report=main(['--family','HaiKu','--id','1','--model-db',str(db),'--book-db',str(books),
        '--L','2','--n','2','--micro-batch','2','--cpu-threads','2','--device','cpu',
        '--no-progress','--output',str(tmp_path/'evaluation')])
    assert report['source_games']==2 and report['segments']==3
    assert report['histogram']['total']==3 and report['steps']==4 and report['padding_steps']==2
    assert db.read_bytes()==model_before and books.read_bytes()==books_before
    assert len(Path(report['artifacts']['samples']).read_text(encoding='utf-8').splitlines())==3
    assert '切段后）：3' in capsys.readouterr().out


def test_dashboard_launcher_configures_live_reload_without_gpu(monkeypatch,tmp_path):
    import importlib
    program=importlib.import_module('tensorboard.program')
    from training.dashboard import launch_tensorboard
    arguments=[]
    class Board:
        def configure(self,*,argv): arguments.extend(argv)
        def launch(self): return 'http://localhost:6006/'
    monkeypatch.setattr(program,'TensorBoard',Board)
    assert launch_tensorboard(tmp_path)=='http://localhost:6006/'
    assert arguments[arguments.index('--reload_interval')+1]=='1'
    assert arguments[arguments.index('--host')+1]=='127.0.0.1'


def test_chinese_viewer_reads_newly_flushed_points_without_restarting(tmp_path):
    from training.dashboard_data import CurveReader,sample_points
    reader=CurveReader(tmp_path)
    assert reader.runs()==[]
    curves=TrainingCurves(tmp_path/'中文训练',{})
    curves.writer.add_scalar('loss/segment_total',3.,1); curves.writer.flush()
    run='中文训练/tensorboard'
    assert reader.runs()==[run] and reader.read(run)['series']['loss/segment_total'][0][:2]==[1,3.]
    curves.writer.add_scalar('loss/segment_total',2.,2); curves.writer.flush()
    assert [p[:2] for p in reader.read(run)['series']['loss/segment_total']]==[[1,3.],[2,2.]]
    with pytest.raises(ValueError): reader.read('../other/tensorboard')
    curves.close()
    points=list(range(10000)); sampled=sample_points(points)
    assert len(sampled)==2000 and sampled[0]==0 and sampled[-1]==9999


def test_chinese_page_contains_distinct_axis_units():
    from training import dashboard
    html=Path(dashboard.__file__).with_name('dashboard.html').read_text(encoding='utf-8')
    assert '横轴：训练轮次' in html and '纵轴：合法动作交叉熵' in html
    assert '反向传播段次数' in html and '累计参数更新次数' in html
    assert 'setInterval(refresh,1000)' in html

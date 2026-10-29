"""KV windows slide within a sequence and never leak into another batch."""
import copy
import json
import weakref

import pytest
torch=pytest.importorskip('torch')
from models.common import Attention,KVCache
from models.HaiKu.config import Haiku5Config
from models.HaiKu.network import Haiku5
from models.Opus.config import OpusConfig
from models.Opus.network import Opus5
from training.engine import iter_chunks
from training.loss import split_output


def tiny(family):
    options=dict(dim=16,ffn_dim=32,cnn_channels=8,checkpoint_attention=False,dropout=0)
    if family=='HaiKu': return Haiku5(Haiku5Config(heads=4,context_plies=3,**options))
    return Opus5(OpusConfig(heads=(4,2,1,2,1),context_windows=(3,3,3,3,3),
                           value_hidden=8,**options))


def test_haiku_default_wires_all_nine_kv_windows_to_30_and_evicts_oldest_board():
    assert Haiku5Config().context_plies==30
    model=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8))
    cached=[m for m in model.modules() if isinstance(m,Attention) and m.limit==30]
    # Each core's two history and one cross layers use caches; spatial layers
    # share the window config but never receive a KVCache.
    assert len(cached)==15 and sum(len(c.history)+1 for c in model.cores)==9
    kv=KVCache()
    for step in range(35): kv.append(torch.tensor(step),torch.tensor(-step),torch.tensor(True),30)
    assert len(kv.keys)==len(kv.values)==len(kv.valid)==30
    assert [int(k) for k in kv.keys]==list(range(5,35))


@pytest.mark.parametrize('family,entry',[('HaiKu','forward'),('HaiKu','common_chunks'),
                                        ('Opus','forward'),('Opus','common_chunks'),('Opus','opus_chunks')])
def test_same_size_next_batch_gets_empty_cache_and_time_zero_but_chunks_retain_history(family,entry,monkeypatch):
    torch.set_num_threads(2); torch.manual_seed(27)
    model=tiny(family).eval(); reference=copy.deepcopy(model)
    created=[]; lengths_before=[]; times=[]
    original_new=model.new_cache; original_step=model.forward_step
    def new_cache(batch_size):
        cache=original_new(batch_size)
        assert cache.time==0
        assert all(not kv.keys and not kv.values and not kv.valid
                   for core in cache.cores for kv in (*core.history,core.cross))
        created.append(weakref.ref(cache))
        return cache
    def forward_step(x,cache=None,valid=None):
        times.append(0 if cache is None else cache.time)
        return original_step(x,cache,valid)
    monkeypatch.setattr(model,'new_cache',new_cache)
    monkeypatch.setattr(model,'forward_step',forward_step)
    def before_attention(module,args): lengths_before.append(len(args[3].keys))
    hook=model.cores[0].history[0].attn.register_forward_pre_hook(before_attention)
    def run(x):
        lengths=torch.tensor([len(x)])
        if entry=='forward': return split_output(model(x,lengths))
        parts=(iter_chunks(model,x,lengths,2) if entry=='common_chunks'
               else model.forward_chunks(x,lengths,2))
        outputs=[split_output(part[2]) for part in parts]
        return tuple(torch.cat([out[i] for out in outputs]) if outputs[0][i] is not None else None
                     for i in (0,1))
    with torch.inference_mode():
        run(torch.randn(5,1,15,10,9))
        assert created[-1]() is None  # Completed sequence retains no cache reference.
        x=torch.randn(3,1,15,10,9)
        actual=run(x); expected=split_output(reference(x))
    hook.remove()
    assert len(created)==2 and all(ref() is None for ref in created)
    assert times==[0,1,2,3,4,0,1,2]
    assert lengths_before==[0,1,2,3,3,0,1,2]
    for a,b in zip(actual,expected):
        if a is not None: assert torch.equal(a,b)


@pytest.mark.parametrize('window',[None,7])
def test_book_training_changes_loaded_haiku_window_and_saves_effective_value(book_run,tmp_path,window):
    from training.__main__ import main
    from model_store import load_checkpoint
    model,database,ident,books=book_run
    args=['--family','HaiKu','--id','1','--source','books','--model-db',str(database),
          '--book-db',str(books),'--epochs','1','--T','2','--validation','none',
          '--batch-size','2','--micro-batch','1','--device','cpu','--no-progress','--smoke',
          '--output',str(tmp_path/'runs')]
    if window is not None: args+=['--context-plies',str(window)]
    result=main(args)
    restored,_,state=load_checkpoint(database,result['checkpoints'][0],resume=True)
    expected=30 if window is None else window
    config=json.loads((tmp_path/'runs'/result['run_id']/'config.json').read_text(encoding='utf-8'))
    assert restored.config.context_plies==state['context_plies']==config['context_plies']==expected
    original,_,_=load_checkpoint(database,ident)
    assert original.config.context_plies==120


def test_ppo_preserves_sampling_window_and_rejects_an_override_before_writing(book_run,tmp_path):
    from training.__main__ import main
    from selfplay.generator import SelfPlayGenerator,SelfPlayRules
    from model_store import load_checkpoint
    _,database,ident,_=book_run
    generator=SelfPlayGenerator.from_version('HaiKu',ident,model_db=database,
                                             rules=SelfPlayRules(max_plies=2))
    data=tmp_path/'selfplay.sqlite3'; generated=generator.generate(1,path=data)
    output=tmp_path/'runs'
    args=['--family','HaiKu','--id','1','--source','selfplay','--model-db',str(database),
          '--selfplay-db',str(data),'--rollout-run',generated['run_id'],'--epochs','1',
          '--batch-size','1','--micro-batch','1','--C','2','--device','cpu','--smoke','--no-progress',
          '--output',str(output)]
    with pytest.raises(SystemExit) as error: main(args+['--context-plies','30'])
    assert error.value.code==1 and not output.exists()
    result=main(args)
    restored,_,state=load_checkpoint(database,result['checkpoints'][0],resume=True)
    assert restored.config.context_plies==state['context_plies']==120


@pytest.fixture
def book_run(tmp_path):
    from data_pipeline.database import connect
    from data_pipeline.notation import STANDARD_FEN
    from encoder import move_to_action
    from model_store import save_checkpoint
    model=Haiku5(Haiku5Config(dim=16,heads=4,ffn_dim=32,cnn_channels=8,
                            context_plies=120,checkpoint_attention=False))
    database=tmp_path/'models.sqlite3'
    ident=save_checkpoint(model,path=database,run_id='old',epoch=1,global_step=1,metrics={})
    books=tmp_path/'books.sqlite3'
    with connect(books) as conn:
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                     ('book',STANDARD_FEN,json.dumps([move_to_action((6,0,5,0))]),1,'*','human_master','train','book','{}'))
    return model,database,ident,books

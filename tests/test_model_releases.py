import json
import sqlite3

import pytest

torch=pytest.importorskip('torch')
from ai import load_network
from model_store import (init_store,list_models,list_hint_models,load_hint_model,
                         publish_model,unpublish_model,save_checkpoint,main as admin)
from models.haiku5.config import Haiku5Config
from models.haiku5.network import Haiku5


@pytest.fixture
def store(tmp_path):
    path=tmp_path/'models.sqlite3'
    net=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8))
    ids=[]
    for epoch in (1,2):
        with torch.no_grad(): net.policy_head.bias.fill_(epoch)
        ids.append(save_checkpoint(net,path=path,run_id='private-training-run',epoch=epoch,
                                   global_step=epoch*10,metrics={'val_loss':4-epoch},best=True))
    return path,net,ids


def test_public_catalog_requires_explicit_binding_and_uses_selected_weights(store):
    path,net,(first,second)=store
    assert len(list_models(path))==2
    assert list_hint_models(path)==[] and load_network(path) is None
    publish_model('HaiKu-5',first,path=path)
    publish_model('HaiKu-5.1',second,path=path)
    assert list_hint_models(path)==[{'name':'HaiKu-5'},{'name':'HaiKu-5.1'}]
    loaded=load_network(path)
    assert loaded.display_name=='HaiKu-5' and torch.all(loaded.policy_head.bias==1)
    assert torch.all(load_network(path,'HaiKu-5.1').policy_head.bias==2)
    # A later epoch must not change published choices or their weights.
    save_checkpoint(net,path=path,run_id='private-training-run',epoch=3,global_step=30,metrics={})
    assert len(list_hint_models(path))==2
    assert torch.all(load_hint_model('HaiKu-5',path=path).policy_head.bias==1)
    publish_model('HaiKu-5',second,path=path)
    assert len(list_hint_models(path))==2
    assert torch.all(load_hint_model('HaiKu-5',path=path).policy_head.bias==2)


def test_unpublish_keeps_checkpoints_and_blocks_public_loading(store):
    path,_,ids=store
    publish_model('HaiKu-5',ids[0],path=path)
    unpublish_model('HaiKu-5',path=path)
    assert list_hint_models(path)==[] and len(list_models(path))==2
    with pytest.raises(ValueError,match='暂不可用'): load_hint_model('HaiKu-5',path=path)
    publish_model('HaiKu-5',ids[1],path=path)
    assert list_hint_models(path)==[{'name':'HaiKu-5'}]


def test_smoke_untrained_and_missing_ids_cannot_replace_public_model(store):
    path,net,ids=store
    publish_model('HaiKu-5',ids[0],path=path)
    smoke=save_checkpoint(net,path=path,run_id='smoke',epoch=1,global_step=1,metrics={},published=False)
    untrained=save_checkpoint(net,path=path,run_id='zero',epoch=0,global_step=0,metrics={})
    for ident in (smoke,untrained,'missing'):
        with pytest.raises(ValueError): publish_model('HaiKu-5',ident,path=path)
    assert torch.all(load_hint_model('HaiKu-5',path=path).policy_head.bias==1)
    with pytest.raises(ValueError): publish_model(' \n ',ids[0],path=path)
    with pytest.raises(ValueError): publish_model('bad\nname',ids[0],path=path)
    with pytest.raises(ValueError): load_hint_model('missing',path=path)


def test_legacy_database_is_migrated_without_rewriting_checkpoints(store):
    path,_,ids=store
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE model_releases')
        before=conn.execute('SELECT checkpoint_id,sha256 FROM checkpoints ORDER BY checkpoint_id').fetchall()
    assert list_hint_models(path)==[]
    publish_model('HaiKu-5',ids[1],path=path)
    with sqlite3.connect(path) as conn:
        assert before==conn.execute('SELECT checkpoint_id,sha256 FROM checkpoints ORDER BY checkpoint_id').fetchall()
        assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    assert torch.all(load_hint_model('HaiKu-5',path=path).policy_head.bias==2)


def test_backend_admin_lists_training_ids_metrics_and_binds_alias(store,capsys):
    path,net,ids=store
    smoke=save_checkpoint(net,path=path,run_id='smoke',epoch=1,global_step=1,metrics={},published=False)
    admin(['--db',str(path),'list-checkpoints'])
    rows={row['checkpoint_id']:row for row in json.loads(capsys.readouterr().out)}
    assert rows[ids[1]]['epoch']==2 and rows[ids[1]]['metrics']=={'val_loss':2}
    assert rows[ids[1]]['can_publish'] and not rows[smoke]['can_publish']
    admin(['--db',str(path),'publish','--name','HaiKu-5','--run-id','private-training-run','--alias','best'])
    assert '已发布模型' in capsys.readouterr().out
    admin(['--db',str(path),'list-releases'])
    assert json.loads(capsys.readouterr().out)[0]['checkpoint_id']==ids[1]
    admin(['--db',str(path),'unpublish','--name','HaiKu-5'])
    assert list_hint_models(path)==[]


def test_startup_name_and_id_binding_is_persistent_and_changeable(store,monkeypatch):
    import main
    import ui
    path,_,ids=store
    calls=[]
    def open_game(**kwargs):
        net=load_network(kwargs['model_path'],kwargs['model_name'])
        calls.append((net.display_name,float(net.policy_head.bias[0].detach()),kwargs['initial_mode']))
    monkeypatch.setattr(ui,'run_app',open_game)
    base=['gui','--mode','pvp','--model-db',str(path),'--model','HaiKu-5']
    main.main(base+['--checkpoint',ids[0]])
    main.main(base)  # no ID needed on subsequent starts
    main.main(base+['--checkpoint',ids[1]])
    assert calls==[('HaiKu-5',1.,'pvp'),('HaiKu-5',1.,'pvp'),('HaiKu-5',2.,'pvp')]
    assert list_hint_models(path)==[{'name':'HaiKu-5'}]


def test_invalid_startup_binding_stops_before_opening_game(store,monkeypatch):
    import main
    import ui
    path,_,ids=store
    calls=[]
    monkeypatch.setattr(ui,'run_app',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(SystemExit) as missing_name:
        main.main(['gui','--model-db',str(path),'--checkpoint',ids[0]])
    with pytest.raises(SystemExit) as invalid_id:
        main.main(['gui','--model-db',str(path),'--model','HaiKu-5','--checkpoint','missing'])
    assert missing_name.value.code==invalid_id.value.code==2
    assert not calls and list_hint_models(path)==[]


def test_missing_or_initial_store_has_no_public_models(tmp_path):
    path=tmp_path/'initial.sqlite3'
    assert list_hint_models(path)==[] and list_models(path)==[]
    init_store(path)
    assert list_hint_models(path)==[] and load_network(path) is None

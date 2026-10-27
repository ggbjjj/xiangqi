import threading
import time
from types import SimpleNamespace

import pygame
import pytest

import ui
from ai import NeuralAI
from ui import OpponentModelMenu,XiangqiUI


def wait_loading(menu):
    deadline=time.monotonic()+5
    while menu.loading and time.monotonic()<deadline:
        menu._poll_loading(); time.sleep(.01)
    assert not menu.loading


def test_requires_explicit_selection_and_start_loads_only_chosen_name(monkeypatch):
    monkeypatch.setattr(ui,'list_hint_models',lambda path:[{'name':'HaiKu-5'},{'name':'HaiKu-5.1'}])
    calls=[]; net=SimpleNamespace(display_name='HaiKu-5.1')
    def load(path,name):
        calls.append((path,name)); return net
    monkeypatch.setattr(ui,'load_network',load)
    menu=OpponentModelMenu('pve_black','private.sqlite3')
    try:
        menu._draw(); menu._start()
        assert menu.result is None and not menu.loading and not calls
        menu._on_click(menu.model_rects[1][0].center)
        assert menu.selected_name=='HaiKu-5.1' and not calls
        menu._on_click(menu.start_rect.center); wait_loading(menu)
        assert calls==[('private.sqlite3','HaiKu-5.1')]
        assert menu.result==('start',net)
    finally: pygame.quit()


def test_preselection_still_requires_confirmation_and_no_private_details(monkeypatch):
    monkeypatch.setattr(ui,'list_hint_models',lambda path:[{'name':'HaiKu-5'},{'name':'HaiKu-5.1'}])
    shown=[]; original=ui.text_fit
    def record(surface,text,*args,**kwargs):
        shown.append(text); return original(surface,text,*args,**kwargs)
    monkeypatch.setattr(ui,'text_fit',record)
    menu=OpponentModelMenu('pve_red',preferred_name='HaiKu-5.1')
    try:
        menu._draw()
        assert menu.selected_name=='HaiKu-5.1' and menu.result is None and not menu.loading
        assert 'HaiKu-5' in shown and 'HaiKu-5.1' in shown
        assert not any(word in ' '.join(shown) for word in ('epoch','Epoch','训练','checkpoint','sqlite3','run_id'))
    finally: pygame.quit()


def test_empty_catalog_keeps_explicit_builtin_choice_without_torch(monkeypatch):
    monkeypatch.setattr(ui,'list_hint_models',lambda path:[])
    monkeypatch.setattr(ui,'load_network',lambda *args:pytest.fail('builtin selection must not load torch'))
    menu=OpponentModelMenu('pve_red')
    try:
        menu._draw()
        assert menu.choices==[{'name':'内置搜索'}] and menu.selected_name is None
        menu._on_click(menu.model_rects[0][0].center); menu._start()
        assert menu.result==('start',None)
    finally: pygame.quit()


def test_model_load_failure_stays_on_selection_without_fallback(monkeypatch):
    monkeypatch.setattr(ui,'list_hint_models',lambda path:[{'name':'HaiKu-5'}])
    def fail(*args): raise ValueError('private checkpoint ID and training epoch 99')
    monkeypatch.setattr(ui,'load_network',fail)
    menu=OpponentModelMenu('pve_red',preferred_name='HaiKu-5')
    try:
        menu._start(); wait_loading(menu)
        assert menu.result is None and menu.selected_name=='HaiKu-5'
        assert menu.message=='该模型暂不可用，请重试或重新选择'
    finally: pygame.quit()


def test_back_during_load_discards_late_model(monkeypatch):
    monkeypatch.setattr(ui,'list_hint_models',lambda path:[{'name':'HaiKu-5'}])
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    def load(*args):
        started.set(); release.wait(3); finished.set(); return object()
    monkeypatch.setattr(ui,'load_network',load)
    menu=OpponentModelMenu('pve_black',preferred_name='HaiKu-5')
    try:
        menu._start(); assert started.wait(1)
        menu._on_click(menu.back_rect.center)
        release.set(); assert finished.wait(1)
        menu._poll_loading()
        assert menu.result==('back',None)
    finally: release.set(); pygame.quit()


@pytest.mark.parametrize('mode',['pve_red','pve_black'])
def test_direct_launch_also_selects_model_before_constructing_match(monkeypatch,mode):
    net=object(); events=[]
    class Selection:
        def __init__(self,chosen_mode,path,preferred):
            assert chosen_mode==mode and preferred=='HaiKu-5.1'
            self.selected_name='HaiKu-5.1'
        def run(self):
            events.append('selected'); return 'start',net
    class Match:
        restart_requested=False
        def __init__(self,chosen_mode,chosen_net,**kwargs):
            assert events==['selected'] and chosen_mode==mode and chosen_net is net
            events.append('match')
        def run(self): return False
    monkeypatch.setattr(ui,'OpponentModelMenu',Selection)
    monkeypatch.setattr(ui,'XiangqiUI',Match)
    ui.run_app(initial_mode=mode,model_name='HaiKu-5.1')
    assert events==['selected','match']


def test_cancel_selection_does_not_start_match(monkeypatch):
    class Selection:
        def __init__(self,*args): pass
        def run(self): return 'back',None
    monkeypatch.setattr(ui,'OpponentModelMenu',Selection)
    monkeypatch.setattr(ui.MainMenu,'run',lambda self:None)
    monkeypatch.setattr(ui,'XiangqiUI',lambda *args,**kwargs:pytest.fail('cancelled selection must not start a game'))
    ui.run_app(initial_mode='pve_black')


def test_new_game_returns_to_selection_and_can_choose_different_model(monkeypatch):
    nets=[object(),object()]; choices=[]; matches=[]
    class Selection:
        def __init__(self,*args): self.selected_name=f'model-{len(choices)}'
        def run(self):
            index=len(choices); choices.append(index); return 'start',nets[index]
    class Match:
        def __init__(self,mode,net,**kwargs):
            matches.append(net); self.restart_requested=len(matches)==1
        def run(self): return False
    monkeypatch.setattr(ui,'OpponentModelMenu',Selection)
    monkeypatch.setattr(ui,'XiangqiUI',Match)
    ui.run_app(initial_mode='pve_red')
    assert choices==[0,1] and matches==nets


@pytest.fixture
def published_pair(tmp_path):
    torch=pytest.importorskip('torch'); torch.set_num_threads(2)
    from models.haiku5.config import Haiku5Config
    from models.haiku5.network import Haiku5
    from model_store import save_checkpoint,publish_model,load_hint_model
    db=tmp_path/'models.sqlite3'; ids=[]
    net=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8))
    for epoch,name in [(1,'HaiKu-5'),(2,'HaiKu-5.1')]:
        with torch.no_grad(): net.policy_head.bias.fill_(epoch)
        ident=save_checkpoint(net,path=db,run_id='fixture',epoch=epoch,global_step=epoch,metrics={})
        publish_model(name,ident,path=db); ids.append(ident)
    return db,load_hint_model('HaiKu-5',path=db),ids


def test_hint_other_model_and_backend_rebinding_cannot_change_opponent(published_pair):
    from model_store import publish_model
    db,net,ids=published_pair
    app=XiangqiUI('pve_red',net,model_db=db)
    try:
        original_ai=app.ai
        app._start_hint({'name':'HaiKu-5.1'})
        deadline=time.monotonic()+5
        while app.hint_busy and time.monotonic()<deadline:
            app._poll_hints(); time.sleep(.01)
        assert len(app.hints)==3 and app.hint_model_name=='HaiKu-5.1'
        assert app.ai is original_ai and app.ai.net is net and app.opponent_name=='HaiKu-5'
        publish_model('HaiKu-5',ids[1],path=db)
        assert app.game.move(6,0,5,0)=='ok' and app.game.move(3,0,4,0)=='ok'
        app.action_undo()
        assert app.ai.net is net and float(app.ai.net.policy_head.bias[0].detach())==1.
        app.action_new()
        assert app.restart_requested and not app.running
    finally: app._invalidate_ai(); pygame.quit()


@pytest.mark.parametrize('move,error',[(None,'private checkpoint secret'),(None,None),((0,0,1,0),None)])
def test_opponent_failure_keeps_model_and_retry_uses_same_weights(published_pair,move,error):
    db,net,_=published_pair
    app=XiangqiUI('pve_black',net,model_db=db)
    try:
        original=app.ai
        app._results.put((app._generation,app.game.turn,move,error))
        app._poll_ai()
        assert app.ai_failed and app.ai is original and isinstance(app.ai,NeuralAI)
        assert not app.game.history and 'secret' not in app.message
        app._ai_move(); assert not app.thinking
        app._draw(); app.action_pause()
        assert not app.ai_failed and app.ai.net is net and app.opponent_name=='HaiKu-5'
    finally: app._invalidate_ai(); pygame.quit()

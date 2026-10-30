"""Spectator setup and legal temperature/epsilon sampling share rollout semantics."""
from concurrent.futures import CancelledError
import threading

import pygame
import pytest

torch=pytest.importorskip('torch')

import ui
from ai import NeuralAI
from encoder import action_to_move,legal_action_mask,move_to_action
from game import Game
from training.exploration import Exploration


class FixedPolicy(torch.nn.Module):
    def __init__(self,board,turn,name='Opus-5'):
        super().__init__()
        self.display_name=name
        legal=torch.from_numpy(legal_action_mask(board,turn)).bool()
        logits=torch.full((8100,),100.)  # Illegal moves must never enter sampling.
        logits[legal]=torch.linspace(-2,4,int(legal.sum()))
        self.register_buffer('logits',logits)
        self.calls=0

    def forward_step(self,x,cache=None):
        assert not self.training and not torch.is_grad_enabled()
        self.calls+=1
        return self.logits[None].expand(len(x),-1),cache


@pytest.mark.parametrize('temperature,epsilon',[(1.2,.02),(1.,0.),(.4,1.)])
def test_moves_follow_full_legal_mixture_and_hints_keep_original_probabilities(temperature,epsilon):
    game=Game(); net=FixedPolicy(game.board,game.turn)
    legal=torch.from_numpy(legal_action_mask(game.board,game.turn)).bool()
    q=(net.logits/temperature).masked_fill(~legal,-torch.inf).softmax(-1)
    mu=(1-epsilon)*q+epsilon*legal/float(legal.sum())
    rng=torch.Generator().manual_seed(723)
    reference_rng=torch.Generator().manual_seed(723)
    ai=NeuralAI(net,exploration=Exploration(temperature,epsilon),generator=rng)
    before=rng.get_state().clone()
    ranked=ai.rank_moves(game.board,game.turn,topk=3)
    assert torch.equal(before,rng.get_state())  # Asking for hints cannot consume move randomness.
    raw=net.logits.masked_fill(~legal,-torch.inf).softmax(-1)
    assert ranked[0][0]==action_to_move(int(raw.argmax()))
    assert ranked[0][1]==pytest.approx(float(raw.max()))
    actual=[]
    for _ in range(16):
        expected=int(torch.multinomial(mu,1,generator=reference_rng))
        move=ai.choose_move(game.board,game.turn,history=game.history)
        assert move==action_to_move(expected) and legal[move_to_action(move)]
        actual.append(move)
    assert any(move!=ranked[0][0] for move in actual)
    assert net.calls==1  # Reuse inference at an unchanged position without resetting history.
    if epsilon==1:
        assert any(move not in {m for m,_ in ranked} for move in actual)


def test_greedy_still_selects_top_policy_and_cancelled_sampling_cannot_play():
    game=Game(); net=FixedPolicy(game.board,game.turn)
    ai=NeuralAI(net)
    best=ai.rank_moves(game.board,game.turn,topk=1)[0][0]
    assert ai.choose_move(game.board,game.turn)==best
    ai=NeuralAI(net,exploration=Exploration())
    cancelled=threading.Event(); cancelled.set()
    with pytest.raises(CancelledError):
        ai.choose_move(game.board,game.turn,cancel_event=cancelled)
    assert not game.history


def key(menu,key,unicode='',mod=0):
    menu._on_key(pygame.event.Event(pygame.KEYDOWN,key=key,unicode=unicode,mod=mod))


def test_options_form_edits_parameters_and_switches_strategy():
    menu=ui.SelfPlayOptionsMenu('Opus-5')
    try:
        assert menu.strategy=='explore' and menu.result is None
        assert menu.values=={'temperature':'1.2','epsilon':'0.02'}
        menu._on_click(menu.input_rects['temperature'].center)
        for c in '0.8': key(menu,ord(c),c)
        key(menu,pygame.K_TAB)
        for c in '0.15': key(menu,ord(c),c)
        key(menu,pygame.K_a,'a',pygame.KMOD_CTRL)
        key(menu,pygame.K_BACKSPACE)
        for c in '2e-2': key(menu,ord(c),c)
        menu._draw()
        menu._on_click(menu.strategy_rects['greedy'].center)
        key(menu,pygame.K_TAB)  # Disabled parameter fields cannot be edited in greedy mode.
        assert menu.active is None
        menu._on_click(menu.strategy_rects['explore'].center)
        menu._on_click(menu.start_rect.center)
        assert menu.result==('start',{'strategy':'explore','temperature':.8,'epsilon':.02})
    finally: pygame.quit()


@pytest.mark.parametrize('temperature,epsilon',[
    ('0','.02'),('-1','.02'),('nan','.02'),('inf','.02'),('', '.02'),
    ('1.2','-0.1'),('1.2','1.1'),('1.2','nan'),('1.2','inf'),
])
def test_invalid_parameters_stay_on_form_but_do_not_block_explicit_greedy(temperature,epsilon):
    menu=ui.SelfPlayOptionsMenu('HaiKu-5',temperature=temperature,epsilon=epsilon)
    try:
        menu._start()
        assert menu.result is None and '有效数字' in menu.message
        menu._on_click(menu.strategy_rects['greedy'].center)
        menu._start()
        assert menu.result==('start',{'strategy':'greedy'})
    finally: pygame.quit()


def test_builtin_search_and_back_do_not_start_neural_exploration():
    menu=ui.SelfPlayOptionsMenu('内置搜索',can_explore=False)
    try:
        menu._on_click(menu.strategy_rects['explore'].center)
        menu._on_click(menu.input_rects['temperature'].center)
        assert menu.strategy=='greedy' and menu.active is None
        menu._start()
        assert menu.result==('start',{'strategy':'greedy'})
        menu.result=None
        key(menu,pygame.K_ESCAPE)
        assert menu.result==('back',None)
    finally: pygame.quit()


@pytest.mark.parametrize('name',['HaiKu-5','Opus-5'])
def test_both_colors_use_fixed_exploration_after_undo_and_retry(name):
    game=Game(); net=FixedPolicy(game.board,game.turn,name)
    app=ui.XiangqiUI('ai_vs_ai',net,watch_options={'strategy':'explore','temperature':.8,'epsilon':.1})
    try:
        assert app.ai.exploration==Exploration(.8,.1)
        for _ in range(2):
            assert app._is_ai_turn()
            move=app.ai.choose_move(app.game.board,app.game.turn,history=app.game.history)
            assert app.game.move(*move)=='ok'
        app.action_undo()
        assert len(app.game.history)==1 and app.ai.net is net
        assert app.ai.exploration==Exploration(.8,.1)
        app.ai_failed=True; app.action_pause()
        assert not app.ai_failed and app.ai.exploration==Exploration(.8,.1)
        assert app.ai.net is net
        app._draw()
    finally: app._invalidate_ai(); pygame.quit()


@pytest.mark.parametrize('mode,options',[('pve_red',None),('pve_black',None),('ai_vs_ai',{'strategy':'greedy'})])
def test_pve_and_explicit_greedy_keep_argmax(mode,options):
    game=Game(); net=FixedPolicy(game.board,game.turn)
    app=ui.XiangqiUI(mode,net,watch_options=options)
    try:
        assert app.ai.exploration is None
        assert app.ai.choose_move(app.game.board,app.game.turn)==app.ai.rank_moves(app.game.board,app.game.turn,topk=1)[0][0]
    finally: app._invalidate_ai(); pygame.quit()


def test_run_app_requires_settings_before_first_move_and_retains_them_on_new_game(monkeypatch):
    calls=[]; net=object()
    settings={'strategy':'explore','temperature':.7,'epsilon':.12}
    class Models:
        selected_name='Opus-5'
        def __init__(self,*args,**kwargs): pass
        def run(self): calls.append('model'); return 'start',net
    class Options:
        count=0
        def __init__(self,name,**kwargs):
            assert name=='Opus-5' and kwargs['can_explore'] is True
            if self.count: assert {k:kwargs[k] for k in settings}==settings
            self.settings=settings
        def run(self):
            calls.append('settings'); Options.count+=1
            return ('back',None) if self.count==1 else ('start',settings)
    class Match:
        count=0
        def __init__(self,mode,selected,**kwargs):
            calls.append('match')
            assert mode=='ai_vs_ai' and selected is net and kwargs['watch_options']==settings
            Match.count+=1; self.restart_requested=self.count==1
        def run(self): return False
    monkeypatch.setattr(ui,'OpponentModelMenu',Models)
    monkeypatch.setattr(ui,'SelfPlayOptionsMenu',Options)
    monkeypatch.setattr(ui,'XiangqiUI',Match)
    ui.run_app(initial_mode='ai_vs_ai')
    assert calls==['model','settings','model','settings','match','model','settings','match']

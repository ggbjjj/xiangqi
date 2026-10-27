import time
import pygame
import pytest

from ui import XiangqiUI,board_to_pixel


def test_hint_requires_selection_and_empty_store_is_explicit(tmp_path):
    app=XiangqiUI('pvp',model_db=tmp_path/'empty.sqlite3')
    try:
        app._on_click(app.hint_rect.center); app._draw()
        assert app.model_dialog and not app.model_choices and not app.hint_busy
        assert '重试' in app.dialog_message and not app.hints
        app._on_click(board_to_pixel(6,0))
        assert app.selected is None  # modal prevents accidental board moves
        app._escape(); assert not app.model_dialog
    finally: app._invalidate_ai(); pygame.quit()


@pytest.mark.parametrize('mode,enabled',[('pvp',True),('pve_red',True),('pve_black',False),('ai_vs_ai',False)])
def test_hint_human_turns_only(tmp_path,mode,enabled):
    app=XiangqiUI(mode,model_db=tmp_path/'models.sqlite3')
    try:
        app.action_hint(); assert app.model_dialog==enabled
        app.game.status='和棋'; app.model_dialog=False
        app.action_hint(); assert not app.model_dialog
    finally: app._invalidate_ai(); pygame.quit()


def test_real_checkpoint_popup_worker_top3_and_click(tmp_path):
    torch=pytest.importorskip('torch'); torch.set_num_threads(2)
    from models.haiku5.network import Haiku5
    from models.haiku5.config import Haiku5Config
    from model_store import save_checkpoint,publish_model
    from ai import NeuralAI
    net=Haiku5(Haiku5Config(dim=16,ffn_dim=32,cnn_channels=8)).eval()
    path=tmp_path/'models.sqlite3'
    ident=save_checkpoint(net,path=path,run_id='test-fixture',epoch=1,global_step=1,metrics={})
    publish_model('HaiKu-5',ident,path=path)
    app=XiangqiUI('pvp',model_db=path)
    try:
        app.action_hint(); app._draw()
        assert len(app.model_rects)==1 and not app.hints
        assert app.model_choices==[{'name':'HaiKu-5'}]
        app._on_click(app.model_rects[0][0].center)
        assert not app.model_dialog
        deadline=time.monotonic()+10
        while app.hint_busy and time.monotonic()<deadline:
            app._poll_hints(); time.sleep(.01)
        assert len(app.hints)==3 and not app.hint_busy,app.message
        assert app.hint_model_name=='HaiKu-5' and 'test-fixture' not in app.message
        expected=NeuralAI(net).rank_moves(app.game.board,app.game.turn)
        assert [m for m,p in app.hints]==[m for m,p in expected]
        assert all(a[1]>=b[1] for a,b in zip(app.hints,app.hints[1:]))
        app._draw(); move=app.hints[0][0]
        app._on_click(app.hint_rows[0][0].center)
        assert app.selected==move[:2] and len(app.game.history)==0
        app._on_click(board_to_pixel(*move[2:]))
        assert len(app.game.history)==1 and not app.hints
    finally: app._invalidate_ai(); pygame.quit()


def test_model_dialog_renders_only_public_names(monkeypatch):
    import ui
    app=XiangqiUI('pvp')
    try:
        app.model_choices=[{'name':'HaiKu-5'},{'name':'HaiKu-5.1'}]
        shown=[]
        original=ui.text_fit
        def record(surface,text,*args,**kwargs):
            shown.append(text)
            return original(surface,text,*args,**kwargs)
        monkeypatch.setattr(ui,'text_fit',record)
        app._draw_model_dialog()
        assert 'HaiKu-5' in shown and 'HaiKu-5.1' in shown
        assert not any(word in ' '.join(shown) for word in ('Epoch','epoch','训练','checkpoint','run_id','参数'))
        assert len(app.model_rects)==2
    finally: app._invalidate_ai(); pygame.quit()


def test_hint_failure_does_not_show_backend_details(monkeypatch):
    pytest.importorskip('torch')
    import ui
    def fail(*args,**kwargs):
        raise ValueError('checkpoint=private-id Epoch=99 secret/path')
    monkeypatch.setattr(ui,'load_hint_model',fail)
    app=XiangqiUI('pvp')
    try:
        app._start_hint({'name':'HaiKu-5'})
        deadline=time.monotonic()+10
        while app.hint_busy and time.monotonic()<deadline:
            app._poll_hints(); time.sleep(.01)
        assert not app.hint_busy and not app.hints
        assert app.message=='模型暂不可用，请稍后重试或选择其他模型。'
    finally: app._invalidate_ai(); pygame.quit()


def test_stale_hint_discarded_after_restart_undo_and_move():
    app=XiangqiUI('pvp')
    try:
        for action in (app.action_new,app.action_undo):
            old=app._hint_generation
            action()
            app._hint_results.put((old,app.game.turn,[((6,0,5,0),1.)],'test',None))
            app._poll_hints(); assert not app.hints
        old=app._hint_generation
        app.game.move(6,0,5,0); app._after_move()
        app._hint_results.put((old,'b',[((3,0,4,0),1.)],'test',None))
        app._poll_hints(); assert not app.hints
    finally: app._invalidate_ai(); pygame.quit()

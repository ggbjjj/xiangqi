import threading
import time

import pygame
import pytest

from ui import INFO_H, MainMenu, XiangqiUI, board_to_pixel, pixel_to_board


@pytest.fixture
def ui():
    app = XiangqiUI('pvp')
    yield app
    app._invalidate_ai()
    pygame.quit()


def wait_for_move(app, count):
    deadline = time.monotonic() + 8
    while len(app.game.history) < count and time.monotonic() < deadline:
        app._poll_ai()
        time.sleep(0.01)
    assert len(app.game.history) == count


def test_coordinates_and_clicks_match_drawn_board(ui):
    for r in range(10):
        for c in range(9):
            assert pixel_to_board(*board_to_pixel(r, c)) == (r, c)
    assert pixel_to_board(40, 20) is None
    assert board_to_pixel(0, 0)[1] - 28 > INFO_H
    ui._on_click(board_to_pixel(6, 0))
    assert ui.selected == (6, 0) and (5, 0) in ui.legal_targets
    ui._on_click(board_to_pixel(5, 0))
    assert len(ui.game.history) == 1 and ui.game.turn == 'b'
    ui._on_click(board_to_pixel(3, 0))
    ui._on_click(board_to_pixel(4, 0))
    ui._on_click(board_to_pixel(6, 2))
    ui.action_undo()
    assert len(ui.game.history) == 1
    assert ui.selected is None and not ui.legal_targets
    assert ui.last_move == (6, 0, 5, 0)
    ui._draw()


@pytest.mark.parametrize('mode,ai_turn', [('pvp', False), ('pve_red', False),
                                        ('pve_black', True), ('ai_vs_ai', True)])
def test_mode_initial_turn(mode, ai_turn):
    app = XiangqiUI(mode, depth=1)
    try:
        assert app._is_ai_turn() == ai_turn
        app._draw()
    finally:
        app._invalidate_ai()
        pygame.quit()


def test_ai_vs_ai_plays_both_sides_and_pauses():
    app = XiangqiUI('ai_vs_ai', depth=1)
    try:
        app._on_click(board_to_pixel(6, 0))
        assert app.selected is None
        app._ai_move()
        wait_for_move(app, 1)
        assert app.game.turn == 'b' and app._is_ai_turn()
        app._ai_move()
        wait_for_move(app, 2)
        assert app.game.turn == 'r'
        app.action_pause()
        app._ai_move()
        assert app.paused and not app.thinking
        app.action_undo()
        assert len(app.game.history) == 1 and app.paused
    finally:
        app._invalidate_ai()
        pygame.quit()


def test_new_game_discards_worker_result_without_blocking():
    app = XiangqiUI('pve_black')
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class SlowAI:
        def choose_move(self, board, turn, cancel_event=None):
            started.set()
            release.wait(3)
            finished.set()
            return (6, 0, 5, 0)

    try:
        app.ai = SlowAI()
        app._ai_move()
        assert started.wait(1)
        before = time.monotonic()
        app.action_new()
        app._draw()
        assert time.monotonic() - before < 1
        release.set()
        assert finished.wait(1)
        # 等待旧线程放入结果，再处理；旧结果不能作用于同样轮到红方的新局。
        result = app._results.get(timeout=1)
        app._results.put(result)
        app._poll_ai()
        assert not app.game.history and app.game.turn == 'r'
    finally:
        release.set()
        app._invalidate_ai()
        pygame.quit()


def test_terminal_result_stays_visible(ui):
    ui.game.status = '红方胜利！（将死）'
    ui._on_click(board_to_pixel(6, 0))
    assert ui.selected is None
    ui._draw()
    assert ui.game.is_over()


def test_close_exits_instead_of_reopening_menu(ui):
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert ui.run() is False
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert ui.run() is True


def test_menu_keyboard_selection():
    menu = MainMenu()
    try:
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4))
        assert menu.run() == 'ai_vs_ai'
    finally:
        pygame.quit()

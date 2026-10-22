import threading
from concurrent.futures import CancelledError

import pytest

from ai import SearchAI
from board import Board
from game import Game
from main import _parse_move
from piece import Piece


def position(*pieces):
    board = Board([[None] * 9 for _ in range(10)])
    for r, c, color, kind in pieces:
        board.grid[r][c] = Piece(color, kind)
    return board


@pytest.mark.parametrize('color,row,blocked', [('r', 7, 6), ('b', 2, 3)])
@pytest.mark.parametrize('kind', ['K', 'A'])
def test_palace_has_exactly_three_rows(color, row, blocked, kind):
    board = position((row, 4, color, kind))
    assert board.pseudo_legal_moves(row, 4)
    assert all(r != blocked for r, c in board.pseudo_legal_moves(row, 4))
    assert not board.in_palace(blocked, 4, color)


@pytest.mark.parametrize('move', [(-1, 0, 0, 0), (10, 0, 0, 0), (6, 9, 5, 9),
                                  (6, 0, -1, 0), (6, 0, 5, 9), (6, 0, 10, 0)])
def test_out_of_bounds_rejected_without_mutation(move):
    game = Game()
    before = game.board.signature()
    assert game.move(*move) == 'illegal'
    assert game.board.signature() == before
    assert game.history == []
    assert game.legal_moves_at(-1, 0) == []


def test_pvp_undo_exactly_one_move_and_pve_returns_to_player():
    game = Game()
    assert game.move(6, 0, 5, 0) == 'ok'
    assert game.move(3, 0, 4, 0) == 'ok'
    assert game.undo_turn() == 1
    assert len(game.history) == 1 and game.turn == 'b'
    assert game.undo_turn('r') == 1
    assert game.turn == 'r' and not game.history
    game.move(6, 0, 5, 0)
    game.move(3, 0, 4, 0)
    assert game.undo_turn('r') == 2
    game.move(6, 0, 5, 0)
    assert game.undo_turn('b') == 0  # AI 红方的开局不能算作黑方玩家落子
    game.move(3, 0, 4, 0)
    game.move(6, 2, 5, 2)
    assert game.undo_turn('b') == 2
    assert len(game.history) == 1 and game.turn == 'b'


def test_threefold_draw_and_undo_restores_state():
    game = Game()
    cycle = [(9, 1, 7, 2), (0, 1, 2, 2), (7, 2, 9, 1), (2, 2, 0, 1)]
    for move in cycle * 2:
        assert game.move(*move) == 'ok'
    assert game.is_over() and game.winner() is None
    assert '三次' in game.status
    assert game.all_legal_moves() == []
    assert game.undo()
    assert not game.is_over() and game.quiet_moves == 7
    assert game.move(*cycle[-1]) == 'ok'
    assert game.is_over()


def test_quiet_move_limit_and_capture_reset():
    game = Game()
    game.quiet_moves = 119
    game.move(6, 0, 5, 0)
    assert '120' in game.status
    game.undo()
    assert game.quiet_moves == 119 and not game.status
    game.board.grid[5][0] = Piece('b', 'P')
    game.move(6, 0, 5, 0)
    assert game.quiet_moves == 0 and not game.status


@pytest.mark.parametrize('text', ['60-50', '6,0 5,0', '6050', '60 50'])
def test_cli_formats(text):
    assert _parse_move(text) == (6, 0, 5, 0)


@pytest.mark.parametrize('text', ['69-50', '60-59', 'abc6050', '', '100-50', '-60-50'])
def test_cli_bad_input(text):
    assert _parse_move(text) is None


def test_search_ai_finds_mate_and_can_be_cancelled():
    # 红车平到 (1,4) 后将死；另一车守住第 3 列，红帅封住第 5 列。
    board = position((9, 5, 'r', 'K'), (0, 4, 'b', 'K'),
                     (2, 3, 'r', 'R'), (1, 8, 'r', 'R'), (2, 4, 'r', 'P'))
    before = board.signature()
    move = SearchAI(max_depth=2, time_limit=5).choose_move(board, 'r')
    assert move in board.all_legal_moves('r')
    assert board.apply_move(*move).is_checkmate('b')
    assert board.signature() == before
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        SearchAI().choose_move(Board.fresh(), 'r', cancel_event=cancelled)


def test_facing_kings_cannot_be_exposed():
    board = position((9, 4, 'r', 'K'), (0, 4, 'b', 'K'), (5, 4, 'r', 'R'))
    assert (5, 3) not in board.legal_moves(5, 4)
    assert (4, 4) in board.legal_moves(5, 4)


def test_cannon_requires_exactly_one_screen():
    board = position((9, 4, 'r', 'K'), (0, 3, 'b', 'K'),
                     (7, 0, 'r', 'C'), (3, 0, 'b', 'R'))
    assert (3, 0) not in board.legal_moves(7, 0)
    board.grid[5][0] = Piece('r', 'P')
    assert (3, 0) in board.legal_moves(7, 0)
    board.grid[4][0] = Piece('b', 'P')
    assert (3, 0) not in board.legal_moves(7, 0)

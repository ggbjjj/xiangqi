"""Project rule: a consecutive repeated cycle forces the next player to vary."""
from collections import defaultdict

from board import Board

REPETITION_POLICY = 'consecutive-cycle-force-change-v1'


class RepetitionTracker:
    """Exact board + side history, independent of the network's K/V window.

    With limit=3, A -> ... -> A -> ... -> A triggers only when both intervening
    state/action blocks are identical. Scattered returns to A do not accumulate.
    The initial A counts as the first occurrence. Only the next historical move
    from the shortest matching cycle is forbidden, for the current player.
    """
    def __init__(self, board, turn, limit=3):
        if type(limit) is not int or limit < 2:
            raise ValueError('连续重复次数须为至少2的整数')
        if turn not in ('r', 'b'):
            raise ValueError('未知行棋方')
        self.limit = limit
        self.keys = [(board.signature(), turn)]
        self.moves = []
        self.occurrences = defaultdict(list)
        self.occurrences[self.keys[0]].append(0)
        self._forbidden = None

    @classmethod
    def from_history(cls, board, turn, history=None, limit=3):
        history = history or []
        if not history:
            return cls(board, turn, limit)
        result = cls(Board(history[0]['board_state']), history[0]['prev_turn'], limit)
        for index, step in enumerate(history):
            after = Board(history[index + 1]['board_state']) if index + 1 < len(history) else board
            result.push(tuple(step[key] for key in ('fr', 'fc', 'tr', 'tc')), after)
        result.check_position(board, turn)
        return result

    def check_position(self, board, turn):
        if self.keys[-1] != (board.signature(), turn):
            raise ValueError('重复判定历史与当前局面不一致')

    def push(self, move, after):
        turn = 'b' if self.keys[-1][1] == 'r' else 'r'
        self.moves.append(tuple(move))
        key = (after.signature(), turn)
        self.keys.append(key)
        self.occurrences[key].append(len(self.moves))
        self._forbidden = None

    def pop(self):
        if not self.moves:
            raise ValueError('没有可撤销的局面')
        key = self.keys.pop()
        self.occurrences[key].pop()
        if not self.occurrences[key]:
            del self.occurrences[key]
        self.moves.pop()
        self._forbidden = None

    def forbidden_moves(self):
        if self._forbidden is not None:
            return self._forbidden
        self._forbidden = frozenset()
        end = len(self.moves)
        matches = self.occurrences[self.keys[-1]]
        if len(matches) < self.limit:
            return self._forbidden
        # Try actual return points, shortest cycle first. A longer cycle can
        # itself contain an incidental return to the same board.
        for previous in reversed(matches[:-1]):
            period = end - previous
            start = end - period * (self.limit - 1)
            if start < 0:
                break
            states = self.keys[previous:end]
            actions = self.moves[previous:end]
            if all(self.keys[index:index + period] == states and
                   self.moves[index:index + period] == actions
                   for index in range(start, previous, period)):
                self._forbidden = frozenset({self.moves[previous]})
                break
        return self._forbidden

    def legal_moves(self, board, turn):
        self.check_position(board, turn)
        forbidden = self.forbidden_moves()
        return [move for move in board.all_legal_moves(turn) if move not in forbidden]

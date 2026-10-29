"""Unvalidated research draft. Not imported by the application or training."""
from collections import defaultdict
from dataclasses import dataclass

from board import Board

RULE_SET = 'cn2020-force-change-v1'


@dataclass(frozen=True)
class RepetitionDecision:
    responsible: frozenset = frozenset()
    reason: str = ''
    repeated: bool = False


class RepetitionHistory:
    """Keep rule history separately from neural K/V memory and its sliding window."""
    def __init__(self, board, turn, limit=3):
        if type(limit) is not int or limit<2: raise ValueError('重复判定阈值须至少为2')
        self.limit=limit
        self.states=[(board.clone(),turn)]
        self.moves=[]
        self.keys=[(board.signature(),turn)]
        self.occurrences=defaultdict(list)
        self.occurrences[self.keys[0]].append(0)
        self._decision=None

    @classmethod
    def from_history(cls, board, turn, history=None, limit=3):
        history=history or []
        if not history: return cls(board,turn,limit)
        result=cls(Board(history[0]['board_state']),history[0]['prev_turn'],limit)
        for index,step in enumerate(history):
            after=Board(history[index+1]['board_state']) if index+1<len(history) else board
            result.push(tuple(step[k] for k in ('fr','fc','tr','tc')),after)
        if result.keys[-1]!=(board.signature(),turn): raise ValueError('规则历史与当前局面不一致')
        return result

    def push(self, move, after=None):
        board,turn=self.states[-1]
        after=after if after is not None else board.apply_move(*move)
        other='b' if turn=='r' else 'r'
        self.moves.append(tuple(move))
        self.states.append((after.clone(),other))
        key=(after.signature(),other)
        self.keys.append(key)
        self.occurrences[key].append(len(self.moves))
        self._decision=None

    def pop(self):
        if not self.moves: raise ValueError('没有可撤销的规则历史')
        key=self.keys.pop()
        self.occurrences[key].pop()
        if not self.occurrences[key]: del self.occurrences[key]
        self.moves.pop(); self.states.pop()
        self._decision=None

    def decision(self):
        if self._decision is not None: return self._decision
        repeated=self.occurrences[self.keys[-1]]
        if len(repeated)<self.limit: return RepetitionDecision()
        from .cn2020 import analyse_cycle
        start=repeated[-self.limit]
        self._decision=analyse_cycle(self.states,self.moves,start,len(self.moves))
        return self._decision

    def forbidden_moves(self):
        turn=self.states[-1][1]
        if turn not in self.decision().responsible: return frozenset()
        indices=self.occurrences[self.keys[-1]][-self.limit:-1]
        return frozenset(self.moves[index] for index in indices)

    def legal_moves(self):
        board,turn=self.states[-1]
        forbidden=self.forbidden_moves()
        return [move for move in board.all_legal_moves(turn) if move not in forbidden]

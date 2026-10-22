"""对局状态机：维护棋盘、回合、走子历史、终局判定。"""

from board import Board


class Game:
    """一局象棋对局。

    - 棋盘: self.board
    - 当前回合方: self.turn  ('r' 或 'b')
    - 走子历史: self.history
    - 终局状态: self.status  (空字符串表示未结束)
    """

    MODE_PVP = 'pvp'              # 玩家 vs 玩家
    MODE_AI_RED = 'ai_red'        # AI 执黑，玩家执红
    MODE_AI_BLACK = 'ai_black'    # AI 执红，玩家执黑

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = Board.fresh()
        self.turn = 'r'                # 红方先行
        self.history = []              # 每步含走法与走前局面（支持悔棋）
        self.status = ''               # 空串 = 未结束
        self.mode = self.MODE_PVP
        self.ai_color = None
        self.quiet_moves = 0

    @staticmethod
    def opponent(color):
        return 'b' if color == 'r' else 'r'

    def legal_moves_at(self, r, c):
        """返回 (r,c) 位置棋子的合法走法 [(tr,tc), ...]，若该位置无子或不是当前方则返回 []。"""
        if self.is_over() or not self.board.in_bounds(r, c):
            return []
        p = self.board.grid[r][c]
        if p is None or p.color != self.turn:
            return []
        return self.board.legal_moves(r, c)

    def all_legal_moves(self):
        """返回当前方所有合法走法 [(fr,fc,tr,tc), ...]。"""
        return [] if self.is_over() else self.board.all_legal_moves(self.turn)

    def move(self, fr, fc, tr, tc):
        """尝试走子。

        返回:
          - 'ok'      成功
          - 'illegal' 走法不合法（无子、非本方、违反规则）
          - 'over'    对局已结束

        成功后自动切换回合并检查终局。
        """
        if self.status:
            return 'over'

        if not (self.board.in_bounds(fr, fc) and self.board.in_bounds(tr, tc)):
            return 'illegal'

        p = self.board.grid[fr][fc]
        if p is None or p.color != self.turn:
            return 'illegal'

        if (tr, tc) not in self.board.legal_moves(fr, fc):
            return 'illegal'

        captured = self.board.grid[tr][tc]
        self.history.append({
            'fr': fr, 'fc': fc, 'tr': tr, 'tc': tc,
            'turn': self.turn,
            'piece': repr(p),
            'captured': repr(captured) if captured else None,
            'board_state': [row[:] for row in self.board.grid],  # 走前局面
            'prev_turn': self.turn,
            'prev_status': self.status,
            'prev_quiet_moves': self.quiet_moves,
            'position_key': (self.board.signature(), self.turn),
        })
        self.quiet_moves = 0 if captured else self.quiet_moves + 1
        self.board = self.board.apply_move(fr, fc, tr, tc)
        self.turn = self.opponent(self.turn)
        self._evaluate_end()
        return 'ok'

    def undo(self):
        """悔棋一步。成功返回 True。"""
        if not self.history:
            return False
        last = self.history.pop()
        self.board = Board(last['board_state'])
        self.turn = last['prev_turn']
        self.status = last['prev_status']
        self.quiet_moves = last['prev_quiet_moves']
        return True

    def undo_turn(self, player_color=None):
        """双人撤一步；人机撤到玩家最后一次落子之前。"""
        if player_color is None:
            return int(self.undo())
        index = next((i for i in range(len(self.history) - 1, -1, -1)
                      if self.history[i]['turn'] == player_color), None)
        if index is None:
            return 0
        count = len(self.history) - index
        for _ in range(count):
            self.undo()
        return count

    def _evaluate_end(self):
        """每走一步后判断是否终局。"""
        if not self.board.has_any_legal_move(self.turn):
            winner = self.opponent(self.turn)
            tag = '将死' if self.board.is_in_check(self.turn) else '困毙'
        else:
            key = (self.board.signature(), self.turn)
            if sum(h['position_key'] == key for h in self.history) >= 2:
                self.status = '和棋（三次重复局面）'
            elif self.quiet_moves >= 120:
                self.status = '和棋（连续 120 步未吃子）'
            return
        self.status = f"{'红' if winner == 'r' else '黑'}方胜利！（{tag}）"

    def is_over(self):
        return bool(self.status)

    def winner(self):
        """返回胜方 'r'/'b'，未结束或平局返回 None。"""
        if not self.status:
            return None
        if '红方胜利' in self.status:
            return 'r'
        if '黑方胜利' in self.status:
            return 'b'
        return None

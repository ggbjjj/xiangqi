"""棋盘与规则引擎。

负责：
- 初始化/克隆棋盘
- 生成每个棋子的伪合法走法（不考虑送将）
- 生成合法走法（过滤送将、过滤白脸将）
- 判断将军、将死、困毙
- 走子并返回新 Board（不变更当前对象）
"""

from constants import (
    COLS, ROWS, PIECE_KINDS, INITIAL_BOARD, HORSE_LEGS,
    PALACE_COLS, RED_PALACE_ROWS, BLACK_PALACE_ROWS,
)
from piece import Piece


class Board:
    """10 行 × 9 列的中国象棋棋盘。"""

    def __init__(self, grid=None):
        if grid is None:
            grid = self._build_initial_grid()
        self.grid = grid  # list[list[Piece | None]]

    # ---------- 工厂 ----------

    @staticmethod
    def _build_initial_grid():
        g = []
        for row in INITIAL_BOARD:
            g.append([Piece(s[0], s[1]) if s != '.' else None for s in row])
        return g

    @classmethod
    def fresh(cls):
        return cls()

    def clone(self):
        return Board([row[:] for row in self.grid])

    # ---------- 基础查询 ----------

    def piece_at(self, r, c):
        if 0 <= r < ROWS and 0 <= c < COLS:
            return self.grid[r][c]
        return None

    def in_bounds(self, r, c):
        return (isinstance(r, int) and isinstance(c, int)
                and 0 <= r < ROWS and 0 <= c < COLS)

    def in_palace(self, r, c, color):
        if c not in PALACE_COLS:
            return False
        if color == 'r':
            return r in RED_PALACE_ROWS
        return r in BLACK_PALACE_ROWS

    def crossed_river(self, r, color):
        """已过河：红方 r <= 4，黑方 r >= 5。"""
        if color == 'r':
            return r <= 4
        return r >= 5

    def find_king(self, color):
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p and p.color == color and p.kind == 'K':
                    return r, c
        return None

    # ---------- 走法生成 ----------

    def pseudo_legal_moves(self, r, c):
        """生成 (r,c) 位置棋子的所有伪合法走法（不考虑送将）。"""
        if not self.in_bounds(r, c):
            return []
        p = self.grid[r][c]
        if p is None:
            return []
        color, kind = p.color, p.kind
        moves = []

        if kind == 'K':
            # 将/帅：九宫内走一格
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.in_palace(nr, nc, color):
                    t = self.grid[nr][nc]
                    if t is None or t.color != color:
                        moves.append((nr, nc))

        elif kind == 'A':
            # 士/仕：九宫内走斜线一格
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.in_palace(nr, nc, color):
                    t = self.grid[nr][nc]
                    if t is None or t.color != color:
                        moves.append((nr, nc))

        elif kind == 'E':
            # 象/相：田字，不过河，象眼无子
            for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if not self.in_bounds(nr, nc):
                    continue
                if color == 'r' and nr < 5:
                    continue
                if color == 'b' and nr > 4:
                    continue
                if self.grid[mr][mc] is not None:
                    continue
                t = self.grid[nr][nc]
                if t is None or t.color != color:
                    moves.append((nr, nc))

        elif kind == 'H':
            # 马/傌：走日，蹩马腿
            for dr, dc, br, bc in HORSE_LEGS:
                nr, nc = r + dr, c + dc
                mr, mc = r + br, c + bc
                if self.in_bounds(nr, nc) and self.grid[mr][mc] is None:
                    t = self.grid[nr][nc]
                    if t is None or t.color != color:
                        moves.append((nr, nc))

        elif kind == 'R':
            # 车/俥：直线滑动
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                while self.in_bounds(nr, nc):
                    t = self.grid[nr][nc]
                    if t is None:
                        moves.append((nr, nc))
                    else:
                        if t.color != color:
                            moves.append((nr, nc))
                        break
                    nr += dr
                    nc += dc

        elif kind == 'C':
            # 炮/楚：不吃子时滑动，吃子需隔一个炮架
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                jumped = False
                while self.in_bounds(nr, nc):
                    t = self.grid[nr][nc]
                    if not jumped:
                        if t is None:
                            moves.append((nr, nc))
                        else:
                            jumped = True
                    else:
                        if t is not None:
                            if t.color != color:
                                moves.append((nr, nc))
                            break
                    nr += dr
                    nc += dc

        elif kind == 'P':
            # 兵/卒：前进，过河后可横走
            fwd = -1 if color == 'r' else 1
            nr = r + fwd
            if self.in_bounds(nr, c):
                t = self.grid[nr][c]
                if t is None or t.color != color:
                    moves.append((nr, c))
            if self.crossed_river(r, color):
                for dc in (-1, 1):
                    nc = c + dc
                    if self.in_bounds(r, nc):
                        t = self.grid[r][nc]
                        if t is None or t.color != color:
                            moves.append((r, nc))

        return moves

    def legal_moves(self, r, c):
        """(r,c) 位置棋子的合法走法：伪合法 + 不送将 + 不吃对方将帅。"""
        if not self.in_bounds(r, c):
            return []
        p = self.grid[r][c]
        if p is None:
            return []
        color = p.color
        result = []
        for tr, tc in self.pseudo_legal_moves(r, c):
            # 规则：不能直接吃对方的将/帅（违规走法）
            t = self.grid[tr][tc]
            if t and t.kind == 'K':
                continue
            nb = self.apply_move(r, c, tr, tc)
            # 走完后己方将不能被将军（包括白脸将）
            if not nb.is_in_check(color):
                result.append((tr, tc))
        return result

    def all_legal_moves(self, color):
        """生成指定颜色所有合法走法：[(fr, fc, tr, tc), ...]。"""
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p and p.color == color:
                    for tr, tc in self.legal_moves(r, c):
                        moves.append((r, c, tr, tc))
        return moves

    def has_any_legal_move(self, color):
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p and p.color == color and self.legal_moves(r, c):
                    return True
        return False

    # ---------- 胜负判定 ----------

    def can_opponent_capture_king(self, color, by_color=None):
        """对手能否按走棋规则一步吃掉 color 方将/帅。"""
        if by_color is None:
            by_color = 'b' if color == 'r' else 'r'
        kp = self.find_king(color)
        if kp is None:
            return False
        kr, kc = kp
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p is None or p.color != by_color:
                    continue
                if (kr, kc) in self.pseudo_legal_moves(r, c):
                    return True
        return False

    def is_in_check(self, color):
        """color 方是否被将军（含白脸将）。"""
        if self.can_opponent_capture_king(color):
            return True
        # 白脸将：双方将帅同列且中间无子
        my_king = self.find_king(color)
        opp_color = 'b' if color == 'r' else 'r'
        opp_king = self.find_king(opp_color)
        if my_king and opp_king and my_king[1] == opp_king[1]:
            mr, mc = my_king
            or_, oc = opp_king
            lo, hi = min(mr, or_) + 1, max(mr, or_)
            if all(self.grid[r][mc] is None for r in range(lo, hi)):
                return True
        return False

    def is_checkmate(self, color):
        return self.is_in_check(color) and not self.has_any_legal_move(color)

    def is_stalemate(self, color):
        """困毙：无路可走且未被将军（象棋中判负）。"""
        return not self.is_in_check(color) and not self.has_any_legal_move(color)

    # ---------- 走子 ----------

    def apply_move(self, fr, fc, tr, tc):
        """走子并返回新 Board（不变更当前）。"""
        new_grid = [row[:] for row in self.grid]
        moved = new_grid[fr][fc]
        new_grid[tr][tc] = moved
        new_grid[fr][fc] = None
        return Board(new_grid)

    # ---------- 缓存键 ----------

    def signature(self):
        return tuple(tuple((p.color, p.kind) if p else None for p in row)
                     for row in self.grid)

    # ---------- 显示 ----------

    def __str__(self):
        from constants import NAMES_RED, NAMES_BLACK
        lines = []
        for r, row in enumerate(self.grid):
            cells = []
            for p in row:
                if p is None:
                    cells.append('··')
                else:
                    n = NAMES_RED[p.kind] if p.color == 'r' else NAMES_BLACK[p.kind]
                    cells.append(n)
            lines.append(f"{r} " + "  ".join(cells))
        return "\n".join(lines)

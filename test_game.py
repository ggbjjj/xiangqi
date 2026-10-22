"""象棋逻辑测试：验证走子规则、将军、将死、困毙、白脸将。

用法：
    python test_game.py
"""

import sys
import traceback

from board import Board
from encoder import (
    encode_state, move_to_action, action_to_move,
    legal_action_mask, NUM_ACTIONS, IN_CHANNELS,
)
from game import Game


PASS = '[OK]'
FAIL = '[FAIL]'

_stats = {'pass': 0, 'fail': 0}


def expect(cond, msg):
    if cond:
        print(f"  {PASS} {msg}")
        _stats['pass'] += 1
        return True
    else:
        print(f"  {FAIL} {msg}")
        _stats['fail'] += 1
        raise AssertionError(msg)


# ============================================================
# 1) 初始局面
# ============================================================
def test_initial_board():
    print("\n[1] 初始局面")
    b = Board.fresh()
    # 32 个棋子
    n = sum(1 for r in range(10) for c in range(9) if b.grid[r][c] is not None)
    expect(n == 32, f"棋子数 = 32 (实际 {n})")
    # 红帅在 (9,4)，黑将在 (0,4)
    expect(b.find_king('r') == (9, 4), "红帅在 (9,4)")
    expect(b.find_king('b') == (0, 4), "黑将在 (0,4)")
    # 编码形状
    planes = encode_state(b, 'r')
    expect(planes.shape == (IN_CHANNELS, 10, 9), f"编码形状 = ({IN_CHANNELS},10,9)")
    expect(planes[14, 0, 0] == 1.0, "turn 通道: 红方走时 = 1")
    planes2 = encode_state(b, 'b')
    expect(planes2[14, 0, 0] == 0.0, "turn 通道: 黑方走时 = 0")


# ============================================================
# 2) 各棋子走法
# ============================================================
def test_piece_moves():
    print("\n[2] 各棋子走法")

    # 车: 清空一列，孤立放置红车
    b = Board.fresh()
    from piece import Piece
    for r in range(10):
        b.grid[r][0] = None  # 清空第 0 列
    b.grid[5][0] = Piece('r', 'R')  # 红车孤立
    moves = set(b.legal_moves(5, 0))
    expect((0, 0) in moves, "红车 (5,0) 可走到 (0,0)")
    expect((9, 0) in moves, "红车 (5,0) 可走到 (9,0)")

    # 马: (9,1) 初始红马
    b = Board.fresh()
    moves = set(b.legal_moves(9, 1))
    # (9,1) -> (7,0) 或 (7,2)
    expect((7, 0) in moves, "红马 (9,1) -> (7,0)")
    expect((7, 2) in moves, "红马 (9,1) -> (7,2)")

    # 象: (9,2) 初始红象，不能过河
    b = Board.fresh()
    moves = set(b.legal_moves(9, 2))
    expect((7, 0) in moves, "红相 (9,2) -> (7,0)")
    expect((7, 4) in moves, "红相 (9,2) -> (7,4)")
    # 不能过河
    bad = [(r, c) for (r, c) in moves if r < 5]
    expect(len(bad) == 0, f"红相不能过河 (违规: {bad})")

    # 士: (9,3) 初始红仕
    b = Board.fresh()
    moves = set(b.legal_moves(9, 3))
    expect(moves == {(8, 4)}, f"红仕 (9,3) 只能走 (8,4) (实际 {moves})")

    # 帅: (9,4) 初始红帅
    b = Board.fresh()
    moves = set(b.legal_moves(9, 4))
    expect(moves == {(8, 4)}, f"红帅 (9,4) 只能走 (8,4) (实际 {moves})")

    # 兵: (6,0) 红兵（未过河），只能前进
    b = Board.fresh()
    moves = set(b.legal_moves(6, 0))
    expect(moves == {(5, 0)}, f"未过河红兵 (6,0) 只能前进 (实际 {moves})")

    # 炮: (7,1) 红炮
    b = Board.fresh()
    moves = set(b.legal_moves(7, 1))
    # 可以沿列向上走到 (6,1)(5,1)(4,1)(3,1) 不吃子；吃子需隔一个
    for tr, tc in [(6, 1), (5, 1), (4, 1), (3, 1)]:
        expect((tr, tc) in moves, f"红炮 (7,1) -> ({tr},{tc})")

    # 蹩马腿: 堵住 (8,1) 后 (9,1) 马不能走
    b = Board.fresh()
    b.grid[8][1] = b.grid[9][1]  # 在 (8,1) 放一子（同色），实际就是原马不动不可
    # 改：把红马放到 (7,0)，让 (8,1) 有别子
    b = Board.fresh()
    # 构造 (7,0) 是红马，且 (8,1) 是别子 (放个兵)
    from piece import Piece
    b.grid[7][0] = Piece('r', 'H')
    b.grid[6][0] = Piece('r', 'P')  # (7,0) -> (5,1) 的马腿在 (6,0)
    moves = set(b.legal_moves(7, 0))
    # 想去 (5,1)，但 (6,0) 堵腿。
    expect((5, 1) not in moves, "蹩马腿: 红马 (7,0) -> (5,1) 被堵")
    b.grid[6][0] = None
    expect((5, 1) in b.legal_moves(7, 0), "移开马腿后可以走")

    # 象眼: 堵住 (8,1) 后 (9,2) 相不能走 (7,0)
    b = Board.fresh()
    b.grid[8][1] = Piece('r', 'P')
    moves = set(b.legal_moves(9, 2))
    expect((7, 0) not in moves, "塞象眼: 红相 (9,2) -> (7,0) 被堵")


# ============================================================
# 3) 非法走法过滤
# ============================================================
def test_illegal_moves():
    print("\n[3] 非法走法")
    b = Board.fresh()

    # 走空位（用 (1,0)，确实是空的）
    expect(not b.legal_moves(1, 0), "空位 (1,0) 无走法")

    # Board.legal_moves 不区分配色，这是 Game 层的事
    # 用 Game 的 legal_moves_at 检查回合方
    g = Game()
    expect(g.legal_moves_at(0, 0) == [], "Game.legal_moves_at 黑子位置（红方回合） -> []")

    # 不能直接吃对方的将帅（红帅去吃黑将？通过构造场景）
    b = Board.fresh()
    from piece import Piece
    # 把红帅移到 (5,4)，黑将移到 (4,4)，中间无子 → 这其实是白脸将，应该判为非法
    b.grid[5][4] = Piece('r', 'K')
    b.grid[6][4] = None
    b.grid[7][4] = None
    b.grid[8][4] = None
    b.grid[4][4] = Piece('b', 'K')
    b.grid[3][4] = None
    b.grid[2][4] = None
    b.grid[1][4] = None
    b.grid[0][4] = None
    # 红帅 (5,4) 走到 (4,4) 是吃黑将，规则禁止
    moves = set(b.legal_moves(5, 4))
    expect((4, 4) not in moves, "禁止直接吃对方将帅")


# ============================================================
# 4) 白脸将（facing kings）
# ============================================================
def test_facing_kings():
    print("\n[4] 白脸将")
    b = Board.fresh()
    from piece import Piece
    # 清空中间，红帅 (5,4) 黑将 (4,4) 中间无子 → 红方被将军
    for r in range(10):
        for c in range(9):
            b.grid[r][c] = None
    b.grid[5][4] = Piece('r', 'K')
    b.grid[4][4] = Piece('b', 'K')
    expect(b.is_in_check('r'), "白脸将: 红方被将军")
    expect(b.is_in_check('b'), "白脸将: 黑方也被将军")

    # 中间放个棋子，白脸将解除（让两个将之间有合理距离）
    b = Board.fresh()
    for r in range(10):
        for c in range(9):
            b.grid[r][c] = None
    b.grid[6][4] = Piece('r', 'K')
    b.grid[2][4] = Piece('b', 'K')
    # 不放中间子：白脸将
    expect(b.is_in_check('r'), "无中间子: 白脸将")
    # 在 (4,4) 放红兵：阻断白脸将
    b.grid[4][4] = Piece('r', 'P')
    expect(not b.is_in_check('r'), "中间有红兵: 红方未被将军")
    expect(not b.is_in_check('b'), "中间有红兵: 黑方也未被将军")


# ============================================================
# 5) 将死
# ============================================================
def test_checkmate():
    print("\n[5] 将死")
    b = Board.fresh()
    from piece import Piece
    # 构造一个将死场景：
    # 红帅 (9,4)，黑车在 (9,3) 和 (9,5)，红帅被将死
    b.grid[9][3] = Piece('b', 'R')
    b.grid[9][5] = Piece('b', 'R')
    # 黑将在 (0,4)，其他清空避免干扰
    b.grid[0][4] = Piece('b', 'K')
    for r in range(10):
        for c in range(9):
            if (r, c) not in [(9, 4), (9, 3), (9, 5), (0, 4)]:
                b.grid[r][c] = None
    # 红帅被两个车夹住，无路可走
    expect(b.is_in_check('r'), "红帅被将（双车夹）")
    expect(b.is_checkmate('r'), "红帅被将死")
    # 走 game 应判红方输
    g = Game()
    g.board = b
    g.turn = 'r'
    g._evaluate_end()
    expect(g.is_over(), "Game 检测到将死")
    expect(g.winner() == 'b', "胜方 = 黑方")


# ============================================================
# 6) 困毙（无路可走但未被将军）
# ============================================================
def test_stalemate():
    print("\n[6] 困毙")
    b = Board.fresh()
    from piece import Piece
    # 清空棋盘，构造经典困毙：
    # 红帅 (9,4) 三面被对方车远程控制（不被将军），无法移动
    for r in range(10):
        for c in range(9):
            b.grid[r][c] = None
    b.grid[9][4] = Piece('r', 'K')
    # 三个黑车分别控制 (9,5)/(9,3)/(8,4) 但都不攻击 (9,4)
    b.grid[0][5] = Piece('b', 'R')   # 控制第 5 列 -> 攻击 (9,5)
    b.grid[5][3] = Piece('b', 'R')   # 控制第 3 列 -> 攻击 (9,3)
    b.grid[8][0] = Piece('b', 'R')   # 控制第 8 行 -> 攻击 (8,4)
    expect(not b.is_in_check('r'), "困毙: 红帅未被将军")
    expect(not b.has_any_legal_move('r'), "困毙: 红帅无路可走")
    expect(b.is_stalemate('r'), "困毙成立")
    g = Game()
    g.board = b
    g.turn = 'r'
    g._evaluate_end()
    expect(g.is_over(), "Game 检测到困毙")
    expect(g.winner() == 'b', "困毙 = 输")


# ============================================================
# 7) Game.move 三种返回值
# ============================================================
def test_game_move():
    print("\n[7] Game.move 接口")
    g = Game()
    # 炮二平五：红炮从 (7,1) -> (7,4) (中间有子需隔一个炮架)
    # 红炮 (7,1)，向上需要隔一子才能吃到第3格
    # 改为兵五进一：红兵 (6,2) -> (5,2)
    expect(g.move(6, 2, 5, 2) == 'ok', "合法走法 (红兵进一) -> 'ok'")
    expect(g.turn == 'b', "走完后轮到黑方")
    expect(g.move(0, 0, 0, 5) == 'illegal', "走黑子（红方回合） -> 'illegal'")
    expect(g.move(5, 2, 0, 2) == 'illegal', "红兵 (5,2) 不能跳到 (0,2) -> 'illegal'")


# ============================================================
# 8) 悔棋
# ============================================================
def test_undo():
    print("\n[8] 悔棋")
    g = Game()
    g.move(6, 2, 5, 2)  # 红兵进一
    g.move(3, 2, 4, 2)  # 黑兵进一
    expect(len(g.history) == 2, "history 应有 2 步")
    g.undo()
    g.undo()
    expect(len(g.history) == 0, "悔棋 2 步后 history 清空")
    expect(g.turn == 'r', "回到红方回合")
    expect(g.board.grid[6][2] is not None and g.board.grid[6][2].kind == 'P',
           "红兵回到 (6,2)")


# ============================================================
# 9) 完整自对弈（双方轮流走直到终局或满手数）
# ============================================================
def test_random_selfplay():
    print("\n[9] 完整对局（随机走子）")
    import random
    random.seed(42)
    g = Game()
    moves_count = 0
    while not g.is_over() and moves_count < 200:
        legal = g.all_legal_moves()
        if not legal:
            break
        move = random.choice(legal)
        fr, fc, tr, tc = move
        result = g.move(fr, fc, tr, tc)
        if result != 'ok':
            print(f"    {FAIL} 非法走法: {move}, status={g.status}")
            break
        moves_count += 1
    expect(moves_count > 0, f"走了 {moves_count} 步")
    if g.is_over():
        print(f"    终局: {g.status}")
        print(f"    {PASS} 完整对局正常结束")
    else:
        print(f"    [info] 走满 {moves_count} 步未终局（合理：随机走子可能僵持）")


# ============================================================
# 10) 编码器 + 动作编码互逆
# ============================================================
def test_encoder():
    print("\n[10] 编码器")
    b = Board.fresh()
    planes = encode_state(b, 'r')
    expect(planes.dtype.name == 'float32', "dtype = float32")
    expect(planes.sum() == 32.0 + 90.0, f"非零元素 = 32 棋子 + 90 turn 格 = 122 (实际 {int(planes.sum())})")

    # 动作编码互逆
    for fr, fc, tr, tc in [(0, 0, 1, 0), (9, 4, 8, 4), (5, 5, 0, 5)]:
        a = move_to_action((fr, fc, tr, tc))
        expect(0 <= a < NUM_ACTIONS, f"action {a} ∈ [0, {NUM_ACTIONS})")
        m = action_to_move(a)
        expect(m == (fr, fc, tr, tc), f"action_to_move 互逆: {m} == {(fr,fc,tr,tc)}")


# ============================================================
# 主入口
# ============================================================
def main():
    tests = [
        test_initial_board,
        test_piece_moves,
        test_illegal_moves,
        test_facing_kings,
        test_checkmate,
        test_stalemate,
        test_game_move,
        test_undo,
        test_random_selfplay,
        test_encoder,
    ]
    err_count = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            err_count += 1
            print(f"\n  {FAIL} {t.__name__}: {e}")
            traceback.print_exc()
    total = _stats['pass'] + _stats['fail']
    print(f"\n========== 测试结果 ==========")
    print(f"断言: {_stats['pass']}/{total} 通过, {_stats['fail']} 失败, {err_count} 异常")
    if _stats['fail'] == 0 and err_count == 0:
        print("全部通过！[OK]")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

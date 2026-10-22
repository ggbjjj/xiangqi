"""状态/动作编码器：把棋盘变成神经网络可用的张量。"""

import numpy as np

from constants import COLS, ROWS, PIECE_KINDS


# 通道定义：
#   0-6  : 红方 7 种棋子 (K, A, E, H, R, C, P)
#   7-13 : 黑方 7 种棋子
#   14   : 当前回合方 (红=1, 黑=0)
IN_CHANNELS = 15

NUM_SQUARES = ROWS * COLS          # 90
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 8100


def encode_state(board, turn):
    """返回 (15, 10, 9) 的 float32 张量。"""
    planes = np.zeros((IN_CHANNELS, ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            p = board.grid[r][c]
            if p is None:
                continue
            kind_idx = PIECE_KINDS.index(p.kind)
            base = 0 if p.color == 'r' else 7
            planes[base + kind_idx, r, c] = 1.0
    planes[14, :, :] = 1.0 if turn == 'r' else 0.0
    return planes


def move_to_action(move):
    """(fr, fc, tr, tc) → [0, 8100) 的动作 id。"""
    fr, fc, tr, tc = move
    return (fr * COLS + fc) * NUM_SQUARES + (tr * COLS + tc)


def action_to_move(action):
    """[0, 8100) 的动作 id → (fr, fc, tr, tc)。"""
    to_sq = action % NUM_SQUARES
    from_sq = action // NUM_SQUARES
    return from_sq // COLS, from_sq % COLS, to_sq // COLS, to_sq % COLS


def legal_action_mask(board, turn):
    """返回长度 8100 的 bool 数组，合法动作为 True。"""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.all_legal_moves(turn):
        mask[move_to_action(move)] = True
    return mask

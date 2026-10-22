"""AI 封装：网络 + MCTS，对外暴露 choose_move。"""

from concurrent.futures import CancelledError
from time import monotonic

from constants import COLS, PIECE_VALUES

NUM_SQUARES = 90


class SearchAI:
    """无需模型的限时迭代加深搜索，使用子力、兵的位置与将死评分。"""

    def __init__(self, max_depth=2, time_limit=1.5):
        self.max_depth = max_depth
        self.time_limit = time_limit

    def reset(self):
        pass

    @staticmethod
    def _evaluate(board, turn):
        score = 0
        for r, row in enumerate(board.grid):
            for c, piece in enumerate(row):
                if piece is None:
                    continue
                value = PIECE_VALUES[piece.kind]
                if piece.kind == 'P':
                    progress = 9 - r if piece.color == 'r' else r
                    value += progress * 8 + (40 if board.crossed_river(r, piece.color) else 0)
                elif piece.kind in ('H', 'C'):
                    value += (4 - abs(c - 4)) * 3
                score += value if piece.color == turn else -value
        return score

    @staticmethod
    def _ordered(board, moves):
        def score(move):
            fr, fc, tr, tc = move
            target = board.grid[tr][tc]
            return (10 * PIECE_VALUES[target.kind] - PIECE_VALUES[board.grid[fr][fc].kind]
                    if target else 0)
        return sorted(moves, key=score, reverse=True)

    def choose_move(self, board, turn, cancel_event=None):
        deadline = monotonic() + self.time_limit

        def check_budget():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()
            if monotonic() >= deadline:
                raise TimeoutError()

        def search(position, color, depth, alpha, beta, ply):
            check_budget()
            moves = position.all_legal_moves(color)
            if not moves:
                return -100000 + ply
            if depth == 0:
                return self._evaluate(position, color)
            best = -float('inf')
            other = 'b' if color == 'r' else 'r'
            for move in self._ordered(position, moves):
                value = -search(position.apply_move(*move), other, depth - 1, -beta, -alpha, ply + 1)
                best = max(best, value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best

        moves = self._ordered(board, board.all_legal_moves(turn))
        if not moves:
            return None
        chosen = moves[0]
        other = 'b' if turn == 'r' else 'r'
        for depth in range(1, self.max_depth + 1):
            best_score, best_move = -float('inf'), chosen
            try:
                for move in moves:
                    value = -search(board.apply_move(*move), other, depth - 1,
                                    -float('inf'), -best_score, 1)
                    if value > best_score:
                        best_score, best_move = value, move
            except TimeoutError:
                break
            chosen = best_move
            moves.remove(chosen)
            moves.insert(0, chosen)
        return chosen


def load_network(model_path, channels=64, blocks=6):
    """按 checkpoint 元数据加载；基本游戏不导入 PyTorch。"""
    from pathlib import Path
    if not model_path or not Path(model_path).is_file():
        return None
    import torch
    from net import PolicyValueNet
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    net = PolicyValueNet(channels=config.get('channels', channels),
                         num_blocks=config.get('num_blocks', blocks))
    net.load_state_dict(checkpoint.get('model', checkpoint))
    net.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return net


class NeuralAI:
    """AlphaZero 风格 AI：每次决策用 MCTS 跑若干模拟，从访问分布选走法。

    Args:
        net:                PolicyValueNet
        device:             'cpu' / 'cuda'
        num_simulations:    每步 MCTS 模拟数
        temperature:        默认采样温度（0=贪心）
        temperature_moves:  前 N 手用温度，之后贪心
        add_noise:          根节点是否加 Dirichlet 噪声
    """

    def __init__(self, net, device='cpu', num_simulations=200,
                 temperature=0.0, temperature_moves=0, add_noise=False):
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        self.add_noise = add_noise
        self._moves_played = 0

    def choose_move(self, board, turn, return_diag=False, cancel_event=None):
        """返回 (fr, fc, tr, tc) 或 None；若 return_diag=True 同时返回诊断信息。"""
        from mcts import MCTS
        mcts = MCTS(self.net, device=self.device,
                    num_simulations=self.num_simulations,
                    add_noise_at_root=self.add_noise, cancel_event=cancel_event)
        root, _ = mcts.run(board.clone(), turn)

        temp = self.temperature if self._moves_played < self.temperature_moves else 0.0
        action, distribution = mcts.select_move(root, temperature=temp)
        import numpy as np
        self.last_policy = np.zeros(NUM_SQUARES * NUM_SQUARES, dtype=np.float32)
        if distribution is not None:
            self.last_policy[list(root.children)] = distribution
        self._moves_played += 1

        if action is None:
            return (None, None) if return_diag else None
        move = self._action_to_move(action)
        if return_diag:
            diag = mcts.diagnose(root)
            diag.insert(0, {'root_visits': root.N})
            return move, diag
        return move

    @staticmethod
    def _action_to_move(action):
        to_sq = action % NUM_SQUARES
        from_sq = action // NUM_SQUARES
        return from_sq // COLS, from_sq % COLS, to_sq // COLS, to_sq % COLS

    def reset(self):
        self._moves_played = 0

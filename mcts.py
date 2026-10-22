"""蒙特卡洛树搜索（PUCT 算法，参考 AlphaZero）。

关键点：
- 选择用 PUCT: Q + c_puct * P * sqrt(N_parent) / (1 + N)
- 叶子用神经网络评估（策略 logits + 价值）
- 回溯时每层翻转价值符号
- 终局（将死/困毙）严格按规则返回 ±1
- 局面缓存避免重复计算
"""

import math
from concurrent.futures import CancelledError

import numpy as np
import torch

from encoder import (
    NUM_ACTIONS, NUM_SQUARES, COLS,
    encode_state, legal_action_mask,
)


# 超参数
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


class MCTSNode:
    """MCTS 树节点。"""

    __slots__ = ('P', 'N', 'W', 'children')

    def __init__(self, prior=0.0):
        self.P = prior        # 先验概率（来自策略网络）
        self.N = 0            # 访问次数
        self.W = 0.0          # 累计价值
        self.children = {}    # action -> MCTSNode

    def Q(self):
        if self.N == 0:
            return 0.0
        return self.W / self.N

    def ucb_score(self, parent_n, c_puct=C_PUCT):
        # Q 存储当前节点走方的价值；父节点应选择对手价值最低的分支。
        return -self.Q() + c_puct * self.P * math.sqrt(max(parent_n, 1)) / (1 + self.N)

    def expand(self, prior_array):
        """根据先验概率向量扩张所有合法子节点。"""
        nz = np.where(prior_array > 0)[0]
        for a in nz:
            self.children[int(a)] = MCTSNode(prior=float(prior_array[a]))


class MCTS:
    """单进程 MCTS。"""

    def __init__(self, net, device='cpu', num_simulations=200,
                 c_puct=C_PUCT, dirichlet_alpha=DIRICHLET_ALPHA,
                 dirichlet_eps=DIRICHLET_EPS, add_noise_at_root=False,
                 cancel_event=None):
        self.net = net
        self.device = device
        if num_simulations < 1:
            raise ValueError('num_simulations 必须大于 0')
        self.num_simulations = num_simulations
        self.cancel_event = cancel_event
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise_at_root = add_noise_at_root
        # 局面缓存
        self._cache = {}

    # ---------- 评估（带缓存） ----------

    def _evaluate(self, board, turn):
        """返回 (policy_logits numpy, value float)。"""
        key = (board.signature(), turn)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        planes = encode_state(board, turn)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_logits, v = self.net(x)
        p = p_logits.cpu().numpy()[0]
        v = float(v.squeeze().item())
        self._cache[key] = (p, v)
        return p, v

    @staticmethod
    def _legal_prior(policy_logits, board, turn):
        """把 logits 转成只在合法动作上非零的概率分布。"""
        mask = legal_action_mask(board, turn)
        masked = policy_logits.copy()
        masked[~mask] = -1e9
        m = masked.max()
        exp = np.exp(masked - m)
        exp[~mask] = 0.0
        s = exp.sum()
        if s <= 0:
            return mask.astype(np.float32) / max(mask.sum(), 1)
        return (exp / s).astype(np.float32)

    # ---------- 终局判定 ----------

    @staticmethod
    def _terminal_value(board, turn):
        """终局返回 (value, is_terminal)；turn 方无路可走 = -1。"""
        if not board.has_any_legal_move(turn):
            return -1.0, True
        return 0.0, False

    # ---------- 主循环 ----------

    def run(self, root_board, root_turn):
        """运行 num_simulations 次模拟，返回 (root, root_value)。"""
        root = MCTSNode(prior=0.0)
        terminal_value, terminal = self._terminal_value(root_board, root_turn)
        if terminal:
            return root, terminal_value
        policy, value = self._evaluate(root_board, root_turn)
        root.expand(self._legal_prior(policy, root_board, root_turn))

        # 根节点加 Dirichlet 噪声（自对弈时增加探索）
        if self.add_noise_at_root and root.children:
            moves = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            for a, n in zip(moves, noise):
                root.children[a].P = (
                    (1 - self.dirichlet_eps) * root.children[a].P
                    + self.dirichlet_eps * n
                )

        for _ in range(self.num_simulations):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise CancelledError()
            node = root
            path = [node]
            board = root_board
            turn = root_turn

            # Selection
            seen = {(board.signature(), turn)}
            repeated = False
            while node.children:
                a = self._select_child(node)
                if a is None:
                    break
                child = node.children[a]
                fr, fc, tr, tc = self._action_to_move(a)
                board = board.apply_move(fr, fc, tr, tc)
                turn = 'b' if turn == 'r' else 'r'
                node = child
                path.append(node)
                key = (board.signature(), turn)
                if key in seen:
                    repeated = True
                    break
                seen.add(key)

            # 终局直接判负，否则神经网络评估
            term_val, is_term = self._terminal_value(board, turn)
            if repeated:
                leaf_value = 0.0
            elif is_term:
                leaf_value = term_val
            else:
                p, v = self._evaluate(board, turn)
                node.expand(self._legal_prior(p, board, turn))
                leaf_value = v

            # Backprop
            for n in reversed(path):
                n.N += 1
                n.W += leaf_value
                leaf_value = -leaf_value

        return root, value

    def _select_child(self, node):
        best_a, best_s = None, -float('inf')
        for a, child in node.children.items():
            s = child.ucb_score(node.N, c_puct=self.c_puct)
            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    @staticmethod
    def _action_to_move(action):
        to_sq = action % NUM_SQUARES
        from_sq = action // NUM_SQUARES
        return from_sq // COLS, from_sq % COLS, to_sq // COLS, to_sq % COLS

    # ---------- 选走法 ----------

    def select_move(self, root, temperature=0.0):
        """根据访问次数选走法。

        temperature=0 → 贪心；>0 → 按 N^(1/T) 比例采样。
        返回 (action, prior_distribution)。
        """
        moves = list(root.children.keys())
        if not moves:
            return None, None
        visits = np.array([root.children[a].N for a in moves], dtype=np.float64)
        if temperature <= 1e-6 or visits.sum() == 0:
            idx = int(np.argmax(visits))
        else:
            v = visits ** (1.0 / temperature)
            v /= v.sum()
            idx = int(np.random.choice(len(moves), p=v))
        prior = visits / max(visits.sum(), 1)
        return moves[idx], prior

    # ---------- 诊断 ----------

    def diagnose(self, root, top_k=5):
        """返回 root 子节点按访问次数排序的诊断信息。"""
        items = []
        for a, child in root.children.items():
            fr, fc, tr, tc = self._action_to_move(a)
            items.append({
                'action': a,
                'from': (fr, fc),
                'to': (tr, tc),
                'N': child.N,
                'P': child.P,
                'Q': -child.W / max(child.N, 1),
                'pv_frac': child.N / max(root.N, 1),
            })
        items.sort(key=lambda x: x['N'], reverse=True)
        return items[:top_k]

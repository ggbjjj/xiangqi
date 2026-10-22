"""MCTS / 网络 / AI 自对弈的冒烟测试（不训练，只验证能跑）。"""

import sys
import time
import traceback

import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='安装 neural 扩展后验证网络和训练')
torch.set_num_threads(min(2, torch.get_num_threads()))

from ai import NeuralAI
from encoder import encode_state
from game import Game
from mcts import MCTS
from net import PolicyValueNet


PASS, FAIL = '[OK]', '[FAIL]'
_stats = {'pass': 0, 'fail': 0}


def expect(cond, msg):
    if cond:
        print(f"  {PASS} {msg}")
        _stats['pass'] += 1
    else:
        print(f"  {FAIL} {msg}")
        _stats['fail'] += 1
        raise AssertionError(msg)


def test_net_forward():
    print("\n[A] 网络前向")
    net = PolicyValueNet(channels=32, num_blocks=2)  # 极小网络，CPU 跑得快
    net.eval()
    b = Game().board
    x = torch.from_numpy(encode_state(b, 'r')).unsqueeze(0)
    p_logits, v = net(x)
    expect(p_logits.shape == (1, 8100), f"策略输出形状 {tuple(p_logits.shape)}")
    expect(v.shape == (1, 1), f"价值输出形状 {tuple(v.shape)}")
    expect(-1.001 < v.item() < 1.001, f"价值在 [-1,1]: {v.item():.3f}")

    # 梯度能反传
    net.train()
    p_logits, v = net(x)
    loss = p_logits.sum() + v.sum()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    expect(len(grads) > 0, "有梯度流")
    expect(all(g.abs().sum() > 0 for g in grads), "梯度非零")


def test_mcts_basic():
    print("\n[B] MCTS 单步")
    torch.manual_seed(0)
    np.random.seed(0)
    net = PolicyValueNet(channels=32, num_blocks=2)
    net.eval()
    b = Game().board
    mcts = MCTS(net, device='cpu', num_simulations=20, add_noise_at_root=False)
    t0 = time.perf_counter()
    root, value = mcts.run(b.clone(), 'r')
    expect(root.N == 20, f"根访问次数 = 20 (实际 {root.N})")
    expect(len(root.children) > 0, f"根子节点数 = {len(root.children)}")
    action, prior = mcts.select_move(root, temperature=0.0)
    expect(action is not None, "选出了走法")
    print(f"  用时 {time.perf_counter()-t0:.2f}s")


def test_mcts_temperature():
    print("\n[C] MCTS 温度")
    np.random.seed(1)
    net = PolicyValueNet(channels=32, num_blocks=2)
    net.eval()
    b = Game().board
    mcts = MCTS(net, num_simulations=20)
    root, _ = mcts.run(b.clone(), 'r')
    action_greedy, _ = mcts.select_move(root, temperature=0.0)
    expect(action_greedy is not None, "贪心可选出走法")


def test_ai_selfplay():
    print("\n[D] AI 自我对弈 1 局（少模拟数）")
    torch.manual_seed(42)
    np.random.seed(42)
    net = PolicyValueNet(channels=32, num_blocks=2)
    net.eval()
    ai = NeuralAI(net, device='cpu', num_simulations=10,
                  temperature=1.0, temperature_moves=5, add_noise=True)
    game = Game()
    moves = 0
    t0 = time.perf_counter()
    while not game.is_over() and moves < 60:
        move = ai.choose_move(game.board, game.turn)
        if move is None:
            break
        fr, fc, tr, tc = move
        if game.move(fr, fc, tr, tc) != 'ok':
            print(f"  {FAIL} 非法走法: {move}")
            break
        moves += 1
    print(f"  走了 {moves} 步, 终局: {game.status or '未终局'} 用时 {time.perf_counter()-t0:.1f}s")
    expect(moves > 5, f"至少走了 5 步 (实际 {moves})")


def test_ai_vs_ai():
    print("\n[E] AI vs AI 互博 1 局")
    torch.manual_seed(7)
    np.random.seed(7)
    net = PolicyValueNet(channels=32, num_blocks=2)
    net.eval()
    ai_r = NeuralAI(net, num_simulations=10, temperature=0.0, add_noise=False)
    ai_b = NeuralAI(net, num_simulations=10, temperature=0.0, add_noise=False)
    game = Game()
    moves = 0
    while not game.is_over() and moves < 80:
        ai = ai_r if game.turn == 'r' else ai_b
        move = ai.choose_move(game.board, game.turn)
        if move is None:
            break
        fr, fc, tr, tc = move
        if game.move(fr, fc, tr, tc) != 'ok':
            break
        moves += 1
    expect(moves > 0, f"AI 互博 {moves} 步")
    print(f"  终局: {game.status or '未终局'}")


def test_train_step():
    print("\n[F] 训练一步（小批量）")
    import torch.nn.functional as F
    from train import build_samples, train_step
    torch.manual_seed(0)
    np.random.seed(0)

    # 构造几局样本
    net = PolicyValueNet(channels=32, num_blocks=2)
    net.eval()
    ai = NeuralAI(net, num_simulations=5, temperature=1.0, temperature_moves=5,
                  add_noise=True)

    all_states, all_policies, all_values = [], [], []
    for g in range(2):
        game = Game()
        states, policies = [], []
        moves = 0
        while not game.is_over() and moves < 30:
            move = ai.choose_move(game.board, game.turn)
            if move is None:
                break
            states.append(encode_state(game.board, game.turn))
            # 简化：用均匀分布占位
            from encoder import legal_action_mask, NUM_ACTIONS
            mask = legal_action_mask(game.board, game.turn)
            policy = mask.astype(np.float32) / max(mask.sum(), 1)
            policies.append(policy)
            fr, fc, tr, tc = move
            if game.move(fr, fc, tr, tc) != 'ok':
                break
            moves += 1
        winner = game.winner()
        if not states:
            continue
        s, p, v = build_samples(np.stack(states), np.stack(policies), winner)
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)

    if not all_states:
        print(f"  {FAIL} 没采到样本")
        _stats['fail'] += 1
        return

    states = np.stack(all_states)
    policies = np.stack(all_policies)
    values = np.stack(all_values)
    print(f"  样本数: {len(states)}")

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss, acc = train_step(net, optim, (states, policies, values),
                           batch_size=32)
    expect(loss > 0, f"loss = {loss:.4f}")
    expect(0 <= acc <= 1, f"policy_acc = {acc:.4f}")


def main():
    tests = [test_net_forward, test_mcts_basic, test_mcts_temperature,
             test_ai_selfplay, test_ai_vs_ai, test_train_step]
    for t in tests:
        try:
            t()
        except Exception as e:
            _stats['fail'] += 1
            print(f"\n  {FAIL} {t.__name__}: {e}")
            traceback.print_exc()
    total = _stats['pass'] + _stats['fail']
    print(f"\n========== 冒烟测试结果 ==========")
    print(f"通过 {_stats['pass']}/{total}, 失败 {_stats['fail']}")
    if _stats['fail'] == 0:
        print("全部通过！[OK]")
        sys.exit(0)
    sys.exit(1)


if __name__ == '__main__':
    main()

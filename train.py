"""多智能体并行 AlphaZero 训练。

工作流程：
1. 主进程持有网络 + replay buffer（在 GPU/CPU 上）
2. 多个 worker 进程并行自对弈（在 CPU 上）
3. 主进程收集样本后做多个 epoch 的训练
4. 周期性保存模型与样本

用法：
    python train.py --iterations 5 --sims 600
    python train.py --iterations 5 --sims 600 --cpu
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ai import NeuralAI
from encoder import NUM_ACTIONS, encode_state
from game import Game
from net import PolicyValueNet


# ============================================================
# Worker：单进程跑一局自对弈
# ============================================================

def _play_one_self_game(net_state_dict, num_simulations, max_moves,
                        temperature_moves, seed, use_noise,
                        channels=64, num_blocks=6):
    """子进程中跑一局自对弈。必须在模块顶层（spawn 启动方式）。"""
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    np.random.seed(seed)

    net = PolicyValueNet(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(net_state_dict)
    net.eval()

    ai = NeuralAI(net, device='cpu',
                  num_simulations=num_simulations,
                  temperature=1.0,
                  temperature_moves=temperature_moves,
                  add_noise=use_noise)

    game = Game()
    states, policies = [], []
    moves_count = 0

    while not game.is_over():
        if moves_count >= max_moves:
            break

        turn = game.turn
        legal = game.board.all_legal_moves(turn)
        if not legal:
            break

        move = ai.choose_move(game.board, turn)
        if move is None:
            break

        # 走子与训练标签必须来自同一次搜索，避免重复计算与标签错配。
        states.append(encode_state(game.board, turn))
        policies.append(ai.last_policy.copy())

        fr, fc, tr, tc = move
        game.move(fr, fc, tr, tc)
        moves_count += 1

    winner = game.winner()
    if len(states) == 0:
        return None
    return np.stack(states), np.stack(policies), winner, moves_count


# ============================================================
# 样本构建 + 训练一步
# ============================================================

def build_samples(states, policies, winner):
    """为每步配价值标签（从落子方视角）。"""
    n = len(states)
    values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mover = 'r' if states[i, 14, 0, 0] == 1 else 'b'
        if winner is None:
            values[i] = 0.0
        elif winner == mover:
            values[i] = 1.0
        else:
            values[i] = -1.0
    return states, policies, values


def train_step(net, optimizer, samples, batch_size=256, epochs=1, device='cpu',
               policy_coef=1.0, value_coef=0.5, entropy_coef=0.01,
               sample_weights=None, log_callback=None):
    """loss = policy_coef*CE + value_coef*MSE - entropy_coef*H。

    每 epoch 内随机打乱 minibatch。log_callback(epoch, ep_total, loss, acc)
    可选用于在每个 epoch 后打印进度。
    """
    states, policies, values = samples
    n = len(states)
    if n == 0:
        return 0.0, 0.0

    states_t = torch.from_numpy(states).to(device)
    policies_t = torch.from_numpy(policies).to(device)
    values_t = torch.from_numpy(values).to(device)
    weights_t = (torch.from_numpy(sample_weights.astype(np.float32)).to(device)
                 if sample_weights is not None else None)

    net.train()
    total_loss = total_acc = 0.0
    step = 0

    for ep in range(epochs):
        idx = np.arange(n)
        np.random.shuffle(idx)
        ep_loss = ep_acc = 0.0
        ep_steps = 0

        for s in range(0, n, batch_size):
            mb = idx[s:s + batch_size]
            x = states_t[mb]
            p_target = policies_t[mb]
            v_target = values_t[mb]
            w = weights_t[mb] if weights_t is not None else None

            p_logits, v_pred = net(x)
            log_p = F.log_softmax(p_logits, dim=-1)
            p = log_p.exp()

            pl = -(p_target * log_p).sum(dim=-1)
            vl = (v_pred.squeeze(-1) - v_target) ** 2
            ent = -(p * log_p).sum(dim=-1)

            if w is not None:
                policy_loss = (pl * w).sum() / w.sum().clamp(min=1e-6)
                value_loss = (vl * w).sum() / w.sum().clamp(min=1e-6)
                entropy_loss = (ent * w).sum() / w.sum().clamp(min=1e-6)
            else:
                policy_loss = pl.mean()
                value_loss = vl.mean()
                entropy_loss = ent.mean()

            loss = (policy_coef * policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                top = p.argmax(dim=-1)
                tgt = p_target.argmax(dim=-1)
                acc = (top == tgt).float().mean().item()

            ep_loss += loss.item()
            ep_acc += acc
            ep_steps += 1
            total_loss += loss.item()
            total_acc += acc
            step += 1

        if log_callback and ep_steps > 0:
            log_callback(ep + 1, epochs, ep_loss / ep_steps, ep_acc / ep_steps)

    net.eval()
    return total_loss / max(step, 1), total_acc / max(step, 1)


# ============================================================
# 进度条工具
# ============================================================

class ProgressBar:
    """简单的文本进度条，支持 ETA。"""

    def __init__(self, total, prefix='', width=30):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.t0 = time.perf_counter()
        self.times = []  # 最近几步耗时，用于估算

    def update(self, done, extra=None):
        ratio = min(done / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = '█' * filled + '░' * (self.width - filled)
        elapsed = time.perf_counter() - self.t0
        self.times.append(elapsed)
        if done > 0:
            eta = elapsed / done * (self.total - done)
            eta_str = _fmt_time(eta)
        else:
            eta_str = '?'
        msg = f"\r{self.prefix} |{bar}| {done}/{self.total}  {ratio*100:5.1f}%  " \
              f"用时 {_fmt_time(elapsed)}  剩余 {eta_str}"
        if extra:
            msg += "  " + "  ".join(f"{k}={v}" for k, v in extra.items())
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s//60)}m{int(s%60):02d}s"
    return f"{int(s//3600)}h{int((s%3600)//60):02d}m"


# ============================================================
# 主循环
# ============================================================

def get_device(args):
    """根据参数和硬件自动选择设备。"""
    if args.cpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def print_device_info(device):
    """打印设备详细信息。"""
    if device == 'cuda':
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[device] GPU: {name}  ({mem:.1f} GB)", flush=True)
    else:
        n = os.cpu_count() or 1
        print(f"[device] CPU: {n} cores (未检测到 CUDA，将用 CPU 训练)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5,
                        help='自对弈-训练迭代轮数（默认 5）')
    parser.add_argument('--games-per-iter', type=int, default=4,
                        help='每轮自对弈局数（CPU 跑 600 sims 时建议 2-4）')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='并行 worker 数（自对弈始终在 CPU 上跑）')
    parser.add_argument('--sims', type=int, default=600,
                        help='每步 MCTS 模拟次数（默认 600）')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-moves', type=int, default=120)
    parser.add_argument('--temperature-moves', type=int, default=10)
    parser.add_argument('--replay-size', type=int, default=30000)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=6)
    parser.add_argument('--save-path', type=str, default='model.pt')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--cpu', action='store_true',
                        help='强制用 CPU（默认自动：有 GPU 用 GPU，没有用 CPU）')
    args = parser.parse_args()
    for name in ('iterations', 'games_per_iter', 'num_workers', 'sims', 'epochs',
                 'batch_size', 'max_moves', 'replay_size', 'channels', 'save_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} 必须大于 0')
    if args.num_blocks < 0 or args.temperature_moves < 0 or args.lr <= 0:
        parser.error('残差块数、温度步数必须非负，学习率必须大于 0')
    torch.set_num_threads(min(4, torch.get_num_threads()))

    device = get_device(args)
    print_device_info(device)
    print(f"[config] iter={args.iterations}  games/iter={args.games_per_iter}  "
          f"workers={args.num_workers}  sims/move={args.sims}  "
          f"epochs={args.epochs}  batch={args.batch_size}  "
          f"net=({args.channels}ch x {args.num_blocks}blocks)  "
          f"lr={args.lr}", flush=True)

    # 主网络（GPU/CPU）
    net = PolicyValueNet(channels=args.channels,
                         num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if os.path.exists(args.save_path):
        sd = torch.load(args.save_path, map_location=device)
        net.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
        print(f"[load] 继续训练 {args.save_path}", flush=True)
    net.eval()

    replay_states = deque(maxlen=args.replay_size)
    replay_policies = deque(maxlen=args.replay_size)
    replay_values = deque(maxlen=args.replay_size)

    total_t0 = time.perf_counter()

    for it in range(1, args.iterations + 1):
        print(f"\n========== Iteration {it}/{args.iterations} ==========", flush=True)
        iter_t0 = time.perf_counter()

        # ---------- 自对弈 ----------
        net_state = {k: v.cpu() for k, v in net.state_dict().items()}
        base_seed = int(time.time() * 1000) & 0x7fffffff

        if args.num_workers <= 1:
            results = []
            bar = ProgressBar(args.games_per_iter, prefix="  自对弈")
            for g in range(args.games_per_iter):
                t_g = time.perf_counter()
                r = _play_one_self_game(
                    net_state, args.sims, args.max_moves,
                    args.temperature_moves, base_seed + g, True,
                    args.channels, args.num_blocks,
                )
                if r is not None:
                    results.append(r)
                dt = time.perf_counter() - t_g
                tag = '?'
                if r is not None:
                    tag = '红胜' if r[2] == 'r' else ('黑胜' if r[2] == 'b' else '和')
                bar.update(g + 1, {
                    '结果': tag,
                    '本局': f"{r[3] if r else 0}步",
                    '用时': _fmt_time(dt),
                })
            bar.close()
        else:
            print(f"  自对弈 {args.games_per_iter} 局 × {args.num_workers} workers 并行...",
                  flush=True)
            ctx = mp.get_context('spawn')
            with ctx.Pool(args.num_workers) as pool:
                jobs = [
                    (net_state, args.sims, args.max_moves,
                     args.temperature_moves, base_seed + g, True,
                     args.channels, args.num_blocks)
                    for g in range(args.games_per_iter)
                ]
                results = [r for r in pool.starmap(_play_one_self_game, jobs)
                           if r is not None]
            for i, r in enumerate(results):
                tag = '红胜' if r[2] == 'r' else ('黑胜' if r[2] == 'b' else '和')
                print(f"    局 {i+1}/{len(results)}: {tag}  {r[3]}步", flush=True)

        # ---------- 收集样本 ----------
        win_count = {'r': 0, 'b': 0, 'd': 0}
        for states, policies, winner, moves in results:
            s, p, v = build_samples(states, policies, winner)
            replay_states.extend(s)
            replay_policies.extend(p)
            replay_values.extend(v)
            win_count[winner if winner else 'd'] = win_count.get(winner if winner else 'd', 0) + 1

        sp_time = time.perf_counter() - iter_t0
        print(f"  [selfplay] 局数={len(results)}  "
              f"红胜={win_count.get('r',0)}  黑胜={win_count.get('b',0)}  "
              f"和={win_count.get('d',0)}  "
              f"样本池={len(replay_states)}  "
              f"用时 {_fmt_time(sp_time)}", flush=True)

        if len(replay_states) == 0:
            print("  [train] 样本不足，跳过", flush=True)
            continue

        states_np = np.stack(list(replay_states))
        policies_np = np.stack(list(replay_policies))
        values_np = np.stack(list(replay_values))

        # ---------- 训练 ----------
        t_train0 = time.perf_counter()

        def _on_epoch(ep, ep_total, ep_loss, ep_acc):
            print(f"    [train] epoch {ep}/{ep_total}  loss={ep_loss:.4f}  "
                  f"policy_acc={ep_acc:.4f}", flush=True)

        loss, acc = train_step(
            net, optimizer,
            (states_np, policies_np, values_np),
            batch_size=args.batch_size, epochs=args.epochs, device=device,
            log_callback=_on_epoch,
        )
        tr_time = time.perf_counter() - t_train0
        print(f"  [train]  平均 loss={loss:.4f}  policy_acc={acc:.4f}  "
              f"用时 {_fmt_time(tr_time)}", flush=True)

        # ---------- 保存 ----------
        if it % args.save_every == 0 or it == args.iterations:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
            torch.save({'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': {'channels': args.channels, 'num_blocks': args.num_blocks},
                        'iter': it}, args.save_path)
            print(f"  [save]  {args.save_path}", flush=True)

        # ---------- 总体进度 ----------
        total_elapsed = time.perf_counter() - total_t0
        avg_per_iter = total_elapsed / it
        eta_total = avg_per_iter * (args.iterations - it)
        print(f"  [进度]  本轮用时 {_fmt_time(time.perf_counter()-iter_t0)}  "
              f"累计 {_fmt_time(total_elapsed)}  "
              f"剩余 ~{_fmt_time(eta_total)}", flush=True)

    total_time = time.perf_counter() - total_t0
    print(f"\n========== 训练结束 ==========", flush=True)
    print(f"总用时: {_fmt_time(total_time)}", flush=True)
    print(f"模型已保存到: {args.save_path}", flush=True)


if __name__ == '__main__':
    main()

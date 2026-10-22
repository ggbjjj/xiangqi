"""象棋入口：uv run xiangqi，或 uv run python main.py。"""
import argparse
import re
import sys
from ai import NeuralAI, SearchAI, load_network
from game import Game


def _parse_move(text):
    text = text.strip()
    if not any(re.fullmatch(pattern, text) for pattern in (
            r'[0-9]{4}', r'[0-9]{2}[\s-]+[0-9]{2}',
            r'[0-9]\s*,\s*[0-9]\s+[0-9]\s*,\s*[0-9]')):
        return None
    digits = re.sub(r'[\s,\-]', '', text)
    if len(digits) != 4:
        return None
    move = tuple(map(int, digits))
    return move if move[1] < 9 and move[3] < 9 else None


def _maybe_load_net(model_path, channels, blocks):
    try:
        return load_network(model_path, channels, blocks)
    except Exception as error:
        print(f'[提示] 模型不可用，使用内置 AI：{error}')
        return None


def cmd_gui(args):
    from ui import run_app
    mode = args.mode
    if args.ai_color:
        mode = 'pve_black' if args.ai_color == 'r' else 'pve_red'
    run_app(model_path=args.model, channels=args.channels, blocks=args.blocks,
            sims=args.sims, initial_mode=mode, depth=args.depth)


def cmd_cli(args):
    mode = args.cli_cmd
    game = Game()
    ai = None
    player_color = None
    if mode != 'pvp':
        net = _maybe_load_net(args.model, args.channels, args.blocks)
        ai = (NeuralAI(net, num_simulations=args.sims) if net is not None
              else SearchAI(max_depth=args.depth))
        print('AI：' + ('神经网络搜索' if net is not None else '内置搜索'))
        if mode == 'pve':
            player_color = Game.opponent(args.ai_color)
    print('走法：60-50 / 6,0 5,0 / 6050；u 悔棋，q 退出')
    while not game.is_over():
        print(game.board)
        side = '红' if game.turn == 'r' else '黑'
        if mode == 'ai' or (mode == 'pve' and game.turn != player_color):
            move = ai.choose_move(game.board, game.turn)
            if move is None:
                game._evaluate_end()
                break
            print(f'{side}方 AI：{move}')
        else:
            try:
                text = input(f'{side}方走子：').strip().lower()
            except EOFError:
                return
            if text in ('q', 'quit'):
                return
            if text in ('u', 'undo'):
                count = game.undo_turn(player_color)
                if ai:
                    ai.reset()
                print(f'已撤销 {count} 步')
                continue
            move = _parse_move(text)
            if move is None:
                print('格式或坐标错误')
                continue
        if game.move(*move) != 'ok':
            print('非法走法')
        elif game.board.is_in_check(game.turn) and not game.is_over():
            print('将军！请应将。')
    print(game.board)
    print(f'终局：{game.status}')


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必须是正整数')
    return number


def _add_ai_options(parser):
    parser.add_argument('--model', default='model.pt', help='可选训练模型路径')
    parser.add_argument('--sims', type=positive_int, default=50, help='神经网络搜索次数')
    parser.add_argument('--channels', type=positive_int, default=64, help='旧模型通道数')
    parser.add_argument('--blocks', type=int, default=6, help='旧模型残差块数')
    parser.add_argument('--depth', type=int, choices=(1, 2, 3, 4), default=2,
                        help='内置 AI 搜索深度（每步限时）')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ['gui']
    parser = argparse.ArgumentParser(description='中国象棋：双人、人机和 AI 观战')
    sub = parser.add_subparsers(dest='cmd', required=True)
    gui = sub.add_parser('gui', help='图形界面（默认）')
    _add_ai_options(gui)
    gui.add_argument('--mode', choices=['pvp', 'pve_red', 'pve_black', 'ai_vs_ai'],
                     help='跳过菜单，直接进入指定模式')
    gui.add_argument('--ai-color', choices=['r', 'b'], help='直接进入人机对弈，指定 AI 颜色')
    gui.set_defaults(func=cmd_gui)
    cli = sub.add_parser('cli', help='命令行对弈')
    cli_sub = cli.add_subparsers(dest='cli_cmd', required=True)
    for mode, help_text in [('pvp', '双人'), ('pve', '人机'), ('ai', 'AI 自对弈')]:
        child = cli_sub.add_parser(mode, help=help_text)
        if mode != 'pvp':
            _add_ai_options(child)
        if mode == 'pve':
            child.add_argument('--ai-color', choices=['r', 'b'], default='b')
        child.set_defaults(func=cmd_cli)
    args = parser.parse_args(argv)
    if getattr(args, 'blocks', 0) < 0:
        parser.error('--blocks 不得小于 0')
    try:
        args.func(args)
    except KeyboardInterrupt:
        print('\n已退出。')


if __name__ == '__main__':
    main()

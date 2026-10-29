"""象棋入口：uv run xiangqi，或 uv run python main.py。"""
import argparse
import re
import sys
from ai import NeuralAI, SearchAI, load_network
from game import Game
from project_paths import MODEL_CATALOG as MODEL_DB


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


def _maybe_load_net(model_path, model_name=None, *, family=None,parameter_id=None):
    if family is not None:
        return load_network(model_path,family=family,parameter_id=parameter_id)
    try:
        return load_network(model_path, model_name)
    except Exception as error:
        print(f'[提示] 模型不可用，使用内置 AI：{error}')
        return None


def cmd_gui(args):
    from ui import run_app
    mode = args.mode
    if args.ai_color:
        mode = 'pve_black' if args.ai_color == 'r' else 'pve_red'
    options={}
    if args.family is not None: options={'parameter_family':args.family,'parameter_id':args.parameter_id}
    run_app(model_path=args.model_db, model_name=args.model, initial_mode=mode, depth=args.depth,**options)


def cmd_cli(args):
    mode = args.cli_cmd
    game = Game()
    ai = None
    player_color = None
    if mode != 'pvp':
        net = _maybe_load_net(args.model_db, args.model,family=args.family,parameter_id=args.parameter_id)
        ai = (NeuralAI(net) if net is not None
              else SearchAI(max_depth=args.depth))
        print('AI：' + (net.display_name if net is not None else '内置搜索'))
        if mode == 'pve':
            player_color = Game.opponent(args.ai_color)
    print('走法：60-50 / 6,0 5,0 / 6050；u 悔棋，q 退出')
    while not game.is_over():
        print(game.board)
        side = '红' if game.turn == 'r' else '黑'
        if mode == 'ai' or (mode == 'pve' and game.turn != player_color):
            kwargs = {'history':game.history} if isinstance(ai,(NeuralAI,SearchAI)) else {}
            move = ai.choose_move(game.board, game.turn, **kwargs)
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
    parser.add_argument('--model-db', default=str(MODEL_DB), help='模型参数目录（汇总 HaiKu/Opus）或单个 SQLite 数据库')
    parser.add_argument('--model', help='公开模型名称，例如 HaiKu-5；图形界面预选此模型，开局前确认')
    parser.add_argument('--family',choices=['HaiKu','Opus'],help='按模型系列加载参数，未指定 ID 时使用该系列默认 ID')
    parser.add_argument('--parameter-id','--id',dest='parameter_id',help='例如 ID2 或 2，需同时指定 --family')
    parser.add_argument('--checkpoint', help='后台启动配置：与 --model 一起使用，将该名称绑定到参数 ID 并保存')
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
    if getattr(args,'parameter_id',None) is not None and not getattr(args,'family',None):
        parser.error('--id 需要 --family，以区分不同模型系列的编号')
    if getattr(args,'family',None) is not None:
        if args.model or args.checkpoint: parser.error('--family/--id 与公开名称 --model/--checkpoint 分别使用')
        from model_store import get_parameter_config
        try: args.parameter_id=get_parameter_config(args.family,args.parameter_id,path=args.model_db)['parameter_id']
        except ValueError as error: parser.error(str(error))
    if getattr(args,'model',None):
        args.model = args.model.strip()
    if getattr(args,'checkpoint',None):
        if not args.model:
            parser.error('--checkpoint 需要同时指定 --model 公开名称，例如 --model HaiKu-5 --checkpoint 参数ID')
        from model_store import publish_model
        import sqlite3
        try:
            publish_model(args.model,args.checkpoint,path=args.model_db)
        except (ValueError,sqlite3.Error) as error:
            parser.error(f'后台模型绑定失败：{error}')
        print(f'[后台模型绑定] {args.model} -> {args.checkpoint}')
    try:
        args.func(args)
    except KeyboardInterrupt:
        print('\n已退出。')


if __name__ == '__main__':
    main()

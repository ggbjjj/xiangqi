"""Pygame 象棋：主线程绘图，后台搜索，取消后丢弃过期结果。"""
import os
import queue
import threading
from concurrent.futures import CancelledError

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from ai import NeuralAI, SearchAI, load_network
from constants import COLS, ROWS, NAMES_BLACK, NAMES_RED
from game import Game

CELL = 56
MARGIN_X, MARGIN_Y = 40, 40
INFO_H = 72
BOARD_W, BOARD_H = 528, 584
GAME_W, GAME_H = 760, 720
MENU_W, MENU_H = 560, 580
COLOR_BG = (239, 211, 165)
COLOR_LINE = (93, 62, 36)
COLOR_DARK = (48, 44, 39)
COLOR_LIGHT = (249, 239, 220)
COLOR_GOLD = (226, 180, 102)
COLOR_RED = (176, 48, 44)
MODE_LABELS = {'pvp': '双人对弈', 'pve_red': '玩家执红 · AI 执黑',
               'pve_black': '玩家执黑 · AI 执红', 'ai_vs_ai': 'AI 自对弈 · 观战'}


def board_to_pixel(r, c):
    return MARGIN_X + c * CELL, INFO_H + MARGIN_Y + r * CELL


def pixel_to_board(x, y):
    c = round((x - MARGIN_X) / CELL)
    r = round((y - INFO_H - MARGIN_Y) / CELL)
    if 0 <= r < ROWS and 0 <= c < COLS:
        return r, c
    return None


def font(size, bold=False):
    return pygame.font.SysFont('microsoftyahei,simhei,notosanscjksc,wenquanyi', size, bold=bold)


def text_fit(surface, text, face, color, rect, center=False):
    shown = text
    while shown and face.size(shown)[0] > rect.width:
        text = text[:-1]
        shown = text + '…'
        if not text:
            break
    image = face.render(shown, True, color)
    surface.blit(image, image.get_rect(center=rect.center) if center else rect.topleft)


class MainMenu:
    MODES = list(MODE_LABELS.items())

    def __init__(self, model_loaded=False, model_path='model.pt'):
        pygame.init()
        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
        pygame.display.set_caption('中国象棋 · 选择模式')
        self.clock = pygame.time.Clock()
        self.title_font, self.button_font, self.small_font = font(42, True), font(22), font(16)
        self.model_loaded = model_loaded
        self.choice = None
        self.btn_rects = [(pygame.Rect(70, 162 + i * 76, 420, 56), mode)
                          for i, (mode, _) in enumerate(self.MODES)]

    def run(self):
        while self.choice is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, mode in self.btn_rects:
                        if rect.collidepoint(event.pos):
                            self.choice = mode
                if event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                    self.choice = self.MODES[event.key - pygame.K_1][0]
            self._draw()
            self.clock.tick(30)
        return self.choice

    def _draw(self):
        self.screen.fill(COLOR_DARK)
        text_fit(self.screen, '中国象棋', self.title_font, COLOR_GOLD, pygame.Rect(0, 48, MENU_W, 60), True)
        text_fit(self.screen, '选择对弈模式，也可按数字键 1–4', self.small_font,
                 COLOR_LIGHT, pygame.Rect(0, 115, MENU_W, 30), True)
        for i, ((rect, mode), (_, label)) in enumerate(zip(self.btn_rects, self.MODES), 1):
            color = (128, 94, 57) if rect.collidepoint(pygame.mouse.get_pos()) else (94, 74, 51)
            pygame.draw.rect(self.screen, color, rect, border_radius=9)
            text_fit(self.screen, f'{i}   {label}', self.button_font, COLOR_LIGHT, rect, True)
        status = '已启用训练模型' if self.model_loaded else '内置搜索 AI 已就绪 · 无需训练模型'
        text_fit(self.screen, status, self.small_font, COLOR_GOLD, pygame.Rect(0, 494, MENU_W, 30), True)
        text_fit(self.screen, '红方先行 · 合法落点提示 · 支持悔棋', self.small_font,
                 COLOR_LIGHT, pygame.Rect(0, 532, MENU_W, 24), True)
        pygame.display.flip()


class XiangqiUI:
    def __init__(self, mode, net=None, num_simulations=50, depth=2):
        if mode not in MODE_LABELS:
            raise ValueError(f'unknown mode: {mode}')
        pygame.init()
        self.mode, self.net = mode, net
        self.num_simulations, self.depth = num_simulations, depth
        self.player_color = {'pve_red': 'r', 'pve_black': 'b'}.get(mode)
        self.ai_color = Game.opponent(self.player_color) if self.player_color else None
        self.screen = pygame.display.set_mode((GAME_W, GAME_H))
        pygame.display.set_caption(f'中国象棋 · {MODE_LABELS[mode]}')
        self.clock = pygame.time.Clock()
        self.font, self.small_font, self.button_font = font(34, True), font(17), font(18)
        self.game = Game()
        self.selected, self.legal_targets, self.last_move = None, [], None
        self.message = '红方先行，请选择棋子'
        self.ai = self._make_ai() if mode != 'pvp' else None
        self.paused = False
        self.running = False
        self.exit_to_menu = False
        self._generation = 0
        self._cancel_event = None
        self._results = queue.Queue()
        self.thinking = False
        self._ai_due = 0
        self.buttons = [('新局', self.action_new), ('悔棋', self.action_undo),
                        ('暂停/继续', self.action_pause), ('返回菜单', self.action_back),
                        ('退出', self.action_quit)]
        self.btn_rects = [(pygame.Rect(12 + i * 149, 672, 140, 36), label)
                          for i, (label, _) in enumerate(self.buttons)]

    def _make_ai(self):
        if self.net is not None:
            return NeuralAI(self.net, num_simulations=self.num_simulations)
        return SearchAI(max_depth=self.depth)

    def _is_ai_turn(self):
        return self.ai is not None and (self.mode == 'ai_vs_ai' or self.game.turn == self.ai_color)

    def _invalidate_ai(self):
        self._generation += 1
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None
        self.thinking = False

    def _clear_selection(self):
        self.selected, self.legal_targets = None, []

    def _refresh_last_move(self):
        last = self.game.history[-1] if self.game.history else None
        self.last_move = tuple(last[k] for k in ('fr', 'fc', 'tr', 'tc')) if last else None

    def action_new(self):
        self._invalidate_ai()
        self.game.reset()
        self._clear_selection()
        self.last_move = None
        self.paused = False
        if self.ai:
            self.ai = self._make_ai()
        self.message = '新局开始，红方先行'
        self._ai_due = pygame.time.get_ticks() + 300

    def action_undo(self):
        self._invalidate_ai()
        count = self.game.undo_turn(self.player_color)
        self._clear_selection()
        self._refresh_last_move()
        if self.ai:
            self.ai = self._make_ai()
        if self.mode == 'ai_vs_ai':
            self.paused = True
        self.message = f'已悔棋 {count} 步' if count else '暂无可撤销的走子'
        self._ai_due = pygame.time.get_ticks() + 300

    def action_pause(self):
        if self.mode != 'ai_vs_ai':
            self.message = '暂停功能用于 AI 观战'
            return
        self._invalidate_ai()
        self.paused = not self.paused
        self.message = '观战已暂停' if self.paused else '继续 AI 对弈'

    def action_back(self):
        self._invalidate_ai()
        self.exit_to_menu, self.running = True, False

    def action_quit(self):
        self._invalidate_ai()
        self.exit_to_menu, self.running = False, False

    def run(self):
        self.running = True
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.action_quit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._on_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        action = {pygame.K_ESCAPE: self.action_back, pygame.K_b: self.action_back,
                                  pygame.K_u: self.action_undo, pygame.K_n: self.action_new,
                                  pygame.K_SPACE: self.action_pause}.get(event.key)
                        if action:
                            action()
                    if not self.running:
                        break
                if not self.running:
                    break
                self._poll_ai()
                if (self._is_ai_turn() and not self.game.is_over() and not self.paused
                        and not self.thinking and pygame.time.get_ticks() >= self._ai_due):
                    self._ai_move()
                self._draw()
                self.clock.tick(30)
        finally:
            self._invalidate_ai()
        return self.exit_to_menu

    def _ai_move(self):
        if self.thinking or self.paused or self.game.is_over() or not self._is_ai_turn():
            return
        self.thinking = True
        token, board, turn = self._generation, self.game.board.clone(), self.game.turn
        cancel = self._cancel_event = threading.Event()
        ai, results = self.ai, self._results
        self.message = f'{"红" if turn == "r" else "黑"}方 AI 思考中…'

        def worker():
            try:
                move = ai.choose_move(board, turn, cancel_event=cancel)
                results.put((token, turn, move, None))
            except CancelledError:
                pass
            except Exception as error:
                results.put((token, turn, None, str(error)))

        threading.Thread(target=worker, daemon=True, name='xiangqi-ai').start()

    def _poll_ai(self):
        while True:
            try:
                token, turn, move, error = self._results.get_nowait()
            except queue.Empty:
                break
            if token != self._generation or turn != self.game.turn:
                continue
            self.thinking = False
            self._cancel_event = None
            if error:
                self.ai = SearchAI(max_depth=self.depth)
                self.message = 'AI 计算异常，已切换内置搜索'
                print(f'[AI] {error}')
                self._ai_due = pygame.time.get_ticks() + 800
            elif move is None:
                self.game._evaluate_end()
                if not self.game.is_over():
                    self.ai = SearchAI(max_depth=self.depth)
                self.message = self.game.status or 'AI 未返回走法，正在重试'
            elif self.game.move(*move) == 'ok':
                self._after_move()
            else:
                self.ai = SearchAI(max_depth=self.depth)
                self.message = 'AI 返回非法走法，已切换内置搜索'

    def _after_move(self):
        self._clear_selection()
        self._refresh_last_move()
        side = '红' if self.game.turn == 'r' else '黑'
        self.message = self.game.status or (f'将军！{side}方请应将' if self.game.board.is_in_check(self.game.turn)
                                           else f'轮到{side}方走子')
        self._ai_due = pygame.time.get_ticks() + (500 if self.mode == 'ai_vs_ai' else 150)

    def _on_click(self, pos):
        for rect, label in self.btn_rects:
            if rect.collidepoint(pos):
                dict(self.buttons)[label]()
                return
        if self.game.is_over() or self._is_ai_turn() or self.mode == 'ai_vs_ai':
            return
        square = pixel_to_board(*pos)
        if square is None:
            return
        r, c = square
        if self.selected == square:
            self._clear_selection()
            return
        piece = self.game.board.piece_at(r, c)
        if piece and piece.color == self.game.turn:
            self.selected = square
            self.legal_targets = self.game.legal_moves_at(r, c)
            self.message = f'已选{(NAMES_RED if piece.color == "r" else NAMES_BLACK)[piece.kind]}，绿色位置可落子'
        elif self.selected and square in self.legal_targets:
            if self.game.move(*self.selected, r, c) == 'ok':
                self._after_move()
        elif self.selected:
            self.message = '此处不可落子，请选择绿色提示位置'

    def _draw(self):
        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_DARK, (0, 0, GAME_W, INFO_H))
        side = '红' if self.game.turn == 'r' else '黑'
        status = '对局结束' if self.game.is_over() else f'{side}方回合'
        text_fit(self.screen, f'{MODE_LABELS[self.mode]}     {status}     第 {len(self.game.history)} 步',
                 self.button_font, COLOR_LIGHT, pygame.Rect(18, 10, GAME_W - 36, 25))
        text_fit(self.screen, self.message, self.small_font, COLOR_GOLD, pygame.Rect(18, 41, GAME_W - 36, 24))
        for c in range(COLS):
            segments = [(0, 9)] if c in (0, 8) else [(0, 4), (5, 9)]
            for start, end in segments:
                pygame.draw.line(self.screen, COLOR_LINE, board_to_pixel(start, c), board_to_pixel(end, c), 2)
        for r in range(ROWS):
            pygame.draw.line(self.screen, COLOR_LINE, board_to_pixel(r, 0), board_to_pixel(r, 8), 2)
        for r in (0, 7):
            pygame.draw.line(self.screen, COLOR_LINE, board_to_pixel(r, 3), board_to_pixel(r + 2, 5), 2)
            pygame.draw.line(self.screen, COLOR_LINE, board_to_pixel(r, 5), board_to_pixel(r + 2, 3), 2)
        text_fit(self.screen, '楚 河          汉 界', self.button_font, COLOR_LINE,
                 pygame.Rect(MARGIN_X, INFO_H + MARGIN_Y + 4 * CELL, 8 * CELL, CELL), True)
        if self.last_move:
            for r, c in (self.last_move[:2], self.last_move[2:]):
                x, y = board_to_pixel(r, c)
                pygame.draw.rect(self.screen, (169, 115, 47), (x - 26, y - 26, 52, 52), 3, border_radius=5)
        for r, row in enumerate(self.game.board.grid):
            for c, piece in enumerate(row):
                if piece is None:
                    continue
                x, y = board_to_pixel(r, c)
                pygame.draw.circle(self.screen, (183, 148, 105), (x + 2, y + 3), 25)
                pygame.draw.circle(self.screen, COLOR_LIGHT, (x, y), 25)
                color = COLOR_RED if piece.color == 'r' else COLOR_DARK
                pygame.draw.circle(self.screen, color, (x, y), 22, 1)
                name = (NAMES_RED if piece.color == 'r' else NAMES_BLACK)[piece.kind]
                image = self.font.render(name, True, color)
                self.screen.blit(image, image.get_rect(center=(x, y - 1)))
        if self.selected:
            pygame.draw.circle(self.screen, (36, 130, 96), board_to_pixel(*self.selected), 27, 3)
        for target in self.legal_targets:
            pygame.draw.circle(self.screen, (36, 153, 105), board_to_pixel(*target),
                               26 if self.game.board.piece_at(*target) else 7, 3 if self.game.board.piece_at(*target) else 0)
        if not self.game.is_over() and self.game.board.is_in_check(self.game.turn):
            king = self.game.board.find_king(self.game.turn)
            if king:
                pygame.draw.circle(self.screen, COLOR_RED, board_to_pixel(*king), 28, 3)
        self._draw_sidebar()
        for rect, label in self.btn_rects:
            color = (210, 173, 119) if rect.collidepoint(pygame.mouse.get_pos()) else (221, 187, 138)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            text_fit(self.screen, label, self.button_font, COLOR_DARK, rect, True)
        pygame.display.flip()

    def _draw_sidebar(self):
        pygame.draw.rect(self.screen, (229, 199, 155), (BOARD_W, INFO_H, GAME_W - BOARD_W, BOARD_H))
        lines = ['对局信息', '红先 · 将死 / 困毙判负',
                 '选棋子，再点击绿色落点', '',
                 'U  悔棋      N  新局', 'B / Esc  返回菜单', '空格  暂停 / 继续观战', '',
                 'AI：' + ('未启用' if self.ai is None else '训练模型' if isinstance(self.ai, NeuralAI) else '内置搜索'),
                 '状态：' + ('已结束' if self.game.is_over() else '暂停' if self.paused else '思考中' if self.thinking else '进行中'),
                 '', '最近走子']
        for i, line in enumerate(lines):
            text_fit(self.screen, line, self.small_font, COLOR_DARK, pygame.Rect(542, 92 + i * 26, 206, 24))
        for i, step in enumerate(self.game.history[-7:]):
            name = (NAMES_RED if step['turn'] == 'r' else NAMES_BLACK)[step['piece'][1]]
            label = f'{"红" if step["turn"] == "r" else "黑"}{name} {step["fr"]}{step["fc"]} → {step["tr"]}{step["tc"]}'
            text_fit(self.screen, label, self.small_font, COLOR_LINE, pygame.Rect(542, 414 + i * 28, 206, 24))


def run_app(model_path='model.pt', channels=64, blocks=6, sims=50, initial_mode=None, depth=2):
    net = None
    try:
        net = load_network(model_path, channels, blocks)
    except Exception as error:
        print(f'[提示] 模型加载失败，使用内置搜索 AI：{error}')
    mode = initial_mode
    try:
        while True:
            if mode is None:
                mode = MainMenu(net is not None, model_path).run()
            if mode is None:
                break
            if not XiangqiUI(mode, net, sims, depth).run():
                break
            mode = None
    finally:
        pygame.quit()


if __name__ == '__main__':
    run_app()

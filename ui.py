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
from project_paths import MODEL_CATALOG as MODEL_DB
from model_store import list_hint_models, load_hint_model, default_model_name, get_parameter_config, load_value_model

CELL = 56
MARGIN_X, MARGIN_Y = 40, 40
INFO_H = 72
BOARD_W, BOARD_H = 528, 584
GAME_W, GAME_H = 1020, 720
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

    def __init__(self, model_loaded=False, model_path=None):
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
        status = '人机对弈开局前选择模型'
        text_fit(self.screen, status, self.small_font, COLOR_GOLD, pygame.Rect(0, 494, MENU_W, 30), True)
        text_fit(self.screen, '红方先行 · 合法落点提示 · 支持悔棋', self.small_font,
                 COLOR_LIGHT, pygame.Rect(0, 532, MENU_W, 24), True)
        pygame.display.flip()


class OpponentModelMenu:
    """Select and load an opponent before a match; only public names are rendered."""

    def __init__(self, mode, model_db=MODEL_DB, preferred_name=None,*,parameter_family=None,parameter_id=None):
        pygame.init()
        self.screen = pygame.display.set_mode((MENU_W,MENU_H))
        pygame.display.set_caption('中国象棋 · 选择对弈模型')
        self.clock = pygame.time.Clock()
        self.title_font, self.button_font, self.small_font = font(32,True),font(22),font(16)
        self.mode, self.model_db = mode,model_db
        self.page, self.selected_name = 0,None
        self.loading, self.result = False,None
        self._results = queue.Queue()
        self.message = '请选择模型，再点击开始对弈'
        self._pinned=None
        try:
            if parameter_family is not None:
                row=get_parameter_config(parameter_family,parameter_id,path=model_db)
                preferred_name=row['version_name']
                self._pinned=(preferred_name,parameter_family,row['parameter_id'])
                self.choices=[{'name':preferred_name}]
            else: self.choices = list_hint_models(model_db)
        except Exception as error:
            print(f'[对弈模型列表] {error}')
            self.choices = []
            self.message = '模型列表暂不可用，可稍后重试'
        self.neural_names = {choice['name'] for choice in self.choices}
        if preferred_name is None:
            try: preferred_name = default_model_name(model_db)
            except Exception: pass
        if '内置搜索' not in self.neural_names:
            self.choices = self.choices + [{'name':'内置搜索'}]
        if preferred_name in {choice['name'] for choice in self.choices}:
            self.selected_name = preferred_name
            self.page = next(i for i,c in enumerate(self.choices) if c['name']==preferred_name)//5
        elif preferred_name:
            self.message = '指定模型暂不可用，请重新选择'
        self.model_rects = []
        self.back_rect, self.start_rect = pygame.Rect(34,532,130,40),pygame.Rect(340,532,186,40)
        self.prev_rect, self.next_rect = pygame.Rect(34,455,110,30),pygame.Rect(416,455,110,30)

    def _start(self):
        if self.loading: return
        if self.selected_name is None:
            self.message = '请先选择对弈模型'
            return
        if self.selected_name not in self.neural_names:
            self.result = ('start',None)
            return
        self.loading = True
        self.message = '正在准备对弈…'
        path,name,results = self.model_db,self.selected_name,self._results

        def worker():
            try:
                if self._pinned and name==self._pinned[0]:
                    net=load_network(path,family=self._pinned[1],parameter_id=self._pinned[2])
                else: net = load_network(path,name)
                if net is None: raise ValueError('所选模型不可用')
                results.put((net,None))
            except Exception as error:
                print(f'[对弈模型加载] {error}')
                results.put((None,'该模型暂不可用，请重试或重新选择'))
        threading.Thread(target=worker,daemon=True,name='xiangqi-opponent-load').start()

    def _poll_loading(self):
        if self.result is not None: return
        try: net,error = self._results.get_nowait()
        except queue.Empty: return
        self.loading = False
        if error:
            self.message = error
        else:
            self.result = ('start',net)

    def _on_click(self, pos):
        if self.back_rect.collidepoint(pos):
            self.result = ('back',None)
            return
        if self.loading: return
        if self.start_rect.collidepoint(pos):
            self._start()
        elif self.prev_rect.collidepoint(pos):
            self.page = max(0,self.page-1)
        elif self.next_rect.collidepoint(pos):
            self.page = min((len(self.choices)-1)//5,self.page+1)
        else:
            for rect,choice in self.model_rects:
                if rect.collidepoint(pos):
                    self.selected_name = choice['name']
                    self.message = '已选择模型，点击开始对弈'
                    break

    def run(self):
        while self.result is None:
            self._draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.result = ('quit',None)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.result = ('back',None)
                    elif event.key == pygame.K_RETURN:
                        self._start()
                    elif not self.loading and pygame.K_1 <= event.key <= pygame.K_5:
                        index = self.page*5+event.key-pygame.K_1
                        if index < len(self.choices): self.selected_name = self.choices[index]['name']
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_click(event.pos)
                if self.result is not None: break
            if self.result is None: self._poll_loading()
            self.clock.tick(30)
        return self.result

    def _draw(self):
        self.screen.fill(COLOR_DARK)
        text_fit(self.screen,'选择对弈模型',self.title_font,COLOR_GOLD,pygame.Rect(20,25,520,45),True)
        text_fit(self.screen,MODE_LABELS[self.mode],self.small_font,COLOR_LIGHT,pygame.Rect(20,78,520,25),True)
        text_fit(self.screen,'本局固定使用所选模型，换模型请重新开局',self.small_font,
                 COLOR_LIGHT,pygame.Rect(20,108,520,26),True)
        self.model_rects = []
        for i,choice in enumerate(self.choices[self.page*5:(self.page+1)*5]):
            rect = pygame.Rect(34,150+i*60,492,50)
            selected = choice['name']==self.selected_name
            pygame.draw.rect(self.screen,COLOR_GOLD if selected else (94,74,51),rect,border_radius=7)
            text_fit(self.screen,choice['name'],self.button_font,COLOR_DARK if selected else COLOR_LIGHT,rect,True)
            self.model_rects.append((rect,choice))
        for label,rect in [('上一页',self.prev_rect),('下一页',self.next_rect)]:
            text_fit(self.screen,label,self.small_font,COLOR_LIGHT,rect,True)
        text_fit(self.screen,f'{self.page+1} / {(len(self.choices)+4)//5}',self.small_font,
                 COLOR_LIGHT,pygame.Rect(205,455,150,30),True)
        text_fit(self.screen,self.message,self.small_font,COLOR_GOLD,pygame.Rect(34,494,492,26),True)
        for label,rect in [('返回',self.back_rect),('准备中…' if self.loading else '开始对弈',self.start_rect)]:
            enabled = rect==self.back_rect or (self.selected_name is not None and not self.loading)
            pygame.draw.rect(self.screen,COLOR_GOLD if enabled else (94,74,51),rect,border_radius=7)
            text_fit(self.screen,label,self.button_font,COLOR_DARK if enabled else COLOR_LIGHT,rect,True)
        pygame.display.flip()


class XiangqiUI:
    def __init__(self, mode, net=None, num_simulations=50, depth=2, model_db=MODEL_DB):
        if mode not in MODE_LABELS:
            raise ValueError(f'unknown mode: {mode}')
        pygame.init()
        self.mode, self.net = mode, net
        self._match_net = net
        self.opponent_name = getattr(net,'display_name','AI 模型') if net is not None else '内置搜索'
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
        self.restart_requested, self.ai_failed = False,False
        self._generation = 0
        self._cancel_event = None
        self._results = queue.Queue()
        self.thinking = False
        self._ai_due = 0
        self.model_db = model_db
        self.hint_rect = pygame.Rect(542,322,204,38)
        self.hints, self.hint_rows = [], []
        self.hint_model_name = ''
        self.hint_busy, self.model_dialog = False, False
        self.model_choices, self.model_rects = [], []
        self.model_page = 0
        self.dialog_message = ''
        self._hint_generation, self._hint_cancel = 0, None
        self._hint_results = queue.Queue()
        self.value_prediction=None
        self.value_model_name=''
        self.value_message='等待局势分析'
        self.value_busy=False
        self._value_generation=0
        self._value_requested=False
        self._value_cancel=None
        self._value_results=queue.Queue()
        self._value_lock=threading.Lock()
        self._value_ai=None
        self.buttons = [('新局', self.action_new), ('悔棋', self.action_undo),
                        ('暂停/继续', self.action_pause), ('返回菜单', self.action_back),
                        ('退出', self.action_quit)]
        self.btn_rects = [(pygame.Rect(12 + i * 201, 672, 192, 36), label)
                          for i, (label, _) in enumerate(self.buttons)]

    def _make_ai(self):
        if self._match_net is not None:
            return NeuralAI(self._match_net)
        return SearchAI(max_depth=self.depth)

    def _is_ai_turn(self):
        return self.ai is not None and (self.mode == 'ai_vs_ai' or self.game.turn == self.ai_color)

    def _invalidate_ai(self):
        self._invalidate_value()
        self._invalidate_hints()
        self._generation += 1
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None
        self.thinking = False

    def _invalidate_hints(self):
        self._hint_generation += 1
        if self._hint_cancel is not None:
            self._hint_cancel.set()
        self._hint_cancel = None
        self.hint_busy = False
        self.hints, self.hint_rows = [], []
        self.model_dialog = False

    def _invalidate_value(self):
        self._value_generation+=1
        if self._value_cancel is not None: self._value_cancel.set()
        self._value_cancel=None
        self._value_requested=False
        self.value_busy=False
        self.value_prediction=None
        self.value_message='等待局势分析'

    def _request_value(self):
        if self._value_requested: return
        self._value_requested=True
        if self.game.is_over():
            self.value_message='对局已结束'
            return
        self.value_busy=True
        self.value_message='正在预测当前局势…'
        token=self._value_generation
        board,turn,history=self.game.board.clone(),self.game.turn,list(self.game.history)
        cancel=self._value_cancel=threading.Event()
        def worker():
            try:
                with self._value_lock:
                    if cancel.is_set(): raise CancelledError()
                    if self._value_ai is None:
                        if self._match_net is not None and hasattr(self._match_net,'value_head'):
                            if not getattr(self._match_net,'value_trained',True):
                                raise ValueError('当前模型的价值头尚未训练')
                            net=self._match_net
                        else:
                            net=load_value_model(path=self.model_db)
                        self._value_ai=NeuralAI(net)
                    prediction=self._value_ai.evaluate_position(board,turn,history,cancel)
                    name=getattr(self._value_ai.net,'display_name','Opus-5')
                self._value_results.put((token,prediction,name,None))
            except CancelledError:
                pass
            except Exception as error:
                print(f'[局势预测] {error}')
                self._value_results.put((token,None,'','暂无可用的局势预测模型'))
        threading.Thread(target=worker,daemon=True,name='xiangqi-value').start()

    def _poll_value(self):
        while True:
            try: token,prediction,name,error=self._value_results.get_nowait()
            except queue.Empty: break
            if token!=self._value_generation: continue
            self.value_busy=False
            self._value_cancel=None
            self.value_prediction=prediction
            self.value_model_name=name
            self.value_message=error or ''

    def action_hint(self):
        if self.game.is_over() or self._is_ai_turn() or self.mode == 'ai_vs_ai':
            self.message = '请在玩家走棋时使用 AI 提示'
            return
        self._invalidate_hints()
        self.model_dialog, self.model_page = True, 0
        try:
            self.model_choices = list_hint_models(self.model_db)
            self.dialog_message = '' if self.model_choices else '请稍后重试。'
        except Exception as error:
            self.model_choices = []
            self.dialog_message = '模型列表暂不可用，请稍后重试。'
            print(f'[模型库] {error}')

    def _start_hint(self, choice):
        self.model_dialog = False
        self.hint_busy = True
        token = self._hint_generation
        board,turn,history = self.game.board.clone(),self.game.turn,list(self.game.history)
        cancel = self._hint_cancel = threading.Event()
        path,results = self.model_db,self._hint_results
        name = choice['name']
        self.message = f'{name} 正在分析历史局面…'

        def worker():
            try:
                import torch
                torch.set_num_threads(min(4,torch.get_num_threads()))
                net = load_hint_model(name,path=path)
                ranked = NeuralAI(net).rank_moves(board,turn,history=history,cancel_event=cancel)
                results.put((token,turn,ranked,name,None))
            except CancelledError:
                pass
            except ImportError as error:
                print(f'[AI 提示] {error}')
                results.put((token,turn,[],None,'AI 提示暂不可用，请稍后重试。'))
            except Exception as error:
                print(f'[AI 提示] {error}')
                results.put((token,turn,[],None,'模型暂不可用，请稍后重试或选择其他模型。'))
        threading.Thread(target=worker,daemon=True,name='xiangqi-hint').start()

    def _poll_hints(self):
        while True:
            try: token,turn,ranked,name,error = self._hint_results.get_nowait()
            except queue.Empty: break
            if token != self._hint_generation or turn != self.game.turn: continue
            self.hint_busy, self._hint_cancel = False,None
            if error:
                self.message = error
            else:
                legal = set(self.game.all_legal_moves())
                if any(move not in legal for move,_ in ranked):
                    self.message = '提示局面已变化，请重新分析'
                    continue
                self.hints, self.hint_model_name = ranked,name
                self.message = f'{name}：已显示概率最高的 {len(ranked)} 个合法走法，点击查看落点'

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
        self.ai_failed = False
        if self.ai:
            self.ai = self._make_ai()
        self.message = '新局开始，红方先行'
        self._ai_due = pygame.time.get_ticks() + 300
        if self.mode != 'pvp':
            self.restart_requested = True
            self.running = False

    def action_undo(self):
        self._invalidate_ai()
        count = self.game.undo_turn(self.player_color)
        self._clear_selection()
        self._refresh_last_move()
        if self.ai:
            self.ai = self._make_ai()
        self.ai_failed = False
        if self.mode == 'ai_vs_ai':
            self.paused = True
        self.message = f'已悔棋 {count} 步' if count else '暂无可撤销的走子'
        self._ai_due = pygame.time.get_ticks() + 300

    def action_pause(self):
        if self.ai_failed:
            self._invalidate_ai()
            self.ai, self.ai_failed, self.paused = self._make_ai(),False,False
            self.message = f'正在重试 {self.opponent_name}…'
            self._ai_due = pygame.time.get_ticks() + 150
            return
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
                        action = {pygame.K_ESCAPE: self._escape, pygame.K_b: self.action_back,
                                  pygame.K_u: self.action_undo, pygame.K_n: self.action_new,
                                  pygame.K_h: self.action_hint,
                                  pygame.K_SPACE: self.action_pause}.get(event.key)
                        if action:
                            action()
                    if not self.running:
                        break
                if not self.running:
                    break
                self._poll_ai()
                self._poll_hints()
                self._poll_value()
                self._request_value()
                if (self._is_ai_turn() and not self.game.is_over() and not self.paused
                        and not self.ai_failed and not self.thinking and pygame.time.get_ticks() >= self._ai_due):
                    self._ai_move()
                self._draw()
                self.clock.tick(30)
        finally:
            self._invalidate_ai()
        return self.exit_to_menu

    def _ai_move(self):
        if self.thinking or self.paused or self.ai_failed or self.game.is_over() or not self._is_ai_turn():
            return
        self.thinking = True
        token, board, turn = self._generation, self.game.board.clone(), self.game.turn
        history = list(self.game.history)
        cancel = self._cancel_event = threading.Event()
        ai, results = self.ai, self._results
        self.message = f'{"红" if turn == "r" else "黑"}方 AI 思考中…'

        def worker():
            try:
                kwargs = {'history':history} if isinstance(ai,(NeuralAI,SearchAI)) else {}
                move = ai.choose_move(board, turn, cancel_event=cancel, **kwargs)
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
                self.ai_failed = True
                self.message = 'AI 计算失败，请点击“重试AI”或返回菜单'
                print(f'[AI] {error}')
            elif move is None:
                self.game._evaluate_end()
                if not self.game.is_over():
                    self.ai_failed = True
                self.message = self.game.status or 'AI 未返回走法，请点击“重试AI”'
            elif self.game.move(*move) == 'ok':
                self._after_move()
            else:
                self.ai_failed = True
                self.message = 'AI 未能完成走棋，请点击“重试AI”或返回菜单'

    def _after_move(self):
        self._invalidate_value()
        self._invalidate_hints()
        self._clear_selection()
        self._refresh_last_move()
        side = '红' if self.game.turn == 'r' else '黑'
        self.message = self.game.status or (f'将军！{side}方请应将' if self.game.board.is_in_check(self.game.turn)
                                           else f'轮到{side}方走子')
        if not self.game.is_over() and self.game.must_change_move():
            self.message = f'同一循环连续三次出现，{side}方须变招'
        self._ai_due = pygame.time.get_ticks() + (500 if self.mode == 'ai_vs_ai' else 150)

    def _on_click(self, pos):
        if self.model_dialog:
            if pygame.Rect(570,140,70,34).collidepoint(pos):
                self.model_dialog = False
            elif pygame.Rect(132,560,100,34).collidepoint(pos):
                self.model_page = max(0,self.model_page-1)
            elif pygame.Rect(528,560,100,34).collidepoint(pos):
                self.model_page = min(max(0,(len(self.model_choices)-1)//5),self.model_page+1)
            else:
                for rect,choice in self.model_rects:
                    if rect.collidepoint(pos): self._start_hint(choice); break
            return
        for rect, label in self.btn_rects:
            if rect.collidepoint(pos):
                dict(self.buttons)[label]()
                return
        if self.hint_rect.collidepoint(pos):
            self.action_hint()
            return
        if self.game.is_over() or self._is_ai_turn() or self.mode == 'ai_vs_ai':
            return
        for rect,move in self.hint_rows:
            if rect.collidepoint(pos):
                self.selected, self.legal_targets = move[:2],[move[2:]]
                self.message = '已标出建议走法，点击棋盘落点确认走子'
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
            shown_label = '重试AI' if label=='暂停/继续' and self.ai_failed else label
            text_fit(self.screen, shown_label, self.button_font, COLOR_DARK, rect, True)
        if self.model_dialog:
            self._draw_model_dialog()
        pygame.display.flip()

    def _escape(self):
        if self.model_dialog:
            self.model_dialog = False
        else:
            self.action_back()

    def _draw_model_dialog(self):
        shade = pygame.Surface((GAME_W,GAME_H),pygame.SRCALPHA)
        shade.fill((0,0,0,145)); self.screen.blit(shade,(0,0))
        pygame.draw.rect(self.screen,COLOR_LIGHT,(110,124,540,490),border_radius=12)
        text_fit(self.screen,'选择提示模型',self.button_font,COLOR_DARK,pygame.Rect(132,145,400,28))
        text_fit(self.screen,'关闭',self.small_font,COLOR_RED,pygame.Rect(570,143,70,30),True)
        self.model_rects = []
        if not self.model_choices:
            text_fit(self.screen,'暂无可用模型',self.button_font,COLOR_DARK,pygame.Rect(132,210,490,35))
            text_fit(self.screen,self.dialog_message,self.small_font,COLOR_LINE,pygame.Rect(132,260,490,35))
        for i,choice in enumerate(self.model_choices[self.model_page*5:(self.model_page+1)*5]):
            rect = pygame.Rect(132,196+i*69,496,60)
            pygame.draw.rect(self.screen,(232,214,180),rect,border_radius=6)
            text_fit(self.screen,choice['name'],self.button_font,COLOR_DARK,rect,True)
            self.model_rects.append((rect,choice))
        for label,rect in [('上一页',pygame.Rect(132,560,100,34)),('下一页',pygame.Rect(528,560,100,34))]:
            text_fit(self.screen,label,self.small_font,COLOR_DARK,rect,True)
        text_fit(self.screen,f'{self.model_page+1} / {max(1,(len(self.model_choices)+4)//5)}',
                 self.small_font,COLOR_DARK,pygame.Rect(310,560,140,34),True)

    def _draw_sidebar(self):
        pygame.draw.rect(self.screen, (229, 199, 155), (BOARD_W, INFO_H, GAME_W - BOARD_W, BOARD_H))
        lines = ['对局信息', '红先 · 将死 / 困毙判负',
                 '选棋子，再点击绿色落点', '',
                 'U  悔棋      N  新局', 'B / Esc  返回菜单', 'H  AI 提示 · 先选择模型',
                 'AI：' + ('未启用' if self.ai is None else self.opponent_name),
                 '状态：' + ('已结束' if self.game.is_over() else '待重试' if self.ai_failed else '暂停' if self.paused else '思考中' if self.thinking else '进行中'),
                 '']
        for i, line in enumerate(lines):
            text_fit(self.screen, line, self.small_font, COLOR_DARK, pygame.Rect(542, 92 + i * 26, 206, 24))
        pygame.draw.rect(self.screen,COLOR_GOLD,self.hint_rect,border_radius=6)
        text_fit(self.screen,'分析中…' if self.hint_busy else 'AI 提示 · 选择模型',self.button_font,COLOR_DARK,self.hint_rect,True)
        self.hint_rows = []
        if self.hints:
            text_fit(self.screen,f'{self.hint_model_name} · 合法动作 Top 3',self.small_font,COLOR_DARK,pygame.Rect(542,368,206,24))
            for i,(move,prob) in enumerate(self.hints):
                rect = pygame.Rect(540,399+i*49,208,43)
                pygame.draw.rect(self.screen,COLOR_LIGHT,rect,border_radius=5)
                label = f'{i+1}. {move[0]}{move[1]} → {move[2]}{move[3]}   {prob:.1%}'
                text_fit(self.screen,label,self.small_font,COLOR_DARK,rect,True)
                self.hint_rows.append((rect,move))
            text_fit(self.screen,'概率是走法倾向，不是胜率',self.small_font,COLOR_LINE,pygame.Rect(542,550,206,24))
        else:
            text_fit(self.screen,'最近走子',self.small_font,COLOR_DARK,pygame.Rect(542,378,206,24))
        for i, step in enumerate(self.game.history[-(2 if self.hints else 7):]):
            name = (NAMES_RED if step['turn'] == 'r' else NAMES_BLACK)[step['piece'][1]]
            label = f'{"红" if step["turn"] == "r" else "黑"}{name} {step["fr"]}{step["fc"]} → {step["tr"]}{step["tc"]}'
            text_fit(self.screen, label, self.small_font, COLOR_LINE, pygame.Rect(542, (590 if self.hints else 414) + i * 28, 206, 24))
        self._draw_value_panel()

    def _draw_value_panel(self):
        left,width=772,232
        pygame.draw.rect(self.screen,COLOR_LIGHT,(left-10,92,width+20,362),border_radius=9)
        text_fit(self.screen,'当前局势预测',self.button_font,COLOR_DARK,pygame.Rect(left,106,width,28))
        if self.value_prediction is None:
            text_fit(self.screen,self.value_message,self.small_font,COLOR_LINE,pygame.Rect(left,153,width,26))
            text_fit(self.screen,'需要已训练的 Opus 模型',self.small_font,COLOR_LINE,pygame.Rect(left,188,width,26))
            return
        p=self.value_prediction
        text_fit(self.screen,'分析模型：'+self.value_model_name,self.small_font,COLOR_LINE,pygame.Rect(left,144,width,26))
        values=[('红方胜率',p['red_win'],COLOR_RED),('和棋概率',p['draw'],(103,119,101)),
                ('黑方胜率',p['black_win'],COLOR_DARK)]
        for index,(label,prob,color) in enumerate(values):
            y=186+index*39
            text_fit(self.screen,f'{label}  {prob:.1%}',self.button_font,color,pygame.Rect(left,y,width,28))
        x=left
        for label,prob,color in values:
            segment=width*prob
            pygame.draw.rect(self.screen,color,(round(x),313,round(x+segment)-round(x),12))
            x+=segment
        if p['draw']>=max(p['red_win'],p['black_win']): outlook='预测：和棋概率最高'
        elif abs(p['red_win']-p['black_win'])<1e-6: outlook='预测：双方胜率相同'
        else: outlook='预测：'+('红方' if p['red_win']>p['black_win'] else '黑方')+'胜率较高'
        text_fit(self.screen,outlook,self.small_font,COLOR_DARK,pygame.Rect(left,344,width,25))
        text_fit(self.screen,'红方胜率即黑方负率',self.small_font,COLOR_LINE,pygame.Rect(left,382,width,25))
        text_fit(self.screen,'黑方胜率即红方负率',self.small_font,COLOR_LINE,pygame.Rect(left,411,width,25))


def run_app(model_path=MODEL_DB, model_name=None, initial_mode=None, depth=2,*,parameter_family=None,parameter_id=None):
    mode,preferred_name = initial_mode,model_name
    try:
        while True:
            if mode is None:
                mode = MainMenu(model_path=model_path).run()
            if mode is None:
                break
            net = None
            if mode != 'pvp':
                options={} if parameter_family is None else {'parameter_family':parameter_family,'parameter_id':parameter_id}
                selection = OpponentModelMenu(mode,model_path,preferred_name,**options)
                action,net = selection.run()
                if action == 'quit': break
                if action == 'back':
                    mode = None
                    continue
                preferred_name = selection.selected_name
            app = XiangqiUI(mode,net,depth=depth,model_db=model_path)
            back_to_menu = app.run()
            if app.restart_requested: continue
            if not back_to_menu:
                break
            mode = None
    finally:
        pygame.quit()


if __name__ == '__main__':
    run_app()

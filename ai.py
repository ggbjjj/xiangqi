"""AI 封装：内置搜索与 Haiku-5 历史策略推理。"""

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

    def choose_move(self, board, turn, cancel_event=None, history=None):
        from repetition import RepetitionTracker
        tracker = RepetitionTracker.from_history(board, turn, history)
        deadline = monotonic() + self.time_limit

        def check_budget():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()
            if monotonic() >= deadline:
                raise TimeoutError()

        def search(position, color, depth, alpha, beta, ply):
            check_budget()
            moves = tracker.legal_moves(position, color)
            if not moves:
                return -100000 + ply
            if depth == 0:
                return self._evaluate(position, color)
            best = -float('inf')
            other = 'b' if color == 'r' else 'r'
            for move in self._ordered(position, moves):
                after = position.apply_move(*move)
                tracker.push(move, after)
                try:
                    value = -search(after, other, depth - 1, -beta, -alpha, ply + 1)
                finally:
                    tracker.pop()
                best = max(best, value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best

        moves = self._ordered(board, tracker.legal_moves(board, turn))
        if not moves:
            return None
        chosen = moves[0]
        other = 'b' if turn == 'r' else 'r'
        for depth in range(1, self.max_depth + 1):
            best_score, best_move = -float('inf'), chosen
            try:
                for move in moves:
                    after = board.apply_move(*move)
                    tracker.push(move, after)
                    try:
                        value = -search(after, other, depth - 1, -float('inf'), -best_score, 1)
                    finally:
                        tracker.pop()
                    if value > best_score:
                        best_score, best_move = value, move
            except TimeoutError:
                break
            chosen = best_move
            moves.remove(chosen)
            moves.insert(0, chosen)
        return chosen


def load_network(model_path=None, model_name=None, *, family=None,parameter_id=None):
    """Resolve a public model name through the backend. Basic gameplay needs no torch."""
    from pathlib import Path
    from project_paths import MODEL_CATALOG
    from model_store import list_hint_models, load_hint_model, default_model_name
    path = Path(model_path or MODEL_CATALOG)
    if parameter_id is not None and family is None: raise ValueError('按 ID 加载需要指定模型系列 family')
    if family is not None:
        from model_store import load_parameter_model
        net,meta,_=load_parameter_model(family,parameter_id,path=path)
        net.parameter_id=meta['parameter_id']; net.checkpoint_id=meta['checkpoint_id']
        net.completed_epochs=meta['completed_epochs']
        return net
    if not path.exists():
        return None
    if model_name is None:
        choices = list_hint_models(path)
        if not choices: return None
        model_name = default_model_name(path) or choices[0]['name']
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    return load_hint_model(model_name,path=path)


class NeuralAI:
    """Direct legal policy inference. History is replayed with per-layer K/V caches."""
    def __init__(self, net, device='cpu', *, exploration=None, generator=None):
        self.net, self.device = net, device
        self.net.to(device).eval()
        self.exploration=exploration
        self.generator=generator
        if exploration is not None:
            import torch
            from training.exploration import Exploration
            self.exploration=Exploration(exploration.temperature,exploration.epsilon)
            if self.generator is None:
                self.generator=torch.Generator(device='cpu')
                self.generator.seed()
        self.reset()

    def reset(self):
        self._keys, self._cache, self._logits = [], None, None
        self._value_logits = None

    def _forward_position(self, board, turn, history=None, cancel_event=None):
        import torch
        from board import Board
        from encoder import encode_state
        positions = [(Board(h['board_state']),h['prev_turn']) for h in (history or [])]+[(board,turn)]
        keys = [(b.signature(),color) for b,color in positions]
        if keys[:len(self._keys)] != self._keys:
            self.reset()
        with torch.inference_mode():
            for b,color in positions[len(self._keys):]:
                if cancel_event is not None and cancel_event.is_set():
                    self.reset()
                    raise CancelledError()
                x = torch.from_numpy(encode_state(b,color))[None].to(self.device)
                try:
                    output,self._cache = self.net.forward_step(x,self._cache)
                    self._logits = getattr(output,'policy_logits',output)
                    self._value_logits = getattr(output,'value_logits',None)
                except Exception:
                    self.reset()
                    raise
                self._keys.append((b.signature(),color))
            self._keys = keys
        if cancel_event is not None and cancel_event.is_set():
            self.reset()
            raise CancelledError()

    def evaluate_position(self, board, turn, history=None, cancel_event=None):
        """One WDL prediction, remapped from side-to-move to fixed red/black sides."""
        import torch
        self._forward_position(board,turn,history,cancel_event)
        if self._value_logits is None: raise ValueError('当前模型没有价值头')
        with torch.inference_mode():
            logits=self._value_logits.float()
            if logits.shape!=(1,3) or not torch.isfinite(logits).all():
                raise ValueError('价值预测无效')
            win,draw,loss=map(float,logits.softmax(-1)[0])
        red,black=(win,loss) if turn=='r' else (loss,win)
        return {'red_win':red,'draw':draw,'black_win':black,'turn':turn}

    def rank_moves(self, board, turn, history=None, topk=3, cancel_event=None):
        import torch
        from encoder import legal_action_mask,action_to_move
        from models.HaiKu.loss import masked_probabilities
        self._forward_position(board,turn,history,cancel_event)
        with torch.inference_mode():
            mask = torch.from_numpy(legal_action_mask(board,turn,history))[None].to(self.device)
            count = int(mask.sum())
            if count == 0: return []
            probs = masked_probabilities(self._logits,mask)[0]
            # Top-k is restricted to legal IDs even if valid probabilities underflow to 0.
            ids = mask[0].nonzero().flatten()
            order = torch.argsort(probs[ids],descending=True,stable=True)[:min(topk,count)]
            if cancel_event is not None and cancel_event.is_set(): raise CancelledError()
            return [(action_to_move(int(ids[i])),float(probs[ids[i]])) for i in order]

    def choose_move(self, board, turn, cancel_event=None, history=None):
        if self.exploration is not None:
            import torch
            from encoder import legal_action_mask,action_to_move
            from training.exploration import sample_action
            self._forward_position(board,turn,history,cancel_event)
            with torch.inference_mode():
                legal=torch.from_numpy(legal_action_mask(board,turn,history)).to(self.device)
                if not legal.any(): return None
                action,_=sample_action(self._logits[0],legal,self.exploration,self.generator)
            if cancel_event is not None and cancel_event.is_set(): raise CancelledError()
            return action_to_move(action)
        ranked = self.rank_moves(board,turn,history=history,topk=1,cancel_event=cancel_event)
        return ranked[0][0] if ranked else None

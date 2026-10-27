"""Deterministic Chinese/ICCS notation decoding. Ambiguity is an error."""
import re
import unicodedata

from board import Board
from piece import Piece
from encoder import move_to_action

STANDARD_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
FEN_KINDS = dict(zip('kabnrcphe', 'KAEHRCPHE'))
KINDS = {'帅':'K','将':'K','士':'A','仕':'A','象':'E','相':'E',
         '马':'H','车':'R','炮':'C','兵':'P','卒':'P'}
TRANS = str.maketrans({'車':'车','俥':'车','馬':'马','傌':'马','帥':'帅','將':'将',
    '進':'进','後':'后','砲':'炮', **dict(zip('一二三四五六七八九','123456789'))})


def from_fen(fen):
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in ('w', 'r', 'b'):
        raise ValueError('FEN 缺少有效走方')
    rows = fields[0].split('/')
    if len(rows) != 10:
        raise ValueError('FEN 必须有十行')
    grid = []
    for row in rows:
        line = []
        for c in row:
            if c in '123456789':
                line.extend([None] * int(c))
            elif c.lower() in FEN_KINDS:
                line.append(Piece('r' if c.isupper() else 'b', FEN_KINDS[c.lower()]))
            else:
                raise ValueError(f'FEN 未知棋子 {c}')
        if len(line) != 9:
            raise ValueError('FEN 每行必须九格')
        grid.append(line)
    for color in ('r', 'b'):
        if sum(p is not None and p.kind == 'K' and p.color == color for row in grid for p in row) != 1:
            raise ValueError('初始局面须各有一个将帅')
    return Board(grid), 'r' if fields[1] in ('w', 'r') else 'b'


def to_fen(board, turn):
    letters = dict(zip('KAEHRCP', 'kabnrcp'))
    rows = []
    for row in board.grid:
        out, empty = '', 0
        for p in row:
            if p is None:
                empty += 1
                continue
            if empty:
                out += str(empty)
                empty = 0
            s = letters[p.kind]
            out += s.upper() if p.color == 'r' else s
        if empty:
            out += str(empty)
        rows.append(out)
    return '/'.join(rows) + (' w' if turn == 'r' else ' b') + ' - - 0 1'


def parse_move(board, turn, text):
    text = unicodedata.normalize('NFKC', text).translate(TRANS)
    if re.fullmatch(r'[a-iA-I][0-9][-:]?[a-iA-I][0-9]', text):
        s = re.sub('[-:]', '', text).lower()
        candidates = [(9-int(s[1]), ord(s[0])-97, 9-int(s[3]), ord(s[2])-97)]
    else:
        if len(text) != 4 or text[2] not in '进退平' or text[3] not in '123456789':
            raise ValueError(f'不支持的记谱 {text}')
        prefix, arg, direction, number = text
        number = int(number)
        kind = KINDS.get(prefix if prefix in KINDS else arg)
        if kind is None:
            raise ValueError(f'未知棋子 {text}')
        sources = [(r,c) for r,row in enumerate(board.grid) for c,p in enumerate(row)
                   if p and p.color == turn and p.kind == kind]
        if prefix in KINDS and arg in '123456789':
            col = 9-int(arg) if turn == 'r' else int(arg)-1
            sources = [(r,c) for r,c in sources if c == col]
        elif prefix in '前后中':
            selected = []
            for c in range(9):
                same = sorted((r,c) for r,col in sources if col == c)
                if turn == 'b':
                    same.reverse()
                if len(same) >= 2:
                    if prefix == '前': selected.append(same[0])
                    elif prefix == '后': selected.append(same[-1])
                    elif len(same) == 3: selected.append(same[1])
            sources = selected
        else:
            raise ValueError(f'不支持的源位置 {text}')
        candidates = []
        target_col = 9-number if turn == 'r' else number-1
        forward = -1 if turn == 'r' else 1
        for r,c in sources:
            for tr,tc in board.pseudo_legal_moves(r,c):
                if direction == '平':
                    matches = tr == r and tc == target_col
                else:
                    delta = (tr-r) * forward * (1 if direction == '进' else -1)
                    matches = delta > 0 and (tc == target_col if kind in 'HEA' else tc == c and delta == number)
                if matches:
                    candidates.append((r,c,tr,tc))
    legal = []
    for fr,fc,tr,tc in candidates:
        p, target = board.piece_at(fr,fc), board.piece_at(tr,tc)
        if (p and p.color == turn and (tr,tc) in board.pseudo_legal_moves(fr,fc)
                and not (target and target.kind == 'K')
                and not board.apply_move(fr,fc,tr,tc).is_in_check(turn)):
            legal.append((fr,fc,tr,tc))
    if len(legal) != 1:
        raise ValueError(f'{text} 对应 {len(legal)} 个合法走法')
    return legal[0]


def validate_candidate(record):
    """Process-pool safe: validate every recorded ply, never retain a bad prefix."""
    actions = []
    try:
        board, turn = from_fen(record['start_fen'])
        fen = to_fen(board, turn)
        for t, text in enumerate(record['moves_zh_normalized']):
            try:
                move = parse_move(board, turn, text)
            except ValueError as error:
                raise ValueError(f'ply={t+1}: {error}') from error
            actions.append(move_to_action(move))
            board = board.apply_move(*move)
            turn = 'b' if turn == 'r' else 'r'
        if not actions:
            raise ValueError('空棋谱')
        return record, fen, actions, None
    except (ValueError, KeyError) as error:
        return record, '', [], str(error)

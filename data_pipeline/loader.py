"""Read full book games as consecutive T-ply samples, padding only the tail."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .database import connect
from .notation import from_fen,to_fen
from encoder import encode_state, action_to_move, legal_action_mask
from project_paths import DATA_DB,ENDGAME_DB
from repetition import RepetitionTracker


BOOK_CROP_VERSION = 'contiguous-book-crops-v1'
BOOK_READER_VERSION = 'multi-book-sources-v1'
BOOK_SOURCE_LABELS = {'main':'主棋谱库', 'endgames':'残局库', 'both':'主棋谱库＋残局库'}


def book_index(source, db, endgame_db):
    """Select both libraries before split/limit/cropping; main split wins ties."""
    if source not in BOOK_SOURCE_LABELS: raise ValueError('棋谱来源必须为 main、endgames 或 both')
    paths = ({'main':Path(db)} if source=='main' else {'endgames':Path(endgame_db)}
             if source=='endgames' else {'main':Path(db),'endgames':Path(endgame_db)})
    if source=='both' and paths['main'].resolve()==paths['endgames'].resolve():
        raise ValueError('两库合读须指定不同的主棋谱库和残局库')
    games, main_groups = {}, {}
    for kind,path in paths.items():
        conn = connect(path,readonly=True)
        try:
            rows = conn.execute('SELECT game_id,ply_count,split,split_group,result,metadata_json FROM games').fetchall()
        finally: conn.close()
        for row in rows:
            key = row['game_id']
            if kind=='main':
                if row['split_group'] in main_groups and main_groups[row['split_group']]!=row['split']:
                    raise ValueError('主棋谱库 split_group 跨越多个划分')
                main_groups[row['split_group']] = row['split']
            if key in games:
                old = games[key]
                def content(location):
                    conn = connect(location,readonly=True)
                    try: return conn.execute('SELECT start_fen,moves_json FROM games WHERE game_id=?',(key,)).fetchone()
                    finally: conn.close()
                a,b = content(old['db']),content(path)
                if a['start_fen']!=b['start_fen'] or json.loads(a['moves_json'])!=json.loads(b['moves_json']):
                    raise ValueError(f'两库相同棋局 ID 的局面或走法不一致：{key}')
                old['origins'].append(kind)
                old['labels'].add(row['result'].strip() if isinstance(row['result'],str) else row['result'])
                old['conflict'] |= bool(json.loads(row['metadata_json']).get('result_conflict'))
                continue
            games[key] = {'db':path,'kind':kind,'plies':row['ply_count'],
                'split':main_groups.get(row['split_group'],row['split']) if kind=='endgames' else row['split'],
                'origins':[kind], 'labels':{row['result'].strip() if isinstance(row['result'],str) else row['result']},
                'conflict':bool(json.loads(row['metadata_json']).get('result_conflict'))}
    return paths,games


@dataclass
class BookSegment:
    x: torch.Tensor
    actions: torch.Tensor
    legal: torch.Tensor
    length: int
    value_targets: torch.Tensor
    start_fen: str | None
    result: str | None
    source_game_id: str
    start_ply: int


def result_target(result, turn):
    """Map the original full-game result to the side-to-move W/D/L label."""
    if turn not in ('r', 'b'): raise ValueError('未知行棋方')
    if isinstance(result,str): result = result.strip()
    if result in (None, '', '*'): return -100
    if result == '1/2-1/2': return 1
    if result not in ('1-0', '0-1'): raise ValueError('未知棋局结果')
    return 0 if ('r' if result == '1-0' else 'b') == turn else 2


class GameDataset(Dataset):
    """Keep source IDs/splits intact; expose ceil(K/T) segments for each game.

    Replay all preceding moves on the CPU to restore the board and repetition
    history. Neural state starts fresh for each segment. Outcome supervision
    still checks the complete game for compatibility with the project rules.
    """
    def __init__(self, T=120, split='train', db=DATA_DB, limit=0, *, include_value=False,
                 source='main',endgame_db=ENDGAME_DB):
        if type(T) is not int or T < 1: raise ValueError('T 必须为正整数')
        if type(limit) is not int or limit < 0: raise ValueError('limit 必须为非负整数')
        if split not in ('train','validation','test'): raise ValueError('未知数据划分')
        self.T, self.db, self.include_value = T, db, include_value
        paths,games = book_index(source,db,endgame_db)
        self.ids = sorted(key for key,row in games.items() if row['split']==split)
        if limit: self.ids = self.ids[:limit]
        self._ply_counts = {key:games[key]['plies'] for key in self.ids}
        self._game_dbs = {key:games[key]['db'] for key in self.ids}
        self._result_overrides = {}
        counts = {kind:{'games':0,'segments':0} for kind in paths}
        duplicates = conflicts = 0
        for key in self.ids:
            row = games[key]
            counts[row['kind']]['games'] += 1
            counts[row['kind']]['segments'] += (row['plies']+T-1)//T
            duplicates += len(row['origins'])-1
            if len(row['origins'])>1:
                known = row['labels']-{'*','',None}
                if not known<={'1-0','0-1','1/2-1/2'}: raise ValueError('未知棋局结果')
                conflict = row['conflict'] or len(known)>1
                self._result_overrides[key] = '*' if conflict or not known else next(iter(known))
                conflicts += int(conflict)
        self.segments = [(key,start,min(start+T,self._ply_counts[key]))
                         for key in self.ids for start in range(0,self._ply_counts[key],T)]
        self.summary = {'mode':source,'label':BOOK_SOURCE_LABELS[source],
            'databases':{kind:str(path.resolve()) for kind,path in paths.items()},
            'games':len(self.ids),'segments':len(self.segments),'by_source':counts,
            'duplicate_games':duplicates,'value_conflicts':conflicts,'reader_version':BOOK_READER_VERSION}

    def __len__(self): return len(self.segments)

    def __getitem__(self, index):
        segment = self.read_segment(index)
        sample = segment.x,segment.actions,segment.legal,segment.length
        return (*sample,segment.value_targets) if self.include_value else sample

    def read_segment(self, index):
        """Return tensors and the actual crop origin without changing the DB."""
        game_id,start,stop = self.segments[index]
        conn = connect(self._game_dbs[game_id],readonly=True)
        try:
            row = conn.execute('SELECT start_fen,moves_json,result,ply_count FROM games WHERE game_id=?',(game_id,)).fetchone()
        finally: conn.close()
        if row is None: raise ValueError(f'数据库棋局已删除：{game_id}')
        result = self._result_overrides.get(game_id,row['result'])
        all_actions = json.loads(row['moves_json'])
        if len(all_actions)!=row['ply_count'] or row['ply_count']!=self._ply_counts[game_id]:
            raise ValueError(f'数据库棋局长度在建立裁剪索引后发生变化：{game_id}')
        actions = all_actions if self.include_value else all_actions[:stop]
        length = 0
        board, turn = from_fen(row['start_fen'])
        tracker = RepetitionTracker(board,turn)
        start_fen = None
        x = np.zeros((self.T,15,10,9),dtype=np.float32)
        y = np.full(self.T,-100,dtype=np.int64)
        value = np.full(self.T,-100,dtype=np.int64)
        legal = np.zeros((self.T,8100),dtype=bool)
        for ply, action in enumerate(actions):
            if type(action) is not int or not 0<=action<8100:
                raise ValueError(f'数据库棋谱损坏：{game_id}, ply={ply+1}')
            if ply==start: start_fen = to_fen(board,turn)
            move = action_to_move(action)
            if move in tracker.forbidden_moves():
                # Preserve the source record. A newly forbidden book continuation
                # is not a valid policy target or evidence for a WDL label.
                value.fill(-100)
                break
            if start<=ply<stop:
                t = ply-start
                x[t] = encode_state(board,turn)
                legal[t] = legal_action_mask(board,turn,repetition=tracker)
                if not legal[t,action]: raise ValueError(f'数据库棋谱损坏：{game_id}, ply={ply+1}')
                y[t] = action
                if self.include_value: value[t] = result_target(result, turn)
                length += 1
            else:
                # Prefix moves must be legal too: an invalid source history
                # cannot be used to manufacture a plausible later crop.
                fr,fc,tr,tc = move
                piece = board.piece_at(fr,fc)
                if piece is None or piece.color!=turn or (tr,tc) not in board.legal_moves(fr,fc):
                    raise ValueError(f'数据库棋谱损坏：{game_id}, ply={ply+1}')
            board = board.apply_move(*move)
            tracker.push(move,board)
            turn = 'b' if turn == 'r' else 'r'
        else:
            # Old records may stop at a threefold draw which is no longer a
            # terminal under this project rule. Do not reuse that WDL target.
            if self.include_value and tracker.forbidden_moves(): value.fill(-100)
        return BookSegment(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(legal),length,
                           torch.from_numpy(value),start_fen,result,game_id,start)


def collate_games(samples):
    x,y,legal,lengths = zip(*samples)
    return torch.stack(x,1),torch.stack(y,1),torch.stack(legal,1),torch.tensor(lengths,dtype=torch.long)


def iter_game_batches(T, batch_size=128, split='train', shuffle=True, drop_last=False,
                      seed=42, db=DATA_DB, num_workers=0, one_hot=True, include_legal=False,
                      source='main',endgame_db=ENDGAME_DB):
    """Yield consecutive book segments: X[T,B,15,10,9], Y, lengths[B].

    Training can request compact int64 labels (equivalent one-hot CE).
    A K-ply source contributes ceil(K/T) samples; only its last one is padded.
    The final batch may have B_cur < batch_size unless drop_last=True.
    """
    loader = DataLoader(GameDataset(T,split,db,source=source,endgame_db=endgame_db),batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,
        num_workers=num_workers,collate_fn=collate_games,generator=torch.Generator().manual_seed(seed))
    for x,y,legal,lengths in loader:
        if one_hot:
            dense = torch.zeros((*y.shape,8100),dtype=torch.float32)
            dense.scatter_(-1,y.clamp_min(0)[...,None],1)
            y = dense.masked_fill((y<0)[...,None],0)
        yield (x,y,legal,lengths) if include_legal else (x,y,lengths)

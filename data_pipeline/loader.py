"""Whole-game batches; T counts individual red/black plies, not rounds."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .database import connect
from .notation import from_fen
from encoder import encode_state, action_to_move, legal_action_mask
from project_paths import DATA_DB


class GameDataset(Dataset):
    def __init__(self, T=120, split='train', db=DATA_DB, limit=0):
        if T < 1: raise ValueError('T 必须大于 0')
        if split not in ('train','validation','test'): raise ValueError('未知数据划分')
        self.T, self.db = T, db
        conn = connect(db,readonly=True)
        try:
            self.ids = [r[0] for r in conn.execute('SELECT game_id FROM games WHERE split=? ORDER BY game_id',(split,))]
        finally: conn.close()
        if limit: self.ids = self.ids[:limit]

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        conn = connect(self.db,readonly=True)
        try:
            row = conn.execute('SELECT start_fen,moves_json FROM games WHERE game_id=?',(self.ids[index],)).fetchone()
        finally: conn.close()
        actions = json.loads(row['moves_json'])[:self.T]
        length = len(actions)
        board, turn = from_fen(row['start_fen'])
        x = np.zeros((self.T,15,10,9),dtype=np.float32)
        y = np.full(self.T,-100,dtype=np.int64)
        legal = np.zeros((self.T,8100),dtype=bool)
        for t, action in enumerate(actions):
            x[t] = encode_state(board,turn)
            legal[t] = legal_action_mask(board,turn)
            if not legal[t,action]: raise ValueError(f'数据库棋谱损坏：{self.ids[index]}, ply={t+1}')
            y[t] = action
            board = board.apply_move(*action_to_move(action))
            turn = 'b' if turn == 'r' else 'r'
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(legal),length


def collate_games(samples):
    x,y,legal,lengths = zip(*samples)
    return torch.stack(x,1),torch.stack(y,1),torch.stack(legal,1),torch.tensor(lengths,dtype=torch.long)


def iter_game_batches(T, batch_size=128, split='train', shuffle=True, drop_last=False,
                      seed=42, db=DATA_DB, num_workers=0, one_hot=True, include_legal=False):
    """Default: yield X[T,B,15,10,9], one-hot Y[T,B,8100], lengths[B].

    Training can request compact int64 labels (equivalent one-hot CE).
    The final batch may have B_cur < batch_size unless drop_last=True.
    """
    loader = DataLoader(GameDataset(T,split,db),batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,
        num_workers=num_workers,collate_fn=collate_games,generator=torch.Generator().manual_seed(seed))
    for x,y,legal,lengths in loader:
        if one_hot:
            dense = torch.zeros((*y.shape,8100),dtype=torch.float32)
            dense.scatter_(-1,y.clamp_min(0)[...,None],1)
            y = dense.masked_fill((y<0)[...,None],0)
        yield (x,y,legal,lengths) if include_legal else (x,y,lengths)

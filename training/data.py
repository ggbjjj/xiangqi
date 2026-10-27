"""Reconstruct complete histories with current weights; never reuse stored KV."""
from contextlib import closing
from dataclasses import dataclass
import json
import torch
from torch.utils.data import Dataset
from data_pipeline.loader import GameDataset
from data_pipeline.notation import from_fen
from encoder import encode_state,legal_action_mask,action_to_move
from project_paths import model_family
from selfplay.store import connect,load_game
from .loss import LossBatch


def result_target(result,turn):
    if result=='*': return -100
    if result=='1/2-1/2': return 1
    if result not in ('1-0','0-1'): raise ValueError('未知棋局结果')
    winner='r' if result=='1-0' else 'b'
    return 0 if winner==turn else 2


@dataclass
class Example:
    x: torch.Tensor
    actions: torch.Tensor
    legal: torch.Tensor
    book_mask: torch.Tensor
    selfplay_mask: torch.Tensor
    value_targets: torch.Tensor
    old_log_mu: torch.Tensor
    old_value: torch.Tensor
    temperature: torch.Tensor
    epsilon: torch.Tensor
    provenance: tuple | None = None   # checkpoint_id, sha256, architecture
    length: int | None = None


@dataclass
class TrainingBatch:
    x: torch.Tensor
    loss: LossBatch
    provenance: frozenset = frozenset()


class BookDataset(Dataset):
    def __init__(self,T=120,split='train',db=None,limit=0):
        kwargs={} if db is None else {'db':db}
        self.dataset=GameDataset(T,split,limit=limit,**kwargs)

    def __len__(self): return len(self.dataset)

    def __getitem__(self,index):
        x,actions,legal,n=self.dataset[index]
        # Existing books carry next-action labels only; do not invent results.
        valid=torch.arange(len(x))<n
        return Example(x,actions,legal,valid,torch.zeros_like(valid),
                       torch.full((len(x),),-100,dtype=torch.long),
                       torch.full((len(x),),torch.nan),torch.full((len(x),),torch.nan),
                       torch.ones(len(x)),torch.zeros(len(x)),length=n)


class SelfPlayDataset(Dataset):
    def __init__(self,path,run_id,*,family=None,max_generated_plies=None,T=None):
        if not run_id: raise ValueError('必须指定本轮自弈 run_id，不能将所有历史版本混入 PPO')
        if max_generated_plies is not None and max_generated_plies<1: raise ValueError('最大读取步数必须为正')
        if T is not None and (type(T) is not int or T<1): raise ValueError('T 必须为正整数')
        self.path,self.run_id,self.max_generated_plies=path,run_id,max_generated_plies
        self.T=T
        with closing(connect(path,True)) as conn:
            row=conn.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
            if row is None: raise ValueError('自弈 run_id 不存在')
            self.run=dict(row)
            if family is not None and self.run['family']!=model_family(family): raise ValueError('自弈数据模型系列不匹配')
            self.ids=[r[0] for r in conn.execute('SELECT game_id FROM games WHERE run_id=? ORDER BY created_at,game_id',(run_id,))]

    def __len__(self): return len(self.ids)

    def __getitem__(self,index):
        game,run,steps=load_game(self.path,self.ids[index])
        prefix=json.loads(game['prefix_json']); generated=json.loads(game['moves_json'])
        if len(steps)!=game['ply_count'] or [s['action'] for s in steps]!=generated or any(s['ply']!=t for t,s in enumerate(steps)):
            raise ValueError('自弈动作与逐步记录不一致')
        if self.max_generated_plies is not None:
            generated=generated[:self.max_generated_plies]; steps=steps[:self.max_generated_plies]
        actions=prefix+generated; warm=len(prefix)
        if self.T is not None:
            if warm>=self.T: raise ValueError('T 不大于暖启动前缀长度，没有可训练的新步；请增大 T')
            actions=actions[:self.T]
        n=len(actions); storage=self.T or n
        board,turn=from_fen(game['start_fen'])
        x=torch.zeros(storage,15,10,9); legal=torch.zeros(storage,8100,dtype=torch.bool)
        y=torch.full((storage,),-100,dtype=torch.long); y[:n]=torch.tensor(actions,dtype=torch.long)
        own=(torch.arange(storage)>=warm)&(torch.arange(storage)<n)
        value_targets=torch.full((storage,),-100,dtype=torch.long)
        old_log=torch.full((storage,),torch.nan); old_value=torch.full((storage,),torch.nan)
        tau=torch.ones(storage); epsilon=torch.zeros(storage)
        for t,action in enumerate(actions):
            x[t]=torch.from_numpy(encode_state(board,turn))
            legal[t]=torch.from_numpy(legal_action_mask(board,turn))
            if not 0<=action<8100 or not legal[t,action]: raise ValueError('自弈棋谱包含非法动作')
            if t>=warm:
                s=steps[t-warm]
                if s['turn']!=turn: raise ValueError('自弈记录的行棋方与重放不一致')
                value_targets[t]=result_target(game['result'],turn)
                old_log[t]=s['old_log_mu']; old_value[t]=s['old_value']
                tau[t]=s['temperature']; epsilon[t]=s['epsilon']
            board=board.apply_move(*action_to_move(action)); turn='b' if turn=='r' else 'r'
        return Example(x,y,legal,torch.zeros(storage,dtype=torch.bool),own,value_targets,old_log,
                       old_value,tau,epsilon,(run['checkpoint_id'],run['weights_sha256'],run['architecture']),n)


def collate_examples(samples):
    if not samples: raise ValueError('空训练批次')
    t=max(len(s.x) for s in samples); b=len(samples)
    x=torch.zeros(t,b,15,10,9); actions=torch.full((t,b),-100,dtype=torch.long)
    legal=torch.zeros(t,b,8100,dtype=torch.bool)
    book=torch.zeros(t,b,dtype=torch.bool); own=torch.zeros_like(book)
    value=torch.full((t,b),-100,dtype=torch.long)
    old_log=torch.full((t,b),torch.nan); old_value=torch.full((t,b),torch.nan)
    tau=torch.ones(t,b); epsilon=torch.zeros(t,b)
    lengths=[]; provenance=set()
    for i,s in enumerate(samples):
        n=len(s.x); lengths.append(n if s.length is None else s.length)
        for destination,name in ((x,'x'),(actions,'actions'),(legal,'legal'),(book,'book_mask'),
            (own,'selfplay_mask'),(value,'value_targets'),(old_log,'old_log_mu'),
            (old_value,'old_value'),(tau,'temperature'),(epsilon,'epsilon')):
            destination[:n,i]=getattr(s,name)
        if s.provenance: provenance.add(s.provenance)
    loss=LossBatch(actions,legal,torch.tensor(lengths),book,own,value,
                   old_log_mu=old_log,old_value=old_value,temperature=tau,epsilon=epsilon)
    return TrainingBatch(x,loss,frozenset(provenance))

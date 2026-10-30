"""Read fixed-size book and self-play crops; never reuse stored KV."""
from contextlib import closing
from dataclasses import dataclass
import json
import torch
from torch.utils.data import Dataset
from data_pipeline.loader import GameDataset,result_target
from data_pipeline.notation import from_fen,to_fen
from encoder import encode_state,legal_action_mask,action_to_move
from project_paths import model_family
from selfplay.store import connect,load_game
from .loss import LossBatch
from repetition import RepetitionTracker,REPETITION_POLICY


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
    start_fen: str | None = None
    result: str | None = None
    source_game_id: str | None = None
    start_ply: int = 0              # offset within source book/generated moves


@dataclass
class TrainingBatch:
    x: torch.Tensor
    loss: LossBatch
    provenance: frozenset = frozenset()


class BookDataset(Dataset):
    def __init__(self,T=120,split='train',db=None,limit=0,*,source='main',endgame_db=None):
        kwargs={} if db is None else {'db':db}
        if endgame_db is not None: kwargs['endgame_db']=endgame_db
        self.dataset=GameDataset(T,split,limit=limit,include_value=True,source=source,**kwargs)

    @property
    def ids(self): return self.dataset.ids

    @property
    def summary(self): return self.dataset.summary

    def __len__(self): return len(self.dataset)

    def __getitem__(self,index):
        segment=self.dataset.read_segment(index)
        x,actions,legal,n,value=segment.x,segment.actions,segment.legal,segment.length,segment.value_targets
        valid=torch.arange(len(x))<n
        return Example(x,actions,legal,valid,torch.zeros_like(valid),
                       value,
                       torch.full((len(x),),torch.nan),torch.full((len(x),),torch.nan),
                       torch.ones(len(x)),torch.zeros(len(x)),length=n,
                       start_fen=segment.start_fen,result=segment.result,
                       source_game_id=segment.source_game_id,start_ply=segment.start_ply)


class SelfPlayDataset(Dataset):
    """Split generated moves into C-ply samples without changing source games.

    Replay the prefix and earlier moves on the CPU to restore each crop's
    board, side to move and repetition history. Only the crop enters the
    model, with fresh KV/time state and the original full-game result.
    """
    def __init__(self,path,run_id,*,family=None,max_generated_plies=None,C=128):
        if not run_id: raise ValueError('必须指定本轮自弈 run_id，不能将所有历史版本混入 PPO')
        if max_generated_plies is not None and (type(max_generated_plies) is not int or max_generated_plies<1):
            raise ValueError('最大读取步数必须为正整数')
        if type(C) is not int or C<1: raise ValueError('C 必须为正整数')
        from selfplay.runs import resolve_run_id
        run_id=resolve_run_id(path,run_id,family=family)
        self.path,self.run_id,self.max_generated_plies=path,run_id,max_generated_plies
        self.C=C
        with closing(connect(path,True)) as conn:
            row=conn.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
            if row is None: raise ValueError('自弈 run_id 不存在')
            self.run=dict(row)
            self.rules=json.loads(self.run['rules_json'])
            policy=self.rules.get('repetition_policy')
            if policy not in (None,REPETITION_POLICY): raise ValueError('不支持该自弈run的重复判定规则版本')
            self.use_repetition=policy==REPETITION_POLICY
            if family is not None and self.run['family']!=model_family(family): raise ValueError('自弈数据模型系列不匹配')
            games=conn.execute('SELECT game_id,ply_count FROM games WHERE run_id=? ORDER BY created_at,game_id',(run_id,)).fetchall()
        self.ids=[r['game_id'] for r in games]
        self.segments=[]
        for game in games:
            n=game['ply_count']
            if n<1: raise ValueError('自弈棋局没有新生成动作')
            if max_generated_plies is not None: n=min(n,max_generated_plies)
            self.segments.extend((game['game_id'],start,min(start+C,n)) for start in range(0,n,C))

    def __len__(self): return len(self.segments)

    def __getitem__(self,index):
        game_id,start,stop=self.segments[index]
        game,run,steps=load_game(self.path,game_id)
        prefix=json.loads(game['prefix_json']); generated=json.loads(game['moves_json'])
        if len(steps)!=game['ply_count'] or [s['action'] for s in steps]!=generated or any(s['ply']!=t for t,s in enumerate(steps)):
            raise ValueError('自弈动作与逐步记录不一致')
        if stop>len(generated): raise ValueError('自弈棋局长度在建立裁剪索引后发生变化')
        actions=prefix+generated[:stop]; warm=len(prefix)+start
        n=stop-start; storage=self.C
        board,turn=from_fen(game['start_fen'])
        tracker=RepetitionTracker(board,turn,int(self.rules.get('repetition',3))) if self.use_repetition else None
        x=torch.zeros(storage,15,10,9); legal=torch.zeros(storage,8100,dtype=torch.bool)
        y=torch.full((storage,),-100,dtype=torch.long)
        own=torch.arange(storage)<n
        value_targets=torch.full((storage,),-100,dtype=torch.long)
        old_log=torch.full((storage,),torch.nan); old_value=torch.full((storage,),torch.nan)
        tau=torch.ones(storage); epsilon=torch.zeros(storage)
        start_fen=None
        for ply,action in enumerate(actions):
            mask=legal_action_mask(board,turn,repetition=tracker)
            if type(action) is not int or not 0<=action<8100 or not mask[action]: raise ValueError('自弈棋谱包含非法动作')
            if ply>=len(prefix):
                s=steps[ply-len(prefix)]
                if s['turn']!=turn: raise ValueError('自弈记录的行棋方与重放不一致')
            if ply>=warm:
                t=ply-warm
                if t==0: start_fen=to_fen(board,turn)
                x[t]=torch.from_numpy(encode_state(board,turn))
                legal[t]=torch.from_numpy(mask)
                y[t]=action
                value_targets[t]=result_target(game['result'],turn)
                old_log[t]=s['old_log_mu']; old_value[t]=s['old_value']
                tau[t]=s['temperature']; epsilon[t]=s['epsilon']
            move=action_to_move(action)
            board=board.apply_move(*move)
            if tracker is not None: tracker.push(move,board)
            turn='b' if turn=='r' else 'r'
        return Example(x,y,legal,torch.zeros(storage,dtype=torch.bool),own,value_targets,old_log,
                       old_value,tau,epsilon,(run['checkpoint_id'],run['weights_sha256'],run['architecture']),n,
                       start_fen=start_fen,result=game['result'],source_game_id=game_id,start_ply=start)


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

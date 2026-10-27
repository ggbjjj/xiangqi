"""Freeze an explicitly named checkpoint; sample both colors without MCTS or engines."""
from collections import Counter
from dataclasses import dataclass
import copy
import torch
from encoder import encode_state,action_to_move,legal_action_mask
from data_pipeline.notation import STANDARD_FEN,from_fen,to_fen
from model_store import load_version
from training.exploration import Exploration,sample_actions
from training.loss import split_output
from .store import Snapshot,create_run,save_games,finish_run,committed_game_ids
from training.control import WorkInterrupted,check_stop


def generation_micro_batch(save_every,micro_batch=None):
    if type(save_every) is not int or save_every<1: raise ValueError('save_every 必须为正整数')
    if micro_batch is None:
        micro_batch=max(n for n in (1,2,4,8,16) if n<=save_every and save_every%n==0)
    if type(micro_batch) is not int or micro_batch<1 or micro_batch&(micro_batch-1):
        raise ValueError('生成微批量必须是 1、2、4、8、16、32 等 2 的幂')
    if save_every%micro_batch: raise ValueError('生成微批量必须整除保存局数 K，且不能大于 K')
    return micro_batch


@dataclass(frozen=True)
class SelfPlayRules:
    max_plies: int = 1000             # newly generated moves, red+black combined
    quiet_plies: int = 120
    repetition: int = 3
    draw_no_attackers: bool = True

    def __post_init__(self):
        if any(type(n) is not int or n<1 for n in (self.max_plies,self.quiet_plies)):
            raise ValueError('总步数上限与连续未吃子上限须为正整数')
        if type(self.repetition) is not int or self.repetition<2: raise ValueError('重复和棋次数须至少为 2')


class Position:
    """Explicit terminal reasons; same legal move engine as the desktop game."""
    def __init__(self,fen,rules):
        self.board,self.turn=from_fen(fen); self.rules=rules
        fields=fen.split()
        self.quiet=int(fields[4]) if len(fields)>4 and fields[4].isdigit() else 0
        self.counts=Counter({(self.board.signature(),self.turn):1})

    def terminal(self,generated):
        legal=legal_action_mask(self.board,self.turn)
        if not legal.any():
            return ('0-1' if self.turn=='r' else '1-0',
                    'checkmate' if self.board.is_in_check(self.turn) else 'stalemate'),legal
        if self.rules.draw_no_attackers and not any(p and p.kind in ('R','H','C','P') for row in self.board.grid for p in row):
            return ('1/2-1/2','no_attackers'),legal
        if self.counts[(self.board.signature(),self.turn)]>=self.rules.repetition:
            return ('1/2-1/2','repetition'),legal
        if self.quiet>=self.rules.quiet_plies: return ('1/2-1/2','quiet_limit'),legal
        if generated>=self.rules.max_plies: return ('1/2-1/2','max_plies'),legal
        return None,legal

    def move(self,action,legal):
        if type(action) is not int or not 0<=action<8100 or not legal[action]: raise ValueError('重放或采样包含非法动作')
        fr,fc,tr,tc=action_to_move(action)
        self.quiet=0 if self.board.grid[tr][tc] is not None else self.quiet+1
        self.board=self.board.apply_move(fr,fc,tr,tc)
        self.turn='b' if self.turn=='r' else 'r'
        self.counts[(self.board.signature(),self.turn)]+=1


class SelfPlayGenerator:
    def __init__(self,model,snapshot,*,exploration=Exploration(),rules=SelfPlayRules(),device='cpu',seed=42):
        if model.config.architecture!=snapshot.architecture: raise ValueError('模型和版本快照架构不一致')
        # Own the frozen copy: do not freeze or change the caller's training model.
        self.model=copy.deepcopy(model).to(device).eval().requires_grad_(False)
        self.snapshot,self.exploration,self.rules=snapshot,exploration,rules
        self.device,self.seed=device,seed
        self.has_value_head=hasattr(self.model,'value_head')

    @classmethod
    def from_version(cls,family,version,*,model_db=None,**kwargs):
        model,meta,_=load_version(family,version,path=model_db,device=kwargs.get('device','cpu'))
        return cls(model,Snapshot.from_metadata(meta),**kwargs)

    @torch.inference_mode()
    def _forward(self,position,cache):
        policy,values,cache=self._forward_batch([position],cache)
        return policy[0],float(values[0]),cache

    @torch.inference_mode()
    def _forward_batch(self,positions,cache,valid=None):
        active=[True]*len(positions) if valid is None else valid
        x=torch.zeros(len(positions),15,10,9)
        for i,position in enumerate(positions):
            if active[i]: x[i]=torch.from_numpy(encode_state(position.board,position.turn))
        mask=torch.tensor(active,device=self.device,dtype=torch.bool)
        output,cache=self.model.forward_step(x.to(self.device),cache,mask)
        policy,values=split_output(output)
        value=torch.zeros(len(positions),device=self.device)
        if values is not None:
            if values.shape!=(len(positions),3) or not torch.isfinite(values).all(): raise ValueError('WDL 输出无效')
            p=values.float().softmax(-1); value=p[:,0]-p[:,2]
        return policy,value,cache

    def generate(self,games,*,path=None,start_fen=STANDARD_FEN,prefix=(),cancel_event=None,progress=None,
                 save_every=16,micro_batch=None,on_step=None):
        if type(games) is not int or games<1: raise ValueError('对局数量必须为正整数')
        micro_batch=generation_micro_batch(save_every,micro_batch)
        prefix=tuple(prefix)
        # Validate the entire warm-up before opening a rollout run.
        check=Position(start_fen,self.rules)
        for action in prefix:
            terminal,legal=check.terminal(0)
            if terminal: raise ValueError('暖启动前缀已经结束，不能继续走棋')
            check.move(action,legal)
        if check.terminal(0)[0]: raise ValueError('探索起点已经终局')
        path,run_id=create_run(self.snapshot,self.exploration,self.rules,self.has_value_head,self.seed,path,
                               save_every=save_every,requested_games=games,micro_batch=micro_batch,device=str(self.device))
        ids=[]; pending=[]; completed=0; interrupted=False; active=[]; generated_steps=0; inference_batches=0
        try:
            for wave_start in range(0,games,micro_batch):
                check_stop(cancel_event)
                size=min(micro_batch,games-wave_start)
                rng=[torch.Generator().manual_seed(self.seed+wave_start+i) for i in range(size)]
                positions=[Position(start_fen,self.rules) for _ in range(size)]
                active=[True]*size; cache=None; records=[[] for _ in range(size)]
                for action in prefix:
                    check_stop(cancel_event)
                    _,_,cache=self._forward_batch(positions,cache)
                    inference_batches+=1
                    for position in positions: position.move(action,legal_action_mask(position.board,position.turn))
                    if on_step: on_step({'phase':'历史暖启动','steps':0,'batches':1,'active':size})
                while any(active):
                    check_stop(cancel_event)
                    masks=torch.zeros(size,8100,dtype=torch.bool)
                    for i,position in enumerate(positions):
                        if not active[i]: continue
                        terminal,legal=position.terminal(len(records[i]))
                        if terminal:
                            result,reason=terminal; active[i]=False
                            pending.append({'start_fen':start_fen,'prefix':prefix,'steps':records[i],'result':result,
                                'termination':reason,'final_fen':to_fen(position.board,position.turn),'generation_index':wave_start+i})
                            completed+=1
                            if len(pending)>=save_every:
                                ids.extend(save_games(path,run_id,pending)); pending.clear()
                            if progress:
                                progress({'game':completed,'saved_games':len(ids),'buffered_games':len(pending),
                                    'plies':len(records[i]),'result':result,'termination':reason,'micro_batch':size})
                            check_stop(cancel_event)
                        else: masks[i]=torch.from_numpy(legal)
                    if not any(active): break
                    with torch.inference_mode():
                        logits,values,cache=self._forward_batch(positions,cache,active)
                        actions,log_probs=sample_actions(logits,masks.to(self.device),self.exploration,rng,active)
                        values=values.cpu().tolist()
                    for i,position in enumerate(positions):
                        if not active[i]: continue
                        records[i].append({'turn':position.turn,'action':actions[i],'old_log_mu':log_probs[i],
                            'old_value':values[i],'temperature':self.exploration.temperature,'epsilon':self.exploration.epsilon})
                        position.move(actions[i],masks[i].numpy())
                    inference_batches+=1; generated_steps+=sum(active)
                    if on_step: on_step({'phase':'GPU 批量走棋','steps':sum(active),'batches':1,'active':sum(active)})
            check_stop(cancel_event)
            if pending: ids.extend(save_games(path,run_id,pending)); pending.clear()
        except (WorkInterrupted,KeyboardInterrupt):
            interrupted=True
        ids=committed_game_ids(path,run_id)
        discarded=completed-len(ids)
        pending.clear()
        finish_run(path,run_id,completed=completed,saved=len(ids),discarded=discarded,interrupted=interrupted)
        return {'run_id':run_id,'model':self.snapshot,'data_db':str(path),'game_ids':ids,
                'requested_games':games,'completed_games':completed,'saved_games':len(ids),
                'discarded_games':discarded,'unfinished_game_discarded':any(active),'unfinished_games':sum(active),
                'save_every':save_every,'micro_batch':micro_batch,'generated_steps':generated_steps,
                'inference_batches':inference_batches,'interrupted':interrupted}

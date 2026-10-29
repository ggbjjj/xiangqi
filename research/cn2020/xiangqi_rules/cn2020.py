"""INCOMPLETE CN2020 research draft; fails official cases, do not use in play.

The values here express the rule book's exchange categories, not an AI's
position evaluation. No trained engine or neural evaluator is consulted.
"""
from dataclasses import dataclass
from collections import Counter,deque


def other(side): return 'b' if side=='r' else 'r'


def pieces(board):
    return {id(piece):piece for row in board.grid for piece in row if piece is not None}


def captures(board,side):
    result=[]
    for r,row in enumerate(board.grid):
        for c,piece in enumerate(row):
            if piece is None or piece.color!=side: continue
            for tr,tc in board.pseudo_legal_moves(r,c):
                target=board.grid[tr][tc]
                if target is not None and target.kind!='K' and not board.apply_move(r,c,tr,tc).is_in_check(side):
                    result.append((r,c,tr,tc))
    return result


def margin(before,after,side,ignored_pawns=frozenset()):
    original=pieces(before);remaining=pieces(after)
    gained=Counter(p.kind for ident,p in original.items() if ident not in remaining and p.color!=side
                   and not (p.kind=='P' and ident in ignored_pawns))
    lost=Counter(p.kind for ident,p in original.items() if ident not in remaining and p.color==side)
    strong=lambda counts:2*counts['R']+counts['H']+counts['C']
    weak=lambda counts:counts['A']+counts['E']
    # A pawn exchanged for any non-chariot piece is not a material win (26.7).
    if gained and lost and ((set(lost)=={'P'} and not gained['R']) or
                            (set(gained)=={'P'} and not lost['R'])):
        return (0,0,0)
    return strong(gained)-strong(lost),weak(gained)-weak(lost),gained['P']-lost['P']


@dataclass(frozen=True)
class Target:
    kind: str
    unrooted: bool
    joint: bool=False
    ident: int=0


@dataclass(frozen=True)
class Assessment:
    kind: str='idle'
    targets: tuple=()


@dataclass(frozen=True)
class Proof:
    participants: frozenset=frozenset()
    targets: tuple=()
    first: tuple=()
    score: tuple=(0,0,0)


class Tactics:
    """Legal recaptures and consecutive-check proofs; repeated proof nodes fail."""
    def __init__(self,board,side,ignored_pawns=frozenset(),related=frozenset()):
        self.initial,self.side,self.ignored_pawns=board,side,ignored_pawns
        self.related=related
        self._mate_cache={};self._exchange_cache={}

    @staticmethod
    def mate_in_one(board,side):
        opponent=other(side)
        for move in board.all_legal_moves(side):
            after=board.apply_move(*move)
            if after.is_in_check(opponent) and not after.has_any_legal_move(opponent): return True
        return False

    def exchange(self,board,turn,square=None,seen=frozenset(),related=None):
        if related is None:
            participant=board.grid[square[0]][square[1]] if square is not None else None
            related=self.related|({id(participant)} if participant else set())
        key=(tuple(id(p) if p else 0 for row in board.grid for p in row),turn,frozenset(related))
        score=margin(self.initial,board,self.side,self.ignored_pawns)
        if key in seen:return score,frozenset()
        if key in self._exchange_cache:return self._exchange_cache[key]
        checked=board.is_in_check(turn)
        best=((-100,0,0) if turn==self.side else (100,0,0)) if checked else score
        participants=frozenset()
        moves=board.all_legal_moves(turn) if checked else captures(board,turn)
        for move in moves:
            actor=board.grid[move[0]][move[1]];victim=board.grid[move[2]][move[3]]
            if not checked and id(actor) not in related and id(victim) not in related:continue
            after=board.apply_move(*move)
            candidate,used=self.exchange(after,other(turn),seen=seen|{key},related=related|{id(actor),id(victim)})
            if turn==self.side:
                used=used|{id(board.grid[move[0]][move[1]])}
            if (candidate>best if turn==self.side else candidate<best):
                best,participants=candidate,used
        result=best,participants
        self._exchange_cache[key]=result
        return result

    def direct_gains(self,board):
        for move in captures(board,self.side):
            target=board.grid[move[2]][move[3]]
            if id(target) in self.ignored_pawns: continue
            after=board.apply_move(*move)
            result=self.checking_proof(after,'exchange',start_turn=other(self.side))
            if result is None:continue
            score,used=result.score,result.participants
            opponent=other(self.side)
            if self.mate_in_one(after,opponent): continue
            roots=[m for m in captures(after,other(self.side)) if m[2:]==move[2:]]
            participant=id(board.grid[move[0]][move[1]])
            used=used|{participant}
            yield Proof(used,(Target(target.kind,not roots,len(used)>1,id(target)),),
                        (participant,id(target)),score)

    def checking_proof(self,board,goal='mate',root_gains=True,root_checks=None,start_turn=None,allow_equal=False):
        """Iterative AND/OR proof, so short wins precede unrelated long lines.

        Unknown at the current horizon is distinct from a disproof. There is no
        fixed tactical depth limit. Cycles are defender escapes; dependency-
        tracked disproofs are only reused after their dependencies close.
        """
        def key(position,turn):
            return tuple(id(p) if p else 0 for row in position.grid for p in row),turn
        children_cache={};terminal_cache={};solved={}
        start_turn=start_turn or self.side
        root_key=key(board,start_turn)
        def won_material(position):
            score=margin(self.initial,position,self.side,self.ignored_pawns)
            return score>=(0,0,0) if allow_equal else score>(0,0,0)
        def material_proof(position):
            original=pieces(self.initial);remaining=pieces(position)
            targets=tuple(Target(p.kind,True,False,ident) for ident,p in original.items()
                          if ident not in remaining and p.color!=self.side and ident not in self.ignored_pawns)
            return Proof(targets=targets,score=margin(self.initial,position,self.side,self.ignored_pawns))
        def children(position,turn):
            k=key(position,turn)
            if k not in children_cache:
                attacking=turn==self.side;items=[];checked=position.is_in_check(turn)
                for move in position.all_legal_moves(turn):
                    after=position.apply_move(*move)
                    capture=position.grid[move[2]][move[3]] is not None
                    check=after.is_in_check(other(turn))
                    if k==root_key and root_checks is not None:
                        if check and move not in root_checks:continue
                        if capture and not root_gains:continue
                    if goal=='mate':
                        if attacking and not check:continue
                    elif not checked:
                        if attacking and not (check or capture):continue
                        if not attacking and not (capture or check):continue
                    # Capturing a checker first finds defensive refutations early.
                    priority=(len(after.all_legal_moves(other(turn))) if attacking else
                              -int(position.grid[move[2]][move[3]] is not None))
                    items.append((priority,after,id(position.grid[move[0]][move[1]]),move))
                children_cache[k]=sorted(items,key=lambda item:item[0])
            return children_cache[k]
        def frame(position,turn,depth):
            return {'board':position,'turn':turn,'key':key(position,turn),'children':None,'index':0,
                    'parts':frozenset(),'targets':(),'deps':set(),'edge':None,'entered':False,
                    'done':None,'depth':depth,'unknown':False,'score':None}
        horizon=1
        while True:
            active=set();stack=[frame(board,start_turn,horizon)];result=None;cache={}
            while stack:
                node=stack[-1];position,turn=node['board'],node['turn'];is_or=turn==self.side
                if result is not None:
                    winning,certificate,deps=result;result=None
                    if winning is None:node['unknown']=True
                    elif winning:
                        moved,move=node['edge'];parts=certificate.participants|({moved} if is_or else set())
                        if is_or:
                            node['done']=(True,Proof(parts,certificate.targets,(moved,move[2:]),certificate.score),frozenset())
                        else:
                            node['parts']=node['parts']|parts;node['targets']+=certificate.targets
                            node['score']=certificate.score if node['score'] is None else min(node['score'],certificate.score)
                    elif not is_or:node['done']=(False,None,frozenset(set(deps)-{node['key']}))
                    else:node['deps'].update(deps)
                if node['children'] is None and node['done'] is None:
                    active.add(node['key']);node['entered']=True
                    if goal!='mate' and not position.is_in_check(turn):
                        won=won_material(position)
                        if is_or and won:node['done']=(True,material_proof(position),frozenset())
                        elif not is_or and not won:node['done']=(False,None,frozenset())
                    if node['done'] is None:
                        node['children']=children(position,turn)
                        if not node['children']:
                            wins=((not is_or and position.is_in_check(turn)) if goal=='mate' else won_material(position))
                            node['done']=(wins,material_proof(position) if wins else None,frozenset())
                        elif node['depth']==0:node['done']=(None,None,frozenset())
                if node['done'] is None and node['index']>=len(node['children']):
                    if node['unknown']:node['done']=(None,None,frozenset())
                    else:node['done']=((False,None,frozenset(node['deps']-{node['key']})) if is_or else
                                      (True,Proof(node['parts'],node['targets'],score=node['score'] or (0,0,0)),frozenset()))
                if node['done'] is not None:
                    result=node['done'];stack.pop()
                    if node['entered']:active.remove(node['key'])
                    if result[0] is True or result[0] is False and not result[2]:solved[node['key']]=result
                    cache[node['key'],node['depth']]=result
                    continue
                _,after,moved,move=node['children'][node['index']];node['index']+=1;node['edge']=(moved,move)
                child_key=key(after,other(turn));depth=node['depth']-1
                if child_key in active:
                    if goal!='mate':
                        wins=won_material(after)
                        result=(wins,material_proof(after) if wins else None,frozenset())
                    else:result=(False,None,frozenset({child_key}))
                elif child_key in solved:result=solved[child_key]
                elif (child_key,depth) in cache:result=cache[child_key,depth]
                else:stack.append(frame(after,other(turn),depth))
            if result[0] is not None:return result[1]
            horizon+=2


def ignored_pawns(board,previous=None):
    ignored=set()
    for r,row in enumerate(board.grid):
        for piece in row:
            if piece and piece.kind=='P' and not board.crossed_river(r,piece.color): ignored.add(id(piece))
    if previous is not None:
        before,move=previous
        piece=before.grid[move[0]][move[1]]
        if piece and piece.kind=='P' and not before.crossed_river(move[0],piece.color) and board.crossed_river(move[2],piece.color):
            ignored.add(id(piece))
    return frozenset(ignored)


def assess_move(before,after,side,move,previous=None,related=frozenset()):
    if after.is_in_check(other(side)): return Assessment('check')
    moved=before.grid[move[0]][move[1]]
    if moved.kind=='K' and before.is_in_check(side): return Assessment()
    ignored=ignored_pawns(after,previous)
    related=related|{id(moved)}
    tactics=Tactics(after,side,ignored,related)
    safe_offers=set();offered=set()
    for candidate in captures(after,other(side)):
        attacker=after.grid[candidate[0]][candidate[1]]
        if candidate[2:]!=tuple(move[2:]) or attacker.kind!=moved.kind:continue
        offered.add(id(attacker))
        accepted=after.apply_move(*candidate)
        response=Tactics(after,other(side),ignored,related|{id(moved),id(attacker)})
        score,_=response.exchange(accepted,side,candidate[2:])
        if score>=(0,0,0) and not response.mate_in_one(accepted,side):
            original=Tactics(after,side,ignored,related)
            if original.checking_proof(accepted,'gain') is None:safe_offers.add(id(attacker))
    # Do not attribute a pre-existing capture to an unrelated quiet move. Keep
    # all candidates, since the first capture in board order may be unrelated.
    earlier={proof.first:proof for proof in Tactics(before,side,ignored,related).direct_gains(before)}
    direct=[proof for proof in tactics.direct_gains(after)
            if (proof.first not in earlier or proof.score>earlier[proof.first].score)
            and not (proof.first[0]==id(moved) and proof.first[1] in safe_offers)]
    checks=set()
    before_moves=set(before.all_legal_moves(side))
    for candidate in after.all_legal_moves(side):
        following=after.apply_move(*candidate)
        if not following.is_in_check(other(side)):continue
        piece=after.grid[candidate[0]][candidate[1]]
        if safe_offers and piece is moved:continue
        old=next(((r,c,*candidate[2:]) for r,row in enumerate(before.grid) for c,p in enumerate(row)
                  if p is piece),None)
        if old in before_moves:
            old_following=before.apply_move(*old)
            if old_following.signature()==following.signature():continue
            if old_following.is_in_check(other(side)):
                replies=lambda b:{(id(b.grid[m[0]][m[1]]),m[2:]) for m in b.all_legal_moves(other(side))}
                if replies(old_following)==replies(following):continue
        checks.add(candidate)
    mate_checks={m for m in checks if not (after.grid[m[0]][m[1]] is moved and
                 after.grid[m[2]][m[3]] is not None and id(after.grid[m[2]][m[3]]) in offered)}
    proof=tactics.checking_proof(after,'mate',root_checks=mate_checks)
    if proof is not None: return Assessment('kill')
    proof=next((p for p in direct if moved.kind not in ('K','P') or p.participants!={id(moved)}),None)
    if proof is None:proof=tactics.checking_proof(after,'gain',root_gains=False,root_checks=checks)
    if proof is None or not proof.targets: return Assessment()
    # A king/pawn alone can chase; collaboration with another piece is different.
    if moved.kind in ('K','P') and proof.participants=={id(moved)}: return Assessment()
    targets=tuple(Target(t.kind,t.unrooted,t.joint or len(proof.participants)>1,t.ident) for t in proof.targets)
    return Assessment('chase',targets)


def severity(assessments):
    if not assessments or any(item.kind=='idle' for item in assessments): return -1
    if all(item.kind=='check' for item in assessments): return 3
    def important(item):
        return item.kind in ('check','kill') or (item.targets and all(
            (target.kind=='R' or target.unrooted) and not target.joint for target in item.targets))
    return 2 if all(important(item) for item in assessments) else 0


def analyse_cycle(states,moves,start,stop):
    from . import RepetitionDecision
    by_side={'r':[],'b':[]}
    # Continuous check is inexpensive and takes precedence over all chasing cases.
    checks={side:all(states[i+1][0].is_in_check(other(side)) for i in range(start,stop) if states[i][1]==side)
            for side in by_side}
    if checks['r']!=checks['b']:
        side='r' if checks['r'] else 'b'
        return RepetitionDecision(frozenset({side}),'长将须变招',True)
    if checks['r'] and checks['b']: return RepetitionDecision(repeated=True)
    related=frozenset(id(states[i][0].grid[moves[i][0]][moves[i][1]]) for i in range(start,stop))
    for i in range(start,stop):
        before,side=states[i]
        previous=(states[i-1][0],moves[i-1]) if i else None
        by_side[side].append(assess_move(before,states[i+1][0],side,moves[i],previous,related))
    ranks={side:severity(items) for side,items in by_side.items()}
    responsible={side for side in by_side if ranks[side]>=0 and
                 (ranks[other(side)]<0 or (ranks[side]>=2 and ranks[side]>ranks[other(side)]))}
    return RepetitionDecision(frozenset(responsible),'循环攻击须变招' if responsible else '',True)

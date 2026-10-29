"""Decode the legacy CCBridge v21 records stored inside this project's CBLs.

Record layout: 32 piece coordinates, record type, length-prefixed GB18030
metadata, result, timestamps, six-byte depth-first move nodes, then comments.
Only the first variation at every branch is the training mainline. Other
variations and comments remain available in the original raw record.
"""
from dataclasses import dataclass
import struct

from board import Board
from piece import Piece
from encoder import move_to_action
from .notation import to_fen


DECODER_VERSION='ccbridge-v21-mainline-v1'
RESULTS={1:'1-0',2:'0-1',3:'1/2-1/2',4:'*'}
PIECES='RHEAKAEHRCCPPPPP'


class Reader:
    def __init__(self,data): self.data,self.pos=data,0

    def take(self,n):
        if n<0 or self.pos+n>len(self.data): raise ValueError(f'CBL record truncated at byte {self.pos}')
        value=self.data[self.pos:self.pos+n];self.pos+=n;return value

    def integer(self): return struct.unpack('<I',self.take(4))[0]

    def string(self): return self.take(self.integer()).decode('gb18030',errors='replace')


@dataclass
class DecodedRecord:
    start_fen: str
    actions: list
    result: str
    metadata: dict


def decode_record(data):
    reader=Reader(data)
    positions=reader.take(32)
    record_type=reader.integer()
    folder=reader.string()
    layout_version=reader.integer()
    if layout_version!=1: raise ValueError(f'Unsupported CBL record layout {layout_version}')
    names=('title','remark','event','place','date','red','black')
    metadata=dict(zip(names,(reader.string() for _ in names)))
    result_code=reader.integer()
    if result_code not in RESULTS: raise ValueError(f'Unknown CBL result code {result_code}')
    extra=('source','source_email','author','author_email','root_comment')
    metadata.update(zip(extra,(reader.string() for _ in extra)))
    marker=reader.take(1)[0]
    if marker not in (0,1): raise ValueError(f'Unknown CBL timestamp marker {marker}')
    timestamps=reader.take(38).decode('ascii',errors='replace')
    count=reader.integer()
    nodes=list(struct.iter_unpack('<HBBBB',reader.take(count*6)))
    # The trailing text has one comment per node; it cannot supply move bytes.
    metadata.update(decoder=DECODER_VERSION,record_type=record_type,folder=folder,
        result_code=result_code,node_count=count,timestamps=timestamps)
    grid=[[None]*9 for _ in range(10)]
    for index,coordinate in enumerate(positions):
        if coordinate==0: continue
        col,row=coordinate//16-1,coordinate%16-1
        if not 0<=col<9 or not 0<=row<10: raise ValueError('CBL initial piece coordinate out of bounds')
        if grid[row][col] is not None: raise ValueError('CBL initial pieces overlap')
        grid[row][col]=Piece('r' if index<16 else 'b',PIECES[index%16])
    board=Board(grid)
    if any(sum(p is not None and p.kind=='K' and p.color==color for row in grid for p in row)!=1
           for color in ('r','b')): raise ValueError('CBL initial board must have both kings')
    actions=[];branch_seen=False;previous=0
    for depth,src,dst,_,_ in nodes:
        if depth<1 or depth>previous+1: raise ValueError('CBL variation depth is invalid')
        if depth<=previous: branch_seen=True
        if not branch_seen:
            if depth!=len(actions)+1: raise ValueError('CBL mainline depth is not contiguous')
            fr,fc,tr,tc=src%16,src//16,dst%16,dst//16
            if not (0<=fr<10 and 0<=tr<10 and 0<=fc<9 and 0<=tc<9) or src==dst:
                raise ValueError('CBL move coordinate out of bounds')
            actions.append(move_to_action((fr,fc,tr,tc)))
        previous=depth
    turn='r'
    if actions:
        first=nodes[0][1]
        piece=board.piece_at(first%16,first//16)
        if piece is None: raise ValueError('CBL first move has no piece at source')
        turn=piece.color
    metadata['mainline_plies']=len(actions)
    metadata['start_turn_inferred_from_first_move']=bool(actions)
    return DecodedRecord(to_fen(board,turn),actions,RESULTS[result_code],metadata)


def validate_mainline(record):
    """Validate the complete mainline without cutting off erroneous tails."""
    from .notation import from_fen
    from encoder import action_to_move
    board,turn=from_fen(record.start_fen)
    for ply,action in enumerate(record.actions,1):
        fr,fc,tr,tc=action_to_move(action)
        piece=board.piece_at(fr,fc)
        if piece is None or piece.color!=turn or (tr,tc) not in board.legal_moves(fr,fc):
            raise ValueError(f'ply={ply}: illegal move {fr}{fc}-{tr}{tc}, turn={turn}')
        board=board.apply_move(fr,fc,tr,tc)
        turn='b' if turn=='r' else 'r'
    if not record.actions: raise ValueError('empty mainline')
    return board,turn

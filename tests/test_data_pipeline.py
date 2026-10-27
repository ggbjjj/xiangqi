import hashlib
import json
import pytest

from data_pipeline.notation import from_fen,to_fen,STANDARD_FEN,parse_move,validate_candidate
from data_pipeline.database import connect
from data_pipeline.import_records import store_validated
from encoder import move_to_action


def test_notation_and_complete_record_validation():
    b,t=from_fen(STANDARD_FEN)
    assert to_fen(b,t)==STANDARD_FEN
    move=parse_move(b,t,'炮二平五'); assert move==(7,7,7,4)
    b=b.apply_move(*move)
    assert parse_move(b,'b','馬8進7')==(0,7,2,6)
    assert parse_move(b,'b','h9g7')==(0,7,2,6)
    record={'start_fen':STANDARD_FEN,'moves_zh_normalized':['炮二平五','马8进7','车1进9']}
    result=validate_candidate(record)
    assert result[-1] and 'ply=3' in result[-1] and result[2]==[]


def test_reader_alignment_truncation_padding_and_split(tmp_path):
    torch=pytest.importorskip('torch')
    from data_pipeline.loader import GameDataset,iter_game_batches
    path=tmp_path/'dataset.sqlite3'
    c=connect(path)
    moves=[move_to_action(m) for m in [(6,0,5,0),(3,0,4,0),(6,2,5,2)]]
    for i,length in enumerate([1,3]):
        c.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                   (str(i),STANDARD_FEN,json.dumps(moves[:length]),length,'*','human_master','train',str(i),'{}'))
    c.commit();c.close()
    x,y,lengths=next(iter_game_batches(2,2,db=path,shuffle=False))
    assert x.shape==(2,2,15,10,9) and y.shape==(2,2,8100) and lengths.tolist()==[1,2]
    assert y[0,0,moves[0]]==1 and y[1,1,moves[1]]==1
    assert not x[1,0].count_nonzero() and not y[1,0].count_nonzero()
    # At t=1 the red pawn has already moved and black is to move.
    assert x[1,1,6,5,0]==1 and x[1,1,6,6,0]==0 and not x[1,1,14].count_nonzero()
    assert len(list(iter_game_batches(2,128,db=path)))==1
    assert len(list(iter_game_batches(2,128,db=path,drop_last=True)))==0

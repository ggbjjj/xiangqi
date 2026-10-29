import hashlib
import json

import pytest

from data_pipeline.database import connect
from data_pipeline.import_endgames import decode_pgn, distribution, populate
from data_pipeline.merge_cbl import identity
from data_pipeline.notation import STANDARD_FEN


def pgn(moves='1. a3a4 a6a5 2. c3c4', result='1-0'):
    return f'[FEN "{STANDARD_FEN}"]\n[Result "{result}"]\n\n{moves} {result}\n'.encode()


def entry(name, payload):
    return {'source_id':'CCPD','source_path':f'Dataset/殘局/{name}.pgn',
            'index':0,'payload':payload,'format':'pgn'}


def test_pgn_keeps_full_mainline_and_rejects_unrecognized_tail():
    plain = decode_pgn(pgn())
    annotated = decode_pgn(pgn('1. a3a4 {注释} (1. c3c4 (a6a5)) a6a5 2. c3c4'))
    assert annotated.actions == plain.actions and annotated.start_fen == STANDARD_FEN
    assert len(plain.actions) == 3 and plain.result == '1-0'
    with pytest.raises(ValueError, match='unparsed'):
        decode_pgn(pgn('1. a3a4 a6a5 2. c3c4 BROKEN'))
    with pytest.raises(ValueError, match='ply=2'):
        decode_pgn(pgn('1. a3a4 a6a3'))
    with pytest.raises(ValueError, match='unclosed'):
        decode_pgn(pgn('1. a3a4 {unfinished'))


def test_import_deduplicates_preserves_original_results_and_isolates_bad_records(tmp_path):
    path = tmp_path/'endgames.sqlite3'
    conn = connect(path)
    conn.execute('INSERT INTO sources VALUES (?,?,?,?)', ('CCPD','local-test','CC-BY-4.0','{}'))
    payloads = [entry('red',pgn()), entry('unknown',pgn(result='*')),
        entry('black',pgn(result='0-1')), entry('new',pgn('1. a3a4',result='*')),
        entry('bad',pgn('1. a3a4 a6a3'))]
    record = decode_pgn(pgn())
    key, group, _ = identity(record.start_fen, record.actions)
    old = {key:{'split':'test','split_group':'original-group'}}
    with conn: populate(conn,payloads,old)
    assert conn.execute('SELECT count(*) FROM games').fetchone()[0] == 2
    assert dict(conn.execute('SELECT status,count(*) FROM raw_records GROUP BY status')) == {'validated':4,'rejected':1}
    row = conn.execute('SELECT * FROM games WHERE game_id=?',(key,)).fetchone()
    assert row['split'] == 'test' and row['split_group'] == 'original-group' and row['result'] == '*'
    assert json.loads(row['metadata_json'])['result_conflict']
    assert json.loads(row['metadata_json'])['overlaps_book_database']
    assert len(json.loads(row['metadata_json'])['source_records']) == 3
    assert sorted(r[0] for r in conn.execute('SELECT result FROM decoded_records WHERE validation_error IS NULL')) == ['*','*','0-1','1-0']
    for row in conn.execute('SELECT payload,sha256 FROM raw_records'):
        assert hashlib.sha256(row['payload']).hexdigest() == row['sha256']
    assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    conn.close()
    # The independent file uses the existing reader, including unknown-value masking.
    from data_pipeline.loader import GameDataset
    sample = GameDataset(T=2, split='test', db=path, include_value=True).read_segment(1)
    assert sample.length == 1 and sample.start_ply == 2
    assert sample.value_targets.tolist() == [-100,-100] and not sample.x[1:].any()


def test_distribution_counts_each_unique_game_once_and_bin_edges(tmp_path):
    conn = connect(tmp_path/'stats.sqlite3')
    for i,n in enumerate((1,32,33,64,65)):
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
            (str(i),STANDARD_FEN,'[]',n,'*','endgame_study','train',str(i),'{}'))
    report = distribution(conn,32)
    assert [item['count'] for item in report['bins']] == [2,2,1]
    assert report['stats'] == {'count':5,'total_plies':195,'min':1,'max':65,'mean':39,'median':33}
    assert sum(item['percent'] for item in report['bins']) == pytest.approx(100)
    conn.close()

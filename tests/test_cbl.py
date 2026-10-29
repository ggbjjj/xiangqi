import hashlib
import json
import struct

import pytest

from board import Board
from data_pipeline.cbl import PIECES, decode_record, validate_mainline
from data_pipeline.database import connect
from data_pipeline.merge_cbl import backup_database, identity, length_stats, merge_cbl
from data_pipeline.notation import STANDARD_FEN, to_fen
from encoder import move_to_action


def cbl_record(moves, *, board=None, result=1, branches=(), folder='古谱\\残局'):
    """A minimal v21 record with explicit binary layout, including a variable path."""
    board = board or Board.fresh()
    positions = bytearray(32)
    for color, base in [('r', 0), ('b', 16)]:
        available = {}
        for i, kind in enumerate(PIECES): available.setdefault(kind, []).append(base+i)
        for r, row in enumerate(board.grid):
            for c, piece in enumerate(row):
                if piece is not None and piece.color == color:
                    positions[available[piece.kind].pop(0)] = (c+1)*16+r+1
    def integer(value): return struct.pack('<I', value)
    def string(value):
        value = value.encode('gb18030')
        return integer(len(value))+value
    data = bytes(positions)+integer(2)+string(folder)+integer(1)
    data += b''.join(string(s) for s in ('测试古谱', '', '赛事', '广州', '2012', '红方', '黑方'))
    data += integer(result)+b''.join(string(s) for s in ('来源', '', '作者', '', '根注释'))
    nodes = [(i+1, *move) for i, move in enumerate(moves)]+list(branches)
    data += bytes([1])+b'2012-01-01 12:00:00'*2+integer(len(nodes))
    data += b''.join(struct.pack('<HBBBB', depth, fc*16+fr, tc*16+tr, 0, 0)
                     for depth, fr, fc, tr, tc in nodes)
    return data


MOVES = [(6, 0, 5, 0), (3, 0, 4, 0), (6, 2, 5, 2)]


@pytest.mark.parametrize('code,result', [(1, '1-0'), (2, '0-1'), (3, '1/2-1/2'), (4, '*')])
def test_cbl_layout_mainline_results_and_variable_folder(code, result):
    blob = cbl_record(MOVES, result=code, branches=[(1, 6, 2, 5, 2), (2, 3, 2, 4, 2)])
    record = decode_record(blob)
    assert record.start_fen == STANDARD_FEN
    assert record.actions == [move_to_action(m) for m in MOVES]
    assert record.result == result
    assert record.metadata['folder'] == '古谱\\残局'
    assert record.metadata['title'] == '测试古谱'
    assert record.metadata['node_count'] == 5
    board, turn = validate_mainline(record)
    assert turn == 'b' and board.piece_at(5, 2).color == 'r'


def test_cbl_nonstandard_board_black_to_move_and_value_loading(tmp_path):
    board = Board.fresh().apply_move(6, 0, 5, 0)
    record = decode_record(cbl_record(MOVES[1:], board=board, result=2))
    assert record.start_fen == to_fen(board, 'b')
    validate_mainline(record)
    conn = connect(tmp_path/'dataset.sqlite3')
    add_raw(conn, 'black', cbl_record(MOVES[1:], board=board, result=2))
    report = merge_cbl(conn)
    row = conn.execute('SELECT * FROM games').fetchone()
    assert report['added_games'] == 1
    from data_pipeline.loader import GameDataset
    sample = GameDataset(T=4, split=row['split'], db=tmp_path/'dataset.sqlite3', include_value=True)[0]
    x, y, legal, length, value = sample
    assert length == 2 and value.tolist() == [0, 2, -100, -100]
    assert not x[0, 14].any() and x[1, 14].all()
    assert y[:2].tolist() == record.actions and not x[2:].any()
    conn.close()


def test_cbl_rejects_truncation_bad_depth_empty_and_illegal_tail():
    blob = cbl_record(MOVES)
    with pytest.raises(ValueError, match='truncated'): decode_record(blob[:-1])
    with pytest.raises(ValueError, match='depth'): decode_record(cbl_record(MOVES, branches=[(5, 6, 4, 5, 4)]))
    empty = decode_record(cbl_record([]))
    with pytest.raises(ValueError, match='empty'): validate_mainline(empty)
    record = decode_record(cbl_record([MOVES[0], (3, 0, 6, 0)]))
    with pytest.raises(ValueError, match='ply=2'): validate_mainline(record)


def add_raw(conn, raw_id, blob):
    conn.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?)', ('CNVCS', 'local-test', None, '{}'))
    conn.execute('INSERT INTO raw_records VALUES (?,?,?,?,?,?,?,?,?,?)',
        (raw_id, 'CNVCS', f'raw/cnvcs_gupu/{raw_id}.cbl', 0, blob, 'cbl21',
         hashlib.sha256(blob).hexdigest(), 'pending_decoder', None, None))
    conn.commit()


def test_merge_deduplicates_preserves_splits_and_is_resumable(tmp_path):
    conn = connect(tmp_path/'dataset.sqlite3')
    actions = [move_to_action(m) for m in MOVES]
    game_id, group, _ = identity(STANDARD_FEN, actions)
    # Deliberately preserve an existing split rather than recompute it.
    conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
        (game_id, STANDARD_FEN, json.dumps(actions), 3, '*', 'human_master', 'test', group, '{}'))
    add_raw(conn, 'one', cbl_record(MOVES, result=3))
    add_raw(conn, 'two', cbl_record(MOVES, result=4))
    add_raw(conn, 'new', cbl_record(MOVES[:1]))
    report = merge_cbl(conn)
    assert report['processed_this_run'] == 3 and report['added_games'] == 1
    assert report['duplicate_records_this_run'] == 2
    row = conn.execute('SELECT * FROM games WHERE game_id=?', (game_id,)).fetchone()
    assert row['result'] == '1/2-1/2' and row['split'] == 'test'
    assert conn.execute('SELECT count(*) FROM decoded_records').fetchone()[0] == 3
    assert merge_cbl(conn)['processed_this_run'] == 0
    assert merge_cbl(conn, reprocess=True)['added_games'] == 0
    assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    assert conn.execute('SELECT payload FROM raw_records WHERE raw_id=?', ('two',)).fetchone()[0] == cbl_record(MOVES, result=4)
    conn.close()


def test_merge_preserves_original_results_and_quarantines_errors(tmp_path):
    conn = connect(tmp_path/'dataset.sqlite3')
    for name, moves, result in [('red', MOVES, 1), ('black', MOVES, 2),
                                 ('empty', [], 3), ('bad', [MOVES[0], (3, 0, 6, 0)], 2)]:
        add_raw(conn, name, cbl_record(moves, result=result))
    report = merge_cbl(conn)
    assert report['cbl_status'] == {'validated': 2, 'excluded_empty': 1, 'rejected': 1}
    assert report['added_games'] == 1 and len(report['result_conflicts_this_run']) == 1
    game = conn.execute('SELECT * FROM games').fetchone()
    assert game['result'] == '*' and json.loads(game['metadata_json'])['result_conflict']
    assert dict(conn.execute('SELECT raw_id,result FROM decoded_records')) == {
        'red': '1-0', 'black': '0-1', 'empty': '1/2-1/2', 'bad': '0-1'}
    assert conn.execute('SELECT ply_count FROM decoded_records WHERE raw_id=?', ('bad',)).fetchone()[0] == 2
    add_raw(conn, 'later', cbl_record(MOVES, result=1))
    merge_cbl(conn)
    assert conn.execute('SELECT result FROM games').fetchone()[0] == '*'
    conn.close()


def test_backup_and_statistics(tmp_path):
    path = tmp_path/'dataset.sqlite3'
    conn = connect(path)
    add_raw(conn, 'one', cbl_record(MOVES))
    backup = tmp_path/'backups'/'snapshot.sqlite3'
    backup_database(path, backup)
    merge_cbl(conn)
    original = connect(backup, readonly=True)
    assert original.execute('SELECT count(*) FROM games').fetchone()[0] == 0
    assert conn.execute('SELECT count(*) FROM games').fetchone()[0] == 1
    original.close(); conn.close()
    assert length_stats([0, 2, 3, 9]) == {'count': 4, 'total_plies': 14, 'max': 9, 'mean': 3.5, 'median': 2.5}


def test_cli_report_and_backup_without_plot_dependencies(tmp_path, monkeypatch):
    from data_pipeline import merge_cbl as merger
    monkeypatch.setattr(merger, 'ROOT', tmp_path)
    path = tmp_path/'dataset.sqlite3'
    conn = connect(path)
    add_raw(conn, 'one', cbl_record(MOVES))
    conn.close()
    output = tmp_path/'report'
    merger.main(['--db', str(path), '--output', str(output), '--no-plot'])
    report = json.loads((path.parent/'cbl_merge_report.json').read_text(encoding='utf-8'))
    assert report['added_games'] == 1 and report['cbl_valid_mainlines']['mean'] == 3
    assert (output/'统计报告.md').exists() and not (output/'棋谱步数统计.png').exists()
    assert (tmp_path/'backups').exists()

from contextlib import closing
import json
import sqlite3

import pytest
from selfplay.store import init_store,connect
from selfplay.stats import count_games,game_counts
from selfplay.__main__ import main


def add_games(path,family,run_id,n):
    init_store(family,path)
    with closing(connect(path)) as conn,conn:
        conn.execute('''INSERT INTO runs(run_id,family,architecture,checkpoint_id,version_name,
            weights_sha256,model_config_json,exploration_json,rules_json,has_value_head,seed,
            requested_games,saved_games,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (run_id,family,'Haiku-5' if family=='HaiKu' else 'Opus-5',run_id,run_id,
             'test-sha','{}','{}','{}',0,42,1000,999,'running'))
        for i in range(n):
            conn.execute('''INSERT INTO games(game_id,run_id,start_fen,prefix_json,moves_json,
                result,termination,final_fen,ply_count) VALUES (?,?,?,?,?,?,?,?,?)''',
                (f'{run_id}-{i}',run_id,'test-fen','[]','[4905]','1/2-1/2','max_plies','test-fen',1))


def test_counts_all_versions_and_actual_rows_without_writing(tmp_path,capsys):
    haiku,opus=tmp_path/'haiku.sqlite3',tmp_path/'opus.sqlite3'
    add_games(haiku,'HaiKu','v1',2); add_games(haiku,'HaiKu','v2',1)
    add_games(opus,'Opus','o1',4)
    before=(haiku.read_bytes(),opus.read_bytes())
    rows=main(['count','--haiku-db',str(haiku),'--opus-db',str(opus)])
    assert [(r['family'],r['games']) for r in rows]==[('HaiKu',3),('Opus',4)]
    assert capsys.readouterr().out=='HaiKu：3 局\nOpus：4 局\n'
    main(['count','--family','Opus','--opus-db',str(opus),'--json'])
    assert json.loads(capsys.readouterr().out)[0]['games']==4
    assert before==(haiku.read_bytes(),opus.read_bytes())


def test_missing_stores_report_zero_without_creating_files(tmp_path):
    paths={f:tmp_path/f/'games.sqlite3' for f in ('HaiKu','Opus')}
    assert [r['games'] for r in game_counts(paths=paths)]==[0,0]
    assert not any(p.parent.exists() for p in paths.values())


def test_counts_exclude_uncommitted_games(tmp_path):
    path=tmp_path/'data.sqlite3'; add_games(path,'HaiKu','v1',1)
    with closing(connect(path)) as writer:
        writer.execute('''INSERT INTO games(game_id,run_id,start_fen,prefix_json,moves_json,
            result,termination,final_fen,ply_count) VALUES (?,?,?,?,?,?,?,?,?)''',
            ('buffered','v1','test-fen','[]','[4905]','1/2-1/2','max_plies','test-fen',1))
        assert count_games('HaiKu',path=path)==1
        writer.rollback()
    assert count_games('HaiKu',path=path)==1


def test_wrong_family_or_other_database_is_rejected(tmp_path):
    path=tmp_path/'haiku.sqlite3'; add_games(path,'HaiKu','v1',1)
    with pytest.raises(ValueError,match='系列'): count_games('Opus',path=path)
    other=tmp_path/'other.sqlite3'
    with closing(sqlite3.connect(other)) as conn:
        conn.execute('CREATE TABLE games(game_id TEXT)')
    with pytest.raises(ValueError,match='不是自弈'): count_games('HaiKu',path=other)

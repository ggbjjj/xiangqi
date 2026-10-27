from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tarfile

import pytest

from tools.export_training import export_training,snapshot_sqlite,digest


def database(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as conn,conn:
        conn.execute('CREATE TABLE records(id INTEGER PRIMARY KEY,value TEXT)')
        conn.execute("INSERT INTO records VALUES (1,'checkpoint')")


def test_snapshot_reads_committed_wal_and_preserves_original(tmp_path):
    source=tmp_path/'live.sqlite3'
    with closing(sqlite3.connect(source)) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE records(value INTEGER)')
        writer.execute('INSERT INTO records VALUES (7)'); writer.commit()
        writer.execute('INSERT INTO records VALUES (9)')  # uncommitted data is not part of the snapshot
        target=tmp_path/'backup.sqlite3'; snapshot_sqlite(source,target)
        with closing(sqlite3.connect(target)) as conn:
            assert conn.execute('SELECT value FROM records').fetchall()==[(7,)]
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        writer.rollback()
    with pytest.raises(ValueError,match='覆盖'): snapshot_sqlite(source,target)


def test_return_bundle_preserves_paths_manifests_and_separates_model_family(tmp_path):
    database(tmp_path/'ModelParameters'/'Opus'/'models.sqlite3')
    database(tmp_path/'ModelParameters'/'HaiKu'/'models.sqlite3')
    database(tmp_path/'Data'/'SelfPlay'/'Opus'/'games.sqlite3')
    run=tmp_path/'runs'/'mixed'/'opus-run'; run.mkdir(parents=True)
    (run/'config.json').write_text('{"family":"Opus"}')
    (run/'best.json').write_text('{"parameter_id":3}')
    (run/'tensorboard').mkdir(); (run/'tensorboard'/'events.out.tfevents.test').write_bytes(b'events')
    before=digest(tmp_path/'ModelParameters'/'Opus'/'models.sqlite3')
    report=export_training('Opus',[run],root=tmp_path,include_selfplay=True)
    directory=Path(report['directory'])
    assert digest(tmp_path/'ModelParameters'/'Opus'/'models.sqlite3')==before
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    paths={entry['path'] for entry in manifest['files']}
    assert paths=={'ModelParameters/Opus/models.sqlite3','Data/SelfPlay/Opus/games.sqlite3',
                   'runs/mixed/opus-run/config.json','runs/mixed/opus-run/best.json',
                   'runs/mixed/opus-run/tensorboard/events.out.tfevents.test'}
    assert all(digest(directory/entry['path'])==entry['sha256'] for entry in manifest['files'])
    with tarfile.open(report['archive']) as tar:
        assert directory.name+'/manifest.json' in tar.getnames()
        assert not any('HaiKu/' in name for name in tar.getnames())
    assert report['archive_sha256']==digest(report['archive'])
    second=export_training('Opus',root=tmp_path)
    assert second['directory']!=report['directory']


def test_export_rejects_missing_input_and_wrong_run_before_creating_bundle(tmp_path):
    with pytest.raises(ValueError,match='数据库不存在'): export_training('HaiKu',root=tmp_path)
    assert not (tmp_path/'exports').exists()
    database(tmp_path/'ModelParameters'/'HaiKu'/'models.sqlite3')
    with pytest.raises(ValueError,match='config.json'): export_training('HaiKu',[tmp_path],root=tmp_path)
    with pytest.raises(ValueError,match='数据库不存在'): export_training('HaiKu',root=tmp_path,include_selfplay=True)
    assert not (tmp_path/'exports').exists()


def test_export_rejects_logs_from_another_model_family(tmp_path):
    database(tmp_path/'ModelParameters'/'HaiKu'/'models.sqlite3')
    run=tmp_path/'runs'/'opus'; run.mkdir(parents=True)
    (run/'config.json').write_text('{"family":"Opus"}')
    with pytest.raises(ValueError,match='系列不一致'): export_training('HaiKu',[run],root=tmp_path)
    assert not (tmp_path/'exports').exists()

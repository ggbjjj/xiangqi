"""Reproducible, resumable source import: python -m data_pipeline.import_records."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import zlib

from data_pipeline.database import connect
from data_pipeline.notation import validate_candidate
from project_paths import ROOT, DATA_DB


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cbl_records(data):
    if data[:8] != b'CCBridge': raise ValueError('CBL signature')
    version, flag, size = struct.unpack_from('<III', data, 8)
    if version != 21 or flag not in (0,1,3): raise ValueError('CBL version')
    payload = zlib.decompress(data[20:]) if flag else data[20:]
    if len(payload) != size: raise ValueError('CBL payload size')
    pos = 0
    for _ in range(5): pos += 4 + struct.unpack_from('<I', payload, pos)[0]
    pos += 38
    count = struct.unpack_from('<I', payload, pos)[0]
    pos += 4
    pos += 4 + struct.unpack_from('<I', payload, pos)[0]
    records = []
    for _ in range(count):
        length = struct.unpack_from('<I', payload, pos)[0]
        if length < 4 or pos+length > len(payload): raise ValueError('CBL bounds')
        records.append(payload[pos+4:pos+length])
        pos += length
    if pos != len(payload): raise ValueError('CBL trailing bytes')
    return records


def import_raw(conn, survey):
    sources = [('CCPD','https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset','CC-BY-4.0',
                dumps({'commit':'368a47a947773dd8692c026e286dd19b6277b993'})),
               ('CNVCS','https://www.cnvcs.com/xiangqi/list.html','unspecified','{}')]
    conn.executemany('INSERT OR IGNORE INTO sources VALUES (?,?,?,?)', sources)
    for path in [survey/'raw/ccpd_main.zip', *sorted((survey/'raw').glob('cnvcs_*/*.cbl'))]:
        rel = path.relative_to(survey).as_posix()
        if conn.execute('SELECT 1 FROM source_files WHERE path=?', (rel,)).fetchone(): continue
        data = path.read_bytes()
        source = 'CCPD' if path.suffix == '.zip' else 'CNVCS'
        conn.execute('INSERT INTO source_files VALUES (?,?,?,?)', (rel,source,digest(data),data))
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                records = [('/'.join(n.split('/')[1:]),0,archive.read(n),'pgn',
                            'pending' if '/大師對局/' in n else 'excluded_category')
                           for n in archive.namelist() if n.lower().endswith('.pgn')]
        else:
            records = [(rel,i,raw,'cbl21','pending_decoder') for i,raw in enumerate(cbl_records(data))]
        conn.executemany('INSERT OR IGNORE INTO raw_records VALUES (?,?,?,?,?,?,?,?,?,NULL)',
            [(digest(f'{source}:{name}:{i}'.encode()),source,name,i,raw,fmt,digest(raw),status,
              'CBL 主线解码尚未实现' if status == 'pending_decoder' else None)
             for name,i,raw,fmt,status in records])
        conn.commit()
        print(f'原始文件入库 {rel} ({len(records)} 局)', flush=True)


def store_validated(conn, result):
    record, fen, actions, error = result
    game_id = None
    if error is None:
        game_id = digest((fen+'\n'+dumps(actions)).encode())
        # Same initial board + first 24 plies stay together (reduces near-duplicate leakage).
        group = digest(('haiku5-split-v1\n'+fen+'\n'+dumps(actions[:24])).encode())
        bucket = int(group[:8],16) % 1000
        split = 'train' if bucket < 900 else 'validation' if bucket < 950 else 'test'
        metadata = {k:record.get(k) for k in ('red','black','event','date','game_id')}
        conn.execute('INSERT OR IGNORE INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                     (game_id,fen,dumps(actions),len(actions),record.get('result','*'),
                      'human_master',split,group,dumps(metadata)))
    conn.executemany('UPDATE raw_records SET status=?,error=?,game_id=? WHERE source_id=? AND source_path=?',
                     [('validated' if error is None else 'rejected',error,game_id,'CCPD',p)
                      for p in record['source_paths']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey',type=Path,default=ROOT/'pretrain_data/source_survey_20261024')
    parser.add_argument('--db',type=Path,default=DATA_DB)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='仅校验前 N 个候选，0 为全部')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.limit < 0: parser.error('workers>=1; limit>=0')
    conn = connect(args.db)
    try:
        import_raw(conn,args.survey)
        candidates = []
        with (args.survey/'human_master_candidates.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                row = conn.execute('SELECT status FROM raw_records WHERE source_id=? AND source_path=?',
                                   ('CCPD',record['source_paths'][0])).fetchone()
                if row and row['status'] == 'pending': candidates.append(record)
                if args.limit and len(candidates) >= args.limit: break
        from tqdm import tqdm
        if args.workers == 1:
            results = map(validate_candidate,candidates)
            for i,result in enumerate(tqdm(results,total=len(candidates),desc='逐局规则校验'),1):
                store_validated(conn,result)
                if i % 200 == 0: conn.commit()
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                for i,result in enumerate(tqdm(pool.map(validate_candidate,candidates,chunksize=32),
                                               total=len(candidates),desc='逐局规则校验'),1):
                    store_validated(conn,result)
                    if i % 200 == 0: conn.commit()
        conn.execute("UPDATE raw_records SET status='rejected',error='未通过文本结构筛选或空棋谱' "
                     "WHERE status='pending'") if not args.limit else None
        conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                     ('encoding','absolute-red-black-v1; action=90*(9*r+c)+(9*tr+tc)'))
        conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                     ('validation','piece legality + king safety; no tournament repetition adjudication'))
        conn.commit()
        report = {'database':str(args.db),'raw_records':conn.execute('SELECT count(*) FROM raw_records').fetchone()[0],
                  'status':dict(conn.execute('SELECT status,count(*) FROM raw_records GROUP BY status')),
                  'games':conn.execute('SELECT count(*) FROM games').fetchone()[0],
                  'splits':dict(conn.execute('SELECT split,count(*) FROM games GROUP BY split')),
                  'plies':conn.execute('SELECT sum(ply_count) FROM games').fetchone()[0]}
        (args.db.parent/'import_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        conn.close()


if __name__ == '__main__': main()

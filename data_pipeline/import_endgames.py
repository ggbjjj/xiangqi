"""Build a separate, auditable endgame SQLite and its full-mainline histogram.

    python -m data_pipeline.import_endgames
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata

from .cbl import DECODER_VERSION, DecodedRecord, decode_record, validate_mainline
from .database import connect
from .import_records import cbl_records
from .merge_cbl import identity, length_stats
from .notation import validate_candidate
from project_paths import ROOT, DATA_DB, ENDGAME_DB


VERSION = 'xiangqi-endgames-v1'
PGN_VERSION = 'ccpd-pgn-mainline-v1'
TRANS = str.maketrans({'車':'车','俥':'车','馬':'马','傌':'马','帥':'帅','將':'将',
    '進':'进','後':'后','砲':'炮','一':'1','二':'2','三':'3','四':'4','五':'5',
    '六':'6','七':'7','八':'8','九':'9'})
MOVE = re.compile(r'[车马炮相象士仕帅将兵卒前后中][1-9车马炮相象士仕帅将兵卒][进退平][1-9]|[a-iA-I][0-9][-:]?[a-iA-I][0-9]')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_pgn(payload):
    for encoding in ('utf-8-sig', 'big5hkscs', 'cp950', 'gb18030'):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('unsupported PGN encoding')
    tags = dict(re.findall(r'^\[([A-Za-z]+)\s+"(.*)"\]\s*$', text, re.M))
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('['))
    text = re.sub(r';[^\n]*', '', text)
    body = []
    brace = paren = 0
    for char in text:
        if char == '{': brace += 1; continue
        if char == '}' and brace: brace -= 1; continue
        if brace: continue
        if char == '(': paren += 1; continue
        if char == ')' and paren: paren -= 1; continue
        if not paren: body.append(char)
    if brace or paren:
        raise ValueError('unclosed PGN comment/variation')
    body = unicodedata.normalize('NFKC', ''.join(body)).translate(TRANS)
    moves = MOVE.findall(body)
    residue = MOVE.sub(' ', body)
    residue = re.sub(r'(?<!\d)(?:1/2-1/2|1-0|0-1|\*)', ' ', residue)
    residue = re.sub(r'\b\d+\.{1,3}', ' ', residue)
    residue = re.sub(r'[\s.!?+\-#$0-9]+', '', residue)
    if residue:
        raise ValueError(f'unparsed PGN mainline: {residue[:80]}')
    result = tags.get('Result', '*')
    if result not in ('1-0', '0-1', '1/2-1/2', '*'):
        raise ValueError(f'unknown PGN result: {result}')
    candidate = {'start_fen': tags.get('FEN', ''), 'moves_zh_normalized': moves}
    _, fen, actions, error = validate_candidate(candidate)
    if error:
        raise ValueError(error)
    return DecodedRecord(fen, actions, result, dict(tags, encoding=encoding, decoder=PGN_VERSION))


def populate(conn, records, existing):
    """Retain original bytes/outcomes; deduplicate only valid full mainlines."""
    groups = defaultdict(list)
    for item in records:
        raw_id = digest(f'{item["source_id"]}:{item["source_path"]}:{item["index"]}'.encode())
        decoded = None
        try:
            decoded = decode_record(item['payload']) if item['format'] == 'cbl21' else decode_pgn(item['payload'])
            validate_mainline(decoded)
            key = identity(decoded.start_fen, decoded.actions)[0]
            status, error = 'validated', None
            groups[key].append((raw_id, item, decoded))
        except ValueError as exc:
            key = None
            error = str(exc)
            status = 'excluded_empty' if decoded is not None and not decoded.actions else 'rejected'
        conn.execute('INSERT INTO raw_records VALUES (?,?,?,?,?,?,?,?,?,NULL)',
            (raw_id, item['source_id'], item['source_path'], item['index'], item['payload'],
             item['format'], digest(item['payload']), status, error))
        conn.execute('INSERT INTO decoded_records VALUES (?,?,?,?,?,?,?,?)',
            (raw_id, DECODER_VERSION if item['format'] == 'cbl21' else PGN_VERSION,
             decoded.start_fen if decoded else None, dumps(decoded.actions) if decoded else None,
             len(decoded.actions) if decoded else None, decoded.result if decoded else None,
             dumps(decoded.metadata) if decoded else '{}', error))
    for key, items in groups.items():
        _, _, record = items[0]
        known = {r.result for _, _, r in items} - {'*'}
        conflict = len(known) > 1
        result = '*' if conflict or not known else next(iter(known))
        _, group, split = identity(record.start_fen, record.actions)
        old = existing.get(key)
        if old:
            group, split = old['split_group'], old['split']
        metadata = dict(record.metadata, result_conflict=conflict,
            result_labels=sorted({r.result for _, _, r in items}),
            source_ids=sorted({item['source_id'] for _, item, _ in items}),
            source_records=[raw_id for raw_id, _, _ in items], overlaps_book_database=bool(old))
        conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
            (key, record.start_fen, dumps(record.actions), len(record.actions), result,
             'endgame_study', split, group, dumps(metadata)))
        conn.executemany('UPDATE raw_records SET game_id=? WHERE raw_id=?',
            [(key, raw_id) for raw_id, _, _ in items])


def distribution(conn, bin_width):
    lengths = [r[0] for r in conn.execute('SELECT ply_count FROM games')]
    stats = length_stats(lengths)
    stats['min'] = min(lengths)
    bins = [{'low': low, 'high': low+bin_width-1,
             'count': sum(low <= n < low+bin_width for n in lengths)}
            for low in range(1, max(lengths)+1, bin_width)]
    for item in bins: item['percent'] = item['count']/len(lengths)*100
    assert sum(item['count'] for item in bins) == len(lengths)
    return {'unit': '单步（ply）；红黑各走一次为 2 步 = 1 回合',
        'scope': 'games 表内去重后的完整合法主线；未训练切段、未补零',
        'bin_width': bin_width, 'stats': stats, 'bins': bins,
        'raw_records': conn.execute('SELECT count(*) FROM raw_records').fetchone()[0],
        'raw_status': dict(conn.execute('SELECT status,count(*) FROM raw_records GROUP BY status')),
        'results': dict(conn.execute('SELECT result,count(*) FROM games GROUP BY result')),
        'splits': dict(conn.execute('SELECT split,count(*) FROM games GROUP BY split')),
        'unique_start_positions': conn.execute('SELECT count(DISTINCT start_fen) FROM games').fetchone()[0],
        'overlaps_book_database': conn.execute("SELECT count(*) FROM games WHERE json_extract(metadata_json,'$.overlaps_book_database')=1").fetchone()[0],
        'result_conflicts': conn.execute("SELECT count(*) FROM games WHERE json_extract(metadata_json,'$.result_conflict')=1").fetchone()[0]}


def write_report(report, output, plot=True):
    output.mkdir(parents=True, exist_ok=True)
    (output/'步数区间分布.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    s = report['stats']
    lines = ['# 独立残局库步数统计', '', f"数据库：{report['database']}", '',
        report['unit'], report['scope'], '',
        f"有效去重残局 {s['count']:,} 局；最短 {s['min']} 步，最长 {s['max']} 步，平均 {s['mean']:.2f} 步，中位数 {s['median']:g} 步。", '',
        '| 步数区间 | 局数 | 占比 |', '|---|---:|---:|']
    for item in report['bins']:
        lines.append(f"| {item['low']}–{item['high']} | {item['count']:,} | {item['percent']:.2f}% |")
    lines += ['', f"原始记录 {report['raw_records']:,} 条；状态：{report['raw_status']}。",
        f"结果：{report['results']}（1-0 红胜、0-1 黑胜、1/2-1/2 和棋、* 未知/冲突）。",
        f"与原棋谱库精确重合 {report['overlaps_book_database']} 局，保留在本库并标记；两库联合读取时需去重。",
        f"数据划分：{report['splits']}。原棋谱库重合局沿用原划分，其余沿用项目的起始 FEN + 前 24 步分组算法。",
        '原始 CBL 文件、PGN 原文、逐局来源和原始胜负均存入本 SQLite。无效/空谱保留在原始表，不进入 games。',
        '未知或冲突结果仅参与策略监督，现有读取器会忽略对应价值标签。主线走法校验不证明最优解或长将长捉裁定。']
    (output/'统计说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if not plot: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, max(5.8, 3.6+len(report['bins'])*.48)), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    y = list(range(len(report['bins'])))
    counts = [item['count'] for item in report['bins']]
    ax.barh(y, counts, color='#29877C', height=.66, zorder=3)
    for i, item in enumerate(report['bins']):
        ax.text(item['count']+max(counts)*.02, i, f"{item['count']:,} 局  ·  {item['percent']:.2f}%",
            va='center', fontsize=12, color='#21334A')
    ax.set_yticks(y, [f"{item['low']}–{item['high']}" for item in report['bins']], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts)*1.3)
    ax.set_xlabel('残局局数', fontsize=12, labelpad=12)
    ax.set_ylabel(f"完整主线步数（每区间 {report['bin_width']} 步）", fontsize=12, labelpad=12)
    ax.grid(axis='x', alpha=.16, zorder=0)
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle('独立残局库 · 步数区间分布', x=.13, y=.965, ha='left', fontsize=24, fontweight='bold', color='#21334A')
    fig.text(.13, .887, f"去重后 {s['count']:,} 局  |  平均 {s['mean']:.2f} 步  |  中位数 {s['median']:g} 步  |  最长 {s['max']} 步", fontsize=13, color='#52657A')
    fig.text(.13, .045, '一步 = 一方走一次；区间包含两端。完整原谱主线，未训练切段、未补零。', fontsize=11, color='#52657A')
    fig.subplots_adjust(left=.13, right=.96, top=.80, bottom=.22)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'残局步数区间分布.{suffix}', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def build(args):
    source, target = args.book_db.resolve(), args.db.resolve()
    if source == target: raise ValueError('残局目标库不能覆盖原棋谱库')
    # All reads from the existing book database explicitly use mode=ro.
    old = connect(source, readonly=True)
    try:
        existing = {r['game_id']: dict(r) for r in old.execute('SELECT game_id,split_group,split FROM games')}
        ccpd = [dict(r) for r in old.execute("SELECT * FROM raw_records WHERE source_id='CCPD' AND source_path LIKE 'Dataset/殘局/%' ORDER BY source_path,record_index")]
        ccpd_source = tuple(old.execute("SELECT * FROM sources WHERE source_id='CCPD'").fetchone())
    finally: old.close()
    if not ccpd: raise ValueError('原棋谱库没有 CCPD 残局原文')
    manifest = json.loads((args.survey/'cnvcs-survey.json').read_text(encoding='utf-8'))
    sources = []
    records = []
    for book in manifest['books']:
        name = Path(book['url']).name
        blob = (args.survey/'raw'/name).read_bytes()
        if digest(blob) != book['sha256']: raise ValueError(f'CBL 校验和不同：{name}')
        path = 'cnvcs_canju/'+name
        sources.append((path, 'CNVCS', digest(blob), blob))
        records.extend({'source_id':'CNVCS','source_path':path,'index':i,'payload':payload,'format':'cbl21'}
            for i, payload in enumerate(cbl_records(blob)))
    for row in ccpd:
        sources.append(('ccpd/'+row['source_path'], 'CCPD', row['sha256'], row['payload']))
        records.append({'source_id':'CCPD','source_path':row['source_path'],'index':row['record_index'],
                        'payload':row['payload'],'format':'pgn'})
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        current = connect(target, readonly=True)
        try:
            kind = current.execute("SELECT value FROM metadata WHERE key='dataset_kind'").fetchone()
            if kind is None or kind[0] != VERSION: raise ValueError('目标已存在且不是此工具管理的残局库')
            backup = ROOT/'backups/datasets'/f'endgames-before-{datetime.now():%Y%m%d-%H%M%S-%f}.sqlite3'
            backup.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(backup) as saved: current.backup(saved)
        finally: current.close()
    with tempfile.NamedTemporaryFile(prefix='endgames-', suffix='.sqlite3', dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    conn = connect(temporary)
    try:
        with conn:
            conn.executemany('INSERT INTO sources VALUES (?,?,?,?)',
                [ccpd_source, ('CNVCS', manifest['directory'], 'unspecified', dumps({'collections':len(manifest['books'])}))])
            conn.executemany('INSERT INTO source_files VALUES (?,?,?,?)', sources)
            populate(conn, records, existing)
            conn.executemany('INSERT INTO metadata VALUES (?,?)', [
                ('dataset_kind', VERSION), ('book_database', str(source)),
                ('encoding', 'absolute-red-black-v1; action=90*src+dst'),
                ('validation', 'full mainline piece legality + king safety; no tournament repetition adjudication'),
                ('built_at', datetime.now().isoformat(timespec='seconds'))])
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()
        report = distribution(conn, args.bin_width)
        report.update(database=str(target), backup=str(backup) if backup else None,
                      source_files=len(sources), cnvcs_collections=len(manifest['books']))
    except BaseException:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    conn.close()
    temporary.replace(target)
    write_report(report, args.output, not args.no_plot)
    (target.parent/'endgames_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book-db', type=Path, default=DATA_DB, help='只读提取 CCPD 原始残局及重合局划分')
    parser.add_argument('--survey', type=Path, default=ROOT/'pretrain_data/endgame_survey_20261029')
    parser.add_argument('--db', type=Path, default=ENDGAME_DB)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/endgames-20261029')
    parser.add_argument('--bin-width', type=int, default=32, help='完整主线单步区间宽度，默认 32')
    parser.add_argument('--report-only', action='store_true', help='仅从现有残局库重新统计；不需要原始下载文件')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if args.bin_width < 1: parser.error('--bin-width 必须为正整数')
    if args.report_only:
        conn = connect(args.db, readonly=True)
        try: report = distribution(conn, args.bin_width)
        finally: conn.close()
        report['database'] = str(args.db.resolve())
        write_report(report, args.output, not args.no_plot)
        (args.db.parent/'endgames_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

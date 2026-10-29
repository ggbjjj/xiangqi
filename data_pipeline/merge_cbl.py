"""Decode, validate and merge CBL mainlines, with a backup and length report.

    python -m data_pipeline.merge_cbl
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from .cbl import DECODER_VERSION, decode_record, validate_mainline
from .database import connect
from project_paths import DATA_DB, ROOT


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def identity(fen, actions):
    game_id = hashlib.sha256((fen+'\n'+dumps(actions)).encode()).hexdigest()
    group = hashlib.sha256(('haiku5-split-v1\n'+fen+'\n'+dumps(actions[:24])).encode()).hexdigest()
    bucket = int(group[:8], 16) % 1000
    return game_id, group, 'train' if bucket < 900 else 'validation' if bucket < 950 else 'test'


def length_stats(lengths):
    lengths = list(lengths)
    return {'count': len(lengths), 'total_plies': sum(lengths),
            'max': max(lengths) if lengths else None,
            'mean': statistics.mean(lengths) if lengths else None,
            'median': statistics.median(lengths) if lengths else None}


def database_stats(conn):
    return {split: length_stats(r[0] for r in conn.execute(
        'SELECT ply_count FROM games WHERE split=?', (split,)))
        for split in ('train', 'validation', 'test')}


def merge_cbl(conn, *, reprocess=False):
    """Merge once per decoder version. Callers back up persistent DBs first.

    Conflicting known outcomes disable the game's value label; every original
    outcome remains in decoded_records and the raw bytes. Existing splits stay.
    """
    before = database_stats(conn)
    rows = conn.execute('SELECT r.* FROM raw_records r LEFT JOIN decoded_records d '
        'ON r.raw_id=d.raw_id WHERE r.format=? AND (? OR d.raw_id IS NULL OR d.decoder_version!=?) '
        'ORDER BY r.source_path,r.record_index', ('cbl21', reprocess, DECODER_VERSION)).fetchall()
    prepared = []
    groups = defaultdict(list)
    for row in rows:
        record = None
        try:
            record = decode_record(row['payload'])
            validate_mainline(record)
            error, status = None, 'validated'
        except ValueError as exc:
            error = str(exc)
            status = 'excluded_empty' if record is not None and not record.actions else 'rejected'
        prepared.append((row, record, status, error))
        if status == 'validated':
            groups[identity(record.start_fen, record.actions)[0]].append((row, record))

    added = existing = 0
    conflicts = []
    with conn:
        for game_id, items in groups.items():
            row, record = items[0]
            old = conn.execute('SELECT * FROM games WHERE game_id=?', (game_id,)).fetchone()
            metadata = json.loads(old['metadata_json']) if old else dict(record.metadata,
                source_id=row['source_id'], source_path=row['source_path'], record_index=row['record_index'])
            labels = set(metadata.get('result_labels', []))
            labels.update(r.result for _, r in items)
            if old: labels.add(old['result'] or '*')
            known = labels - {'*', '', None}
            conflict = bool(metadata.get('result_conflict')) or len(known) > 1
            result = '*' if conflict or not known else next(iter(known))
            if conflict:
                metadata.update(result_conflict=True, result_labels=sorted(labels - {None}))
                conflicts.append(game_id)
            if old:
                existing += 1
                conn.execute('UPDATE games SET result=?,metadata_json=? WHERE game_id=?',
                    (result, dumps(metadata), game_id))
            else:
                _, group, split = identity(record.start_fen, record.actions)
                origin = 'book_study' if '/cnvcs_gupu/' in row['source_path'] else 'human_master'
                conn.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)',
                    (game_id, record.start_fen, dumps(record.actions), len(record.actions),
                     result, origin, split, group, dumps(metadata)))
                added += 1
        for row, record, status, error in prepared:
            conn.execute('INSERT OR REPLACE INTO decoded_records VALUES (?,?,?,?,?,?,?,?)',
                (row['raw_id'], DECODER_VERSION, record.start_fen if record else None,
                 dumps(record.actions) if record else None, len(record.actions) if record else None,
                 record.result if record else None, dumps(record.metadata) if record else '{}', error))
            game_id = identity(record.start_fen, record.actions)[0] if status == 'validated' else None
            conn.execute('UPDATE raw_records SET status=?,error=?,game_id=? WHERE raw_id=?',
                (status, error, game_id, row['raw_id']))
        conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('cbl_decoder', DECODER_VERSION))

    decoded = conn.execute('SELECT r.source_path,r.record_index,r.status,r.game_id,d.* '
        'FROM raw_records r JOIN decoded_records d ON r.raw_id=d.raw_id WHERE r.format=?', ('cbl21',)).fetchall()
    valid = [r for r in decoded if r['status'] == 'validated']
    books = defaultdict(lambda: {'records': 0, 'valid_records': 0, 'results': Counter(), 'lengths': []})
    for row in decoded:
        book = books[row['source_path']]
        book['records'] += 1
        book['valid_records'] += row['status'] == 'validated'
        book['results'][row['result'] or '*'] += 1
        if row['ply_count'] is not None: book['lengths'].append(row['ply_count'])
    for book in books.values(): book['length_stats'] = length_stats(book.pop('lengths'))
    return {'decoder': DECODER_VERSION, 'processed_this_run': len(rows),
        'added_games': added, 'matched_existing_games': existing,
        'duplicate_records_this_run': sum(len(v) for v in groups.values())-added,
        'result_conflicts_this_run': conflicts,
        'before': before, 'after': database_stats(conn),
        'cbl_all_mainlines': length_stats(r['ply_count'] for r in decoded if r['ply_count'] is not None),
        'cbl_valid_mainlines': length_stats(r['ply_count'] for r in valid),
        'cbl_unique_games': conn.execute('SELECT count(DISTINCT game_id) FROM raw_records '
                                       'WHERE format=? AND status=?', ('cbl21', 'validated')).fetchone()[0],
        'cbl_status': dict(Counter(r['status'] for r in decoded)),
        'cbl_original_results': dict(Counter(r['result'] or '*' for r in decoded)),
        'cbl_valid_results': dict(Counter(r['result'] for r in valid)),
        'cbl_unique_game_results': dict(conn.execute('SELECT g.result,count(*) FROM games g '
            'WHERE g.game_id IN (SELECT game_id FROM raw_records WHERE format=?) GROUP BY g.result', ('cbl21',))),
        'errors': [{'source_path': r['source_path'], 'record_index': r['record_index'],
                    'title': json.loads(r['metadata_json']).get('title'), 'error': r['validation_error']}
                   for r in decoded if r['validation_error']],
        'books': dict(books)}


def backup_database(path, destination):
    """SQLite's backup API produces a consistent snapshot, including WAL data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    source = connect(path, readonly=True)
    target = sqlite3.connect(destination)
    try: source.backup(target)
    finally: target.close(); source.close()


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12})
    groups = [('原训练集', report['before']['train']), ('CBL 全部主线', report['cbl_all_mainlines']),
              ('CBL 合格主线', report['cbl_valid_mainlines']), ('合并后训练集', report['after']['train'])]
    colors = ['#346CB0', '#D69B36', '#3D9A89']
    fig, ax = plt.subplots(figsize=(14, 7.8), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    x = np.arange(len(groups))
    for offset, key, label, color in zip((-.24, 0, .24), ('max', 'mean', 'median'),
                                        ('最长步数', '平均步数', '步数中位数'), colors):
        values = [stats[key] or 0 for _, stats in groups]
        bars = ax.bar(x+offset, values, .21, label=label, color=color, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:,.2f}'.rstrip('0').rstrip('.'),
                (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 7),
                textcoords='offset points', ha='center', fontsize=11, color='#21334A')
    ax.set_xticks(x, [f'{name}\nn = {stats["count"]:,}' for name, stats in groups])
    ax.set_ylabel('单步（红或黑走一次 = 1 步）', labelpad=14)
    ax.set_ylim(0, max((s['max'] or 0) for _, s in groups)*1.24)
    ax.grid(axis='y', alpha=.18, zorder=0)
    for side in ('top', 'right', 'left'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#CBD3DF')
    ax.tick_params(axis='both', length=0, pad=10)
    ax.legend(frameon=False, loc='upper right', ncol=3)
    fig.suptitle('棋谱完整主线步数统计', x=.07, y=.96, ha='left', fontsize=23, fontweight='bold', color='#21334A')
    fig.text(.07, .89, '统计原始主线长度；不包含分支变化，不受训练 T / 自弈 C 截断与补零影响。', color='#52657A')
    fig.text(.07, .075, f'CBL 共 {report["cbl_all_mainlines"]["count"]:,} 条：'
        f'{report["cbl_status"].get("validated", 0):,} 条合格，'
        f'{report["cbl_status"].get("excluded_empty", 0)} 条空谱，'
        f'{report["cbl_status"].get("rejected", 0)} 条主线错误。全部主线统计包含空谱的 0 步。', fontsize=11, color='#52657A')
    fig.text(.07, .035, 'CBL 合格主线按原始记录计数，可能重复；合并后的训练集已去重，并保留验证 / 测试划分。', fontsize=11, color='#52657A')
    fig.subplots_adjust(left=.08, right=.97, bottom=.2, top=.83)
    for suffix in ('png', 'svg'): fig.savefig(output.with_suffix('.'+suffix), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_reports(conn, report, output, *, include_plot=True):
    output.mkdir(parents=True, exist_ok=True)
    (output/'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# CBL 合并与步数统计', '', '一步指一方走一次，按原始完整主线统计，不计分支与补零。', '',
             '| 范围 | 局数 / 条数 | 最长步数 | 平均步数 | 中位数 |', '|---|---:|---:|---:|---:|']
    for name, stats in [('原训练集', report['before']['train']), ('CBL 全部主线（含空谱）', report['cbl_all_mainlines']),
                        ('CBL 合格主线（含重复）', report['cbl_valid_mainlines']), ('合并后训练集', report['after']['train'])]:
        maximum = str(stats['max']) if stats['max'] is not None else '—'
        mean = f'{stats["mean"]:.2f}' if stats['mean'] is not None else '—'
        median = f'{stats["median"]:g}' if stats['median'] is not None else '—'
        lines.append(f'| {name} | {stats["count"]:,} | {maximum} | {mean} | {median} |')
    lines += ['', f'本次新增 {report["added_games"]} 局；原始 CBL 胜负：{report["cbl_original_results"]}。',
              f'CBL 状态：{report["cbl_status"]}；胜负冲突的棋谱编号：{report["result_conflicts_this_run"]}。',
              f'合并前数据库备份：{report.get("backup", "本次没有待处理记录")}。', '',
              'decoded_records 保留逐条解码的局面、主线、原始结果与错误；raw_records / source_files 保留原始字节。',
              'games.result 中 1-0 为红胜，0-1 为黑胜，1/2-1/2 为和，* 为未知或冲突；未知结果只用于策略学习。',
              f'完整逐谱统计、{len(report["errors"])} 条隔离记录的位置及原因见 merge_report.json。']
    (output/'统计报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if include_plot: plot_report(report, output/'棋谱步数统计')
    summary = {'database': report.get('database', str(DATA_DB)),
        'raw_records': conn.execute('SELECT count(*) FROM raw_records').fetchone()[0],
        'status': dict(conn.execute('SELECT status,count(*) FROM raw_records GROUP BY status')),
        'games': conn.execute('SELECT count(*) FROM games').fetchone()[0],
        'splits': dict(conn.execute('SELECT split,count(*) FROM games GROUP BY split')),
        'plies': conn.execute('SELECT sum(ply_count) FROM games').fetchone()[0],
        'cbl_decoder': DECODER_VERSION, 'cbl_report': str(output/'merge_report.json')}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DATA_DB)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reprocess', action='store_true', help='重新解码此版本已处理的 CBL')
    parser.add_argument('--no-plot', action='store_true', help='只写 JSON/Markdown 报告，不依赖 matplotlib')
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = args.output or ROOT/'artifacts'/f'cbl-merge-{stamp}'
    backup = ROOT/'backups'/'datasets'/f'cbl-before-{stamp}'/'dataset.sqlite3'
    if not args.db.exists(): parser.error(f'数据库不存在：{args.db}')
    backup_database(args.db, backup)
    conn = connect(args.db)
    try:
        report = merge_cbl(conn, reprocess=args.reprocess)
        report.update(database=str(args.db.resolve()), backup=str(backup), generated_at=stamp,
                      ply_unit='one move by red or black; original full mainline, before truncation/padding')
        summary = write_reports(conn, report, output, include_plot=not args.no_plot)
        summary['database'] = str(args.db.resolve())
        (args.db.parent/'import_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.db.parent/'cbl_merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ('books', 'errors')}, ensure_ascii=False, indent=2))
        if not args.no_plot: print(f'统计图：{output/"棋谱步数统计.png"}')
    finally: conn.close()


if __name__ == '__main__': main()

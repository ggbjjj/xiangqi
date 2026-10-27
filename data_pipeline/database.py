import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sources (
 source_id TEXT PRIMARY KEY, url TEXT NOT NULL, license TEXT, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source_files (
 path TEXT PRIMARY KEY, source_id TEXT REFERENCES sources(source_id), sha256 TEXT NOT NULL,
 payload BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS games (
 game_id TEXT PRIMARY KEY, start_fen TEXT NOT NULL, moves_json TEXT NOT NULL,
 ply_count INTEGER NOT NULL CHECK(ply_count>0), result TEXT, origin_type TEXT NOT NULL,
 split TEXT NOT NULL CHECK(split IN ('train','validation','test')), split_group TEXT NOT NULL,
 metadata_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS games_split ON games(split, game_id);
CREATE TABLE IF NOT EXISTS raw_records (
 raw_id TEXT PRIMARY KEY, source_id TEXT REFERENCES sources(source_id), source_path TEXT NOT NULL,
 record_index INTEGER NOT NULL DEFAULT 0, payload BLOB NOT NULL, format TEXT NOT NULL,
 sha256 TEXT NOT NULL, status TEXT NOT NULL, error TEXT, game_id TEXT REFERENCES games(game_id),
 UNIQUE(source_id,source_path,record_index));
CREATE INDEX IF NOT EXISTS raw_status ON raw_records(status);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=60)
        conn.executescript(SCHEMA)
    conn.execute('PRAGMA foreign_keys=ON')
    conn.row_factory = sqlite3.Row
    return conn

"""Shared paths; XIANGQI_HOME can point to a writable deployment directory."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('XIANGQI_HOME', Path(__file__).resolve().parent)).resolve()
DATA_DB = ROOT / 'Data' / 'dataset.sqlite3'
MODEL_CATALOG = ROOT / 'ModelParameters'
LEGACY_MODEL_DB = MODEL_CATALOG / 'models.sqlite3'
HAIKU_MODEL_DB = MODEL_CATALOG / 'HaiKu' / 'models.sqlite3'
OPUS_MODEL_DB = MODEL_CATALOG / 'Opus' / 'models.sqlite3'
MODEL_DB = HAIKU_MODEL_DB  # Compatibility default for existing HaiKu training.


def model_family(name):
    key = name.lower().replace('-', '')
    if key in ('haiku', 'haiku5'): return 'HaiKu'
    if key in ('opus', 'opus5'): return 'Opus'
    raise ValueError(f'未知模型系列：{name}')


def family_model_db(family):
    return MODEL_CATALOG / model_family(family) / 'models.sqlite3'


def selfplay_db(family):
    return ROOT / 'Data' / 'SelfPlay' / model_family(family) / 'games.sqlite3'

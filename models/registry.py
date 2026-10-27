"""Extend this factory when a new model generation is added."""
from .HaiKu.config import Haiku5Config
from .Opus.config import OpusConfig

ARCHITECTURES = {'Haiku-5': Haiku5Config, 'Opus-5': OpusConfig}


def build_model(config):
    name = config.get('architecture')
    if name == 'Haiku-5':
        from .HaiKu.network import Haiku5
        return Haiku5(Haiku5Config(**config))
    if name == 'Opus-5':
        from .Opus.network import Opus5
        return Opus5(OpusConfig(**config))
    raise ValueError(f'未安装模型实现：{name}')

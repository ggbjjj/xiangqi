from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class OpusConfig:
    architecture: str = 'Opus-5'
    version: int = 1
    dim: int = 128
    heads: tuple = (16, 8, 4, 8, 4)
    context_windows: tuple = (10, 20, 30, 20, 30)
    cores: int = 5
    history_depth: int = 2
    spatial_depth: int = 2
    ffn_dim: int = 512
    cnn_channels: int = 64
    value_hidden: int = 64
    tbptt_steps: int = 16
    checkpoint_attention: bool = True
    encoding: str = 'absolute-red-black-v1'
    dropout: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, 'heads', tuple(self.heads))
        object.__setattr__(self, 'context_windows', tuple(self.context_windows))
        if self.architecture != 'Opus-5' or self.version != 1:
            raise ValueError('不支持的 Opus 架构版本')
        if self.cores != 5 or self.history_depth != 2 or self.spatial_depth != 2:
            raise ValueError('Opus 固定五个核心，每个核心历史与空间各两层')
        if len(self.heads) != 5 or len(self.context_windows) != 5:
            raise ValueError('Opus 的 heads 和 context_windows 必须分别有五项')
        if type(self.dim) is not int or self.dim < 4 or self.dim % 2:
            raise ValueError('dim 必须为不小于 4 的偶数')
        if any(type(h) is not int or h < 1 or self.dim % h for h in self.heads):
            raise ValueError('每个注意力头数必须为正整数并整除 dim')
        sizes = (*self.context_windows, self.ffn_dim, self.cnn_channels,
                 self.value_hidden, self.tbptt_steps)
        if any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError('窗口、前馈维度与梯度截断步数必须为正整数')
        if self.encoding != 'absolute-red-black-v1': raise ValueError('不支持的输入或动作编码')
        if type(self.dropout) not in (int,float) or not math.isfinite(self.dropout) or not 0 <= self.dropout <= 1:
            raise ValueError('dropout 必须是 [0,1] 内的有限概率')

    @property
    def context_plies(self): return self.context_windows

    def to_dict(self): return asdict(self)

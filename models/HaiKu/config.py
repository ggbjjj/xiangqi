from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Haiku5Config:
    architecture: str = 'Haiku-5'
    version: int = 1
    dim: int = 128
    heads: int = 4
    cores: int = 3
    history_depth: int = 2
    spatial_depth: int = 2
    ffn_dim: int = 512
    cnn_channels: int = 64
    context_plies: int = 120  # K/V history boards, including current; independent of sequence T.
    checkpoint_attention: bool = True
    encoding: str = 'absolute-red-black-v1'

    def __post_init__(self):
        if self.architecture != 'Haiku-5' or self.version != 1:
            raise ValueError('不支持的 Haiku-5 架构版本')
        if self.dim < 4 or self.dim % 2 or self.heads < 1 or self.dim % self.heads:
            raise ValueError('dim 必须为偶数并能被 heads 整除')
        if self.cores != 3 or self.history_depth != 2 or self.spatial_depth != 2:
            raise ValueError('Haiku-5 固定三份核心，每份历史/空间各两层')
        if min(self.ffn_dim,self.cnn_channels,self.context_plies) < 1:
            raise ValueError('配置尺寸必须为正数')
        if self.encoding != 'absolute-red-black-v1':
            raise ValueError('不支持的输入/动作编码')

    def to_dict(self):
        return asdict(self)

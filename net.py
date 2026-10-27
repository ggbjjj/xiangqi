"""Current network import; the old ResNet has been removed."""
from models.HaiKu.network import Haiku5
from models.HaiKu.config import Haiku5Config
from models.Opus.network import Opus5
from models.Opus.config import OpusConfig
__all__ = ['Haiku5','Haiku5Config','Opus5','OpusConfig']

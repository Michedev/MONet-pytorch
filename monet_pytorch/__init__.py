from math import prod

from monet_pytorch.model import Monet
import omegaconf

omegaconf.OmegaConf.register_new_resolver('prod', lambda *numbers: int(prod(float(x) for x in numbers)))
omegaconf.OmegaConf.register_new_resolver('sum', lambda *numbers: int(sum(float(x) for x in numbers)))


__all__ = ['Monet']
import importlib
from os import path as osp

from CosmicDawnSynergies.utils.misc import scandir
from CosmicDawnSynergies.utils.registry import ARCH_REGISTRY

__all__ = ['build_arch']

# automatically import all *_arch.py so their classes register themselves
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
_arch_modules = [importlib.import_module(f'CosmicDawnSynergies.archs.{name}') for name in arch_filenames]


def build_arch(opt, rngs):
    """Build network architecture from options.

    Args:
        opt (dict): arch options; must contain ``type`` (registry key).
        rngs (nnx.Rngs): PRNG streams for parameter init.
    """
    opt = dict(opt)
    arch_type = opt.pop('type')
    return ARCH_REGISTRY.get(arch_type)(**opt, rngs=rngs)

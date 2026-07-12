import importlib
from os import path as osp

from CosmicDawnSynergies.utils.misc import scandir
from CosmicDawnSynergies.utils.registry import DATA_REGISTRY

from .device_batcher import DeviceBatcher

__all__ = ['build_dataset', 'DeviceBatcher']

# automatically import all *_dataset.py so their classes register themselves
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'CosmicDawnSynergies.data.{name}') for name in dataset_filenames]


def build_dataset(opt):
    """Build dataset from options.

    Args:
        opt (dict): dataset options; must contain ``type`` (registry key).
    """
    dataset = DATA_REGISTRY.get(opt['type'])(opt)
    return dataset

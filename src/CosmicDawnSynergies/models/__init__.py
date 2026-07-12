import importlib
from os import path as osp

from CosmicDawnSynergies.utils.misc import scandir
from CosmicDawnSynergies.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically import all *_model.py so their classes register themselves
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
_model_modules = [importlib.import_module(f'CosmicDawnSynergies.models.{name}') for name in model_filenames]


def build_model(opt, in_dim=None):
    """Build a Model from options.

    Args:
        opt (dict): full options dict; ``model`` selects the registry entry.
        in_dim (int): input dimension inferred from the dataset.
    """
    model = MODEL_REGISTRY.get(opt.get('model', 'MLPModel'))(opt, in_dim=in_dim)
    return model

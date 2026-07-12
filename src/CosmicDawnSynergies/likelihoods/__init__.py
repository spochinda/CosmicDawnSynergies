import importlib
from os import path as osp

from CosmicDawnSynergies.utils.misc import scandir
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY

__all__ = ['build_likelihood']

# automatically import all *_likelihood.py so their classes register themselves
likelihood_folder = osp.dirname(osp.abspath(__file__))
likelihood_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(likelihood_folder)
                        if v.endswith('_likelihood.py')]
_likelihood_modules = [importlib.import_module(f'CosmicDawnSynergies.likelihoods.{name}')
                       for name in likelihood_filenames]


def build_likelihood(name, opt):
    """Build a likelihood module by registry name with its kwargs dict."""
    return LIKELIHOOD_REGISTRY.get(name)(opt)

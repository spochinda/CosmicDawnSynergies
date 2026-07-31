from importlib.metadata import version
import importlib
import os

try: 
    __version__ = version("CosmicDawnSynergies")
except:
    pass

import os
import importlib
import sys
import types

# basicsr==1.4.2 imports torchvision.transforms.functional_tensor, which was
# removed in torchvision>=0.17 (the functions moved into
# torchvision.transforms.functional). Shim it back in before basicsr gets
# imported (via model.py below) rather than hand-patching the installed
# package, so the fix is tracked here instead of living as an invisible,
# unreproducible site-packages edit.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    from torchvision.transforms import functional as _tv_functional
    _shim = types.ModuleType('torchvision.transforms.functional_tensor')
    _shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _shim

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# List all Python files in the directory
modules = [f[:-3] for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

# Import all modules in the current directory. Some (e.g. likelihood.py,
# which needs hera_pspec) depend on packages that require a special/manual
# install (see pyproject.toml and README) and aren't part of the default
# dependency set — skip those with a warning instead of making every user of
# this package (including plain emulator training) require them.
for module in modules:
    try:
        importlib.import_module(f'.{module}', package=__name__)
    except ImportError as e:
        import warnings
        warnings.warn(f"Skipping CosmicDawnSynergies.{module}: {e}")

# Import specific submodule
submodule = 'itamar'
submodule_dir = os.path.join(current_dir, submodule)
if os.path.isdir(submodule_dir):
    importlib.import_module(f'.{submodule}.radio_cutoff_calc', package=__name__)

__all__ = modules + [submodule]
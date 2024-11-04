from importlib.metadata import version
import importlib
import os

try: 
    __version__ = version("CosmicDawnSynergies")
except:
    pass

import os
import importlib

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# List all Python files in the directory
modules = [f[:-3] for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

# Import all modules in the current directory
for module in modules:
    importlib.import_module(f'.{module}', package=__name__)

# Import specific submodule
submodule = 'itamar'
submodule_dir = os.path.join(current_dir, submodule)
if os.path.isdir(submodule_dir):
    importlib.import_module(f'.{submodule}.radio_cutoff_calc', package=__name__)

__all__ = modules + [submodule]
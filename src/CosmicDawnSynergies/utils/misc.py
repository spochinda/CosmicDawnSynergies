import fnmatch
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from os import path as osp

import numpy as np
import scipy.interpolate as sip
import scipy.optimize as sop
import yaml


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def set_random_seed(seed):
    """Seed python and numpy RNGs (JAX keys are threaded explicitly)."""
    random.seed(seed)
    np.random.seed(seed)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + str(k) + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + str(k) + ': ' + str(v) + '\n'
    return msg


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find files of interest (basicsr-compatible)."""
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)
                if suffix is None or return_path.endswith(suffix):
                    yield return_path
            elif recursive and entry.is_dir():
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def make_emu_dirs(opt):
    """Make dirs for emulators."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('emulators_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if (path is None) or ('strict_load' in key) or ('pretrain' in key) or ('resume' in key) or ('param_key' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def copy_file(file, file_root):
    # copy the yml file to the experiment root, stamping generation time + cmd
    cmd = ' '.join(sys.argv)
    filename = osp.join(file_root, osp.basename(file))
    shutil.copyfile(file, filename)

    if filename.endswith(('.yml', '.yaml')):
        with open(filename, 'r+') as f:
            lines = f.readlines()
            lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
            f.seek(0)
            f.writelines(lines)


def include_patterns(*patterns):
    """shutil.copytree() ignore-parameter factory that *keeps* only matches."""
    def _ignore_patterns(path, all_names):
        keep = (name for pattern in patterns
                for name in fnmatch.filter(all_names, pattern))
        dir_names = (name for name in all_names if os.path.isdir(os.path.join(path, name)))
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns


def copy_directory(src, dst, keep_patterns=('*.yml',)):
    """Copy source directory into destination directory, keeping only files
    matching ``keep_patterns`` (directories are always traversed).

    Returns the full destination path ``dst/basename(src)``.
    """
    os.makedirs(dst, exist_ok=True)
    src_basename = os.path.basename(os.path.normpath(src))
    full_dst = os.path.join(dst, src_basename)

    print(f"Copying {src_basename} to {dst}")
    if os.path.exists(full_dst):
        print(f"Destination directory {full_dst} already exists. Skipping copy.")
    else:
        shutil.copytree(src, full_dst, ignore=include_patterns(*keep_patterns))
    return full_dst


def confidence_level(samples, weights=None, level=0.68, method="iso-probability"):
    assert level < 1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order] / np.sum(weights)
    # Compute inverse cumulative distribution function
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invcdf = sip.interp1d(CDF, S)
    if method == "iso-probability":
        # Find smallest interval
        distance = lambda a, level=level: invcdf(a + level) - invcdf(a)
        res = sop.minimize(distance, (1 - level) / 2, bounds=[(0, 1 - level)], method="Nelder-Mead")
        interval = np.array([invcdf(res.x[0]), invcdf(res.x[0] + level)])
    elif method == "lower-limit":
        # Get value from which we reach the desired level
        interval = invcdf(1 - level)
    elif method == "upper-limit":
        # Get value to which we reach the desired level
        interval = invcdf(level)
    else:
        assert False, method
    return interval

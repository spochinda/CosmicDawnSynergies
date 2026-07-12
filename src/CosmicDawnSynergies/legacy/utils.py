import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
import time
import shutil
import sys
import fnmatch
import numpy as np
import scipy.interpolate as sip
import scipy.optimize as sop

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only

def confidence_level(samples, weights=None, level=0.68, method="iso-probability"):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invcdf = sip.interp1d(CDF, S)
    if method=="iso-probability":
        # Find smallest interval
        distance = lambda a, level=level: invcdf(a+level)-invcdf(a)
        res = sop.minimize(distance, (1-level)/2, bounds=[(0,1-level)], method="Nelder-Mead")
        interval =  np.array([invcdf(res.x[0]), invcdf(res.x[0]+level)])
    elif method=="lower-limit":
        # Get value from which we reach the desired level
        interval = invcdf(1-level)
    elif method=="upper-limit":
        # Get value to which we reach the desired level
        interval = invcdf(level)
    else:
        assert False, method 
    return interval

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
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


@master_only
def copy_file(file, file_root):
    # copy the yml file to the experiment root
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
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        dir_names = (name for name in all_names if os.path.isdir(os.path.join(path, name)))
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns

@master_only
def copy_directory(src, dst, emulator=None):
    """Copy source directory into destination directory.

    Args:
        src (str): Source directory path to copy from.
        dst (str): Destination directory path to copy into.
        emulator (str): Optional emulator filename pattern to include (e.g., 'net_g_16.pth').
    """
    emulator = os.path.basename(emulator) if emulator is not None else None
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dst, exist_ok=True)

    # Create the full destination path with the source directory name
    src_basename = os.path.basename(src)
    full_dst = os.path.join(dst, src_basename)

    print(f"Copying {src_basename} to {dst}")

    if os.path.exists(full_dst):
        print(f"Destination directory {full_dst} already exists. Skipping copy.")
    else:
        shutil.copytree(src, full_dst, ignore=include_patterns(emulator, '*.yml'))
    return full_dst


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


@master_only
def make_emu_dirs(opt):
    """Make dirs for emulators."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('emulators_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items():
        if ('strict_load' in key) or ('pretrain_network' in key) or ('resume' in key) or ('param_key' in key):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def parse_emu_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # Single dataset configuration - no preprocessing needed since we handle it in dataset classes

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        emulators_root = opt['path'].get('emulators_root')
        if emulators_root is None:
            emulators_root = osp.join(root_path, 'trained_emulators')
        emulators_root = osp.join(emulators_root, opt['name'])

        opt['path']['emulators_root'] = emulators_root
        opt['path']['models'] = osp.join(emulators_root, 'models')
        opt['path']['training_states'] = osp.join(emulators_root, 'training_states')
        opt['path']['log'] = emulators_root
        opt['path']['visualization'] = osp.join(emulators_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = opt['path'].get('results_root')
        if results_root is None:
            results_root = osp.join(root_path, 'results')
        results_root = osp.join(results_root, opt['name'])

        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args


def parse_inference_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    opt['is_train'] = False

    return opt, args


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


if __name__ == '__main__':
    src = '/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/trained_emulators/debug_Delta21_power_spectrum_emulator'
    dst = '/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/inferences/_SDC3b_2/'
    copy_directory(src, dst, emulator='net_g_16.pth')
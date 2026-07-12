import argparse
import random
from os import path as osp

from .misc import set_random_seed, yaml_load

# Emulator options schema (optax-native, JAX branch). See options/emulators/*.yml:
#
#   name: <emulator name>
#   manual_seed: 51            # root seed: jax PRNG (init, shuffling) + numpy fallback
#   model: MLPModel            # MODEL_REGISTRY key
#   arch:                      # ARCH_REGISTRY key + kwargs (in_dim injected from data)
#     type: MLP
#     hidden_dim: 100
#     ...
#   dataset:                   # DATA_REGISTRY key + kwargs (unchanged from legacy)
#     type: BaseDataset
#     batch_size: 20000
#     ...
#   train:
#     total_iter: 3200
#     loss: mse                # mse | mae | huber
#     grad_clip_norm: 1.0
#     ema_decay: ~
#     warmup_iter: -1
#     optimizer:
#       name: adam             # any optax optimizer; weight_decay on non-adamw
#       weight_decay: 1.0e-4   # applied as coupled L2 (torch-Adam semantics)
#     schedule:
#       name: piecewise_constant
#       init_value: 1.0e-3
#       boundaries_and_scales: {60000: 1.0}
#   path: {pretrain: ~, resume_state: ~}
#   val: {val_freq: 256, save_best: true, val_start: 0}
#   logger: {print_freq: 256, save_checkpoint_freq: 2560, use_tb_logger: true}


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


def force_yml_updates(opt, entries):
    """Apply --force_yml overrides like train:ema_decay=0.999 (no new keys)."""
    for entry in entries:
        keys, value = entry.split('=')
        keys, value = keys.strip(), value.strip()
        value = _postprocess_yml_value(value)
        d = opt
        key_list = keys.split(':')
        for key in key_list[:-1]:
            d = d[key]
        if key_list[-1] not in d:
            raise KeyError(f'--force_yml cannot create new key: {keys}')
        d[key_list[-1]] = value


def parse_emu_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # random seed: root of all JAX PRNG streams for this run
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    if args.force_yml is not None:
        force_yml_updates(opt, args.force_yml)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    # paths
    opt.setdefault('path', {})
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        emulators_root = opt['path'].get('emulators_root')
        if emulators_root is None:
            emulators_root = osp.join(root_path, 'trained_emulators')
        emulators_root = osp.join(emulators_root, opt['name'])

        opt['path']['emulators_root'] = emulators_root
        opt['path']['checkpoints'] = osp.join(emulators_root, 'checkpoints')
        opt['path']['log'] = emulators_root

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

    return opt, args


def parse_inference_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    opt['is_train'] = False

    return opt, args

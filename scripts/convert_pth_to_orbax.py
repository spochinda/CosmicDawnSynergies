"""Convert a legacy torch emulator (net_g_*.pth) into a JAX/orbax emulator dir.

Requires torch (legacy-only dependency): pip install -e '.[convert]'

Usage:
    python scripts/convert_pth_to_orbax.py trained_emulators/Pk_SDC3b_MLP_minmax_extended \
        --opt trained_emulators/Pk_SDC3b_MLP_minmax_extended/Pk_minmax_extended_SDC3b.yml

Produces <src>_jax/ with the standard emulator layout (options yml in the new
schema, param_stats.json, checkpoints/best) and verifies torch-vs-JAX forward
parity on random inputs drawn from the param_stats ranges.
"""
import argparse
import glob
import json
import re
import sys
from collections import OrderedDict
from os import path as osp

import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', 'src')))
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

# torch activation name -> jax.nn name
_ACTIVATIONS = {
    'ReLU': 'relu', 'LeakyReLU': 'leaky_relu', 'GELU': 'gelu', 'Tanh': 'tanh',
    'Sigmoid': 'sigmoid', 'Softplus': 'softplus', 'SiLU': 'silu', 'ELU': 'elu',
}


def convert_options(legacy_opt, name):
    """Legacy emulator yml -> new (optax-native) schema, inference-relevant parts."""
    network_opt = legacy_opt['network_opt']
    out_activation = network_opt.get('out_activation')
    arch = OrderedDict([
        ('type', 'MLP'),
        ('hidden_dim', network_opt['hidden_dim']),
        ('n_hidden', network_opt['n_hidden']),
        ('out_dim', network_opt.get('out_dim', 1)),
        ('activation', _ACTIVATIONS[network_opt['activation']]),
        ('out_activation', _ACTIVATIONS[out_activation] if out_activation else None),
        ('init', 'torch_default'),
    ])

    dataset = OrderedDict(legacy_opt['dataset'])
    for torch_only in ('batch_size_per_gpu', 'num_worker_per_gpu'):
        dataset.pop(torch_only, None)

    return OrderedDict([
        ('name', name),
        ('manual_seed', legacy_opt.get('manual_seed', 0)),
        ('model', 'MLPModel'),
        ('arch', arch),
        ('dataset', dataset),
        ('converted_from', 'legacy torch .pth (scripts/convert_pth_to_orbax.py)'),
    ])


def torch_params_to_pure_dict(state_dict):
    """torch Sequential MLP state_dict -> nnx pure dict.

    torch Linear stores weight (out, in); nnx.Linear kernel is (in, out).
    """
    indices = sorted({int(m.group(1)) for k in state_dict
                      if (m := re.match(r'network\.(\d+)\.weight', k))})
    layers = {}
    for i, idx in enumerate(indices):
        layers[i] = {
            'kernel': state_dict[f'network.{idx}.weight'].numpy().T.copy(),
            'bias': state_dict[f'network.{idx}.bias'].numpy().copy(),
        }
    return {'layers': layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', help='Legacy emulator directory (contains models/*.pth)')
    parser.add_argument('--weights', default='net_g_latest.pth', help='Weights file in <src>/models/')
    parser.add_argument('--opt', default=None, help='Legacy options yml (default: sole yml in <src>)')
    parser.add_argument('--out', default=None, help='Output dir (default: <src>_jax)')
    parser.add_argument('--param-key', default='params', choices=['params', 'params_ema'])
    parser.add_argument('--atol', type=float, default=1e-5, help='Forward-parity tolerance')
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required for conversion: pip install -e '.[convert]'")

    src = args.src.rstrip('/')
    opt_path = args.opt
    if opt_path is None:
        ymls = glob.glob(osp.join(src, '*.yml'))
        if len(ymls) != 1:
            sys.exit(f'Expected exactly one yml in {src}, found {ymls}. Pass --opt explicitly.')
        opt_path = ymls[0]
    out_dir = args.out or (src + '_jax')

    from CosmicDawnSynergies.utils import ordered_yaml, yaml_load

    load_dict = torch.load(osp.join(src, 'models', args.weights), map_location='cpu', weights_only=False)
    state_dict = load_dict[args.param_key]
    param_stats = load_dict['param_stats']
    pure = torch_params_to_pure_dict(state_dict)
    in_dim = pure['layers'][0]['kernel'].shape[0]
    assert in_dim == len(param_stats), (
        f'in_dim from weights ({in_dim}) != param_stats entries ({len(param_stats)})')

    legacy_opt = yaml_load(opt_path)
    new_opt = convert_options(legacy_opt, name=legacy_opt['name'] + '_converted_jax')
    new_opt['arch']['in_dim'] = in_dim

    # write emulator dir: yml + param_stats.json + orbax checkpoints/best
    import os

    import jax
    import orbax.checkpoint as ocp
    import yaml

    os.makedirs(osp.join(out_dir, 'checkpoints'), exist_ok=True)
    with open(osp.join(out_dir, f"{new_opt['name']}.yml"), 'w') as f:
        yaml.dump(new_opt, f, Dumper=ordered_yaml()[1], default_flow_style=False, sort_keys=False)
    with open(osp.join(out_dir, 'param_stats.json'), 'w') as f:
        json.dump(param_stats, f, indent=2)

    pure = jax.tree.map(lambda a: np.asarray(a, dtype=np.float32), pure)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(osp.abspath(osp.join(out_dir, 'checkpoints', 'best')), {'params': pure}, force=True)
    ckptr.wait_until_finished()
    print(f'Wrote {out_dir}')

    # ---- Level-1 parity: torch vs JAX forward on inputs spanning param_stats
    from CosmicDawnSynergies.legacy.model import MLP as TorchMLP
    from CosmicDawnSynergies.models.mlp_model import MLPModel

    network_opt = dict(legacy_opt['network_opt'])
    network_opt['in_dim'] = in_dim
    torch_net = TorchMLP(**network_opt)
    torch_net.load_state_dict(state_dict)
    torch_net.eval()

    model = MLPModel.from_emulator_dir(out_dir)

    rng = np.random.default_rng(0)
    # normalized inputs: cover [-1.5, 1.5] to also probe slight extrapolation
    x = rng.uniform(-1.5, 1.5, size=(4096, in_dim)).astype(np.float32)
    with torch.no_grad():
        ref = torch_net(torch.from_numpy(x)).numpy()
    pred = model.predict(x)
    max_diff = np.max(np.abs(pred - ref))
    print(f'Forward parity: max |jax - torch| = {max_diff:.3e} over {x.shape} inputs')
    assert max_diff < args.atol, f'Parity FAILED (atol={args.atol})'
    print('Level-1 parity PASSED')


if __name__ == '__main__':
    main()

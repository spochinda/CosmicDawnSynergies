import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import torch
import anesthetic
import seaborn as sns
from CosmicDawnSynergies.utils import confidence_level

# SDC3b band-centre redshifts (used when computing xHI via emulator)
_XHI_Z_SDC3b = np.array([
    1420.0 / ((181.0 + 195.9) / 2) - 1,  # 6.5352
    1420.0 / ((166.0 + 180.9) / 2) - 1,  # 7.1868
    1420.0 / ((151.0 + 165.9) / 2) - 1,  # 7.9618
])


def _predict_xhi(emu_path, param_samples, xhi_z):
    """Apply xHI emulator to (N,4) param samples → (N, len(xhi_z)) predictions."""
    from CosmicDawnSynergies.model import MLPModel
    from CosmicDawnSynergies.utils import yaml_load

    emu_dir = osp.dirname(osp.dirname(emu_path))
    yml_path = glob.glob(osp.join(emu_dir, '*.yml'))[0]
    opt = yaml_load(yml_path)
    opt['is_train'] = False
    opt['dist'] = False
    opt['num_gpu'] = 0
    opt['network_opt']['in_dim'] = (len(opt['dataset']['data_dims'])
                                    + len(opt['dataset']['params_opt']['names']))

    model = MLPModel(opt)
    model.load_network(model.net_g, emu_path, strict=True, param_key='params')
    model.net_g.eval()

    ps = model.param_stats
    keys = list(ps.keys())
    mins = np.array([ps[k]['min'] for k in keys])
    maxs = np.array([ps[k]['max'] for k in keys])

    N = len(param_samples)
    xhi = np.zeros((N, len(xhi_z)))
    for iz, z in enumerate(xhi_z):
        inp = np.column_stack([np.full(N, z), param_samples])
        inp_norm = (inp - mins) / (maxs - mins) * 2 - 1
        with torch.no_grad():
            pred = (model.net_g(torch.from_numpy(inp_norm.astype(np.float32)))
                    .detach().cpu().numpy().squeeze())
        xhi[:, iz] = pred
    return xhi


def SDC3b_xHI_hist(path_chain, plot_name="SDC3b_xHI.png", xHI_columns=None,
                   xhi_emu_path=None, xhi_z=None, param_names=None, title=''):
    """
    Plot xHI posteriors for one or more chains.

    Parameters
    ----------
    path_chain   : list of chain root dirs (without '/run' suffix)
    plot_name    : output file path
    xHI_columns  : column names to read xHI from (auto-detected if None and no emu)
    xhi_emu_path : if provided, compute xHI via this emulator instead of chain columns
    xhi_z        : redshifts to evaluate (defaults to SDC3b band centres)
    param_names  : parameter columns to pass to emulator (defaults to SDC3b params)
    title        : plot title
    """
    if xhi_z is None:
        xhi_z = _XHI_Z_SDC3b
    if param_names is None:
        param_names = ['zeta_eff', 'zeta_exp', 'rmfp', 'Vc']

    samples = [anesthetic.read_chains(root=chain + '/run') for chain in path_chain]

    samples_prior = samples[0].prior()
    weights_prior = samples_prior.get_weights()

    # determine z label strings
    if xhi_emu_path is not None:
        z_strs = [f'{z:.2f}' for z in xhi_z]
    else:
        if xHI_columns is None:
            xHI_columns = [col[0] for col in samples[0].columns if 'xHI' in col[0]]
            print(f'Using default xHI columns: {xHI_columns}')
        z_strs = [col.split('_z')[1] for col in xHI_columns]

    n_z = len(xhi_z) if xhi_emu_path is not None else len(xHI_columns)

    fig, axes = plt.subplots(n_z, 1, figsize=(8, 4 * n_z), sharex=True,
                             gridspec_kw={'hspace': 0})
    if n_z == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 100)

    # prior panels
    if xhi_emu_path is not None:
        prior_params = samples_prior[param_names].values
        prior_xhi = _predict_xhi(xhi_emu_path, prior_params, xhi_z)
        prior_xhi_cols = [prior_xhi[:, i] for i in range(n_z)]
    else:
        prior_xhi_cols = [samples_prior[col].values for col in xHI_columns]

    for ax, vals in zip(axes, prior_xhi_cols):
        h, edges = np.histogram(vals, bins=bins, weights=weights_prior)
        ax.hist(edges[:-1], edges, weights=h, color='grey', alpha=0.5,
                label='Prior', density=True)

    # posterior overlays
    ccb = sns.color_palette('colorblind')

    for i, sample in enumerate(samples):
        color = matplotlib.colors.rgb2hex(ccb[i])
        weights = sample.get_weights()

        if xhi_emu_path is not None:
            param_vals = sample[param_names].values
            xhi_arr = _predict_xhi(xhi_emu_path, param_vals, xhi_z)
            xhi_cols = [xhi_arr[:, j] for j in range(n_z)]
        else:
            xhi_cols = [sample[col].values for col in xHI_columns]

        for ax, vals, z_str in zip(axes, xhi_cols, z_strs):
            mean = np.average(vals, weights=weights)
            q95 = confidence_level(samples=vals, weights=weights,
                                   level=0.95, method='iso-probability')
            h, edges = np.histogram(vals, bins=bins, weights=weights)

            lbl = (f'$\\mathrm{{xHI}}_{{z={z_str}}}'
                   f'\\approx{mean:.2f}'
                   f'^{{+{q95[1]-mean:.2f}}}_{{-{mean-q95[0]:.2f}}}$')

            ax.axvspan(q95[0], q95[1], color=color, alpha=0.2)
            ax.hist(edges[:-1], edges, weights=h, color=color, alpha=0.8,
                    density=True, label=lbl)

    for ax in axes:
        ax.legend(loc='upper left')
        ax.set_ylabel('PDF')

    axes[-1].set_xlabel(r'$x_\mathrm{HI}$')
    axes[0].set_title(title)
    axes[0].set_xlim(0, 1)

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()

    return fig, axes


if __name__ == '__main__':
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 18,
    })

    PS = '1'
    plot_name = f'inferences/SDC3b_PS{PS}/SDC3b_PS{PS}_xHI.png'
    fig, axes = SDC3b_xHI_hist(
        path_chain=[f'inferences/SDC3b_PS{PS}/LikelihoodSDC3b'],
        plot_name=plot_name,
        xhi_emu_path='trained_emulators/xHI_SDC3b_minmax/models/net_g_latest.pth',
        title=f'SDC3b PS{PS}',
    )

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
                   xhi_emu_path=None, xhi_z=None, param_names=None, title='',
                   truths=None, truth_ranges=None, chain_labels=None):
    """
    Plot xHI posteriors for one or more chains.

    Parameters
    ----------
    path_chain   : list of chain root dirs (without '/run' suffix)
    plot_name    : output file path
    xHI_columns  : column names to read xHI from (auto-detected per-chain if None
                   and no emu — different chains may label the same z band with
                   slightly different precision, e.g. 'xHI_z6.53' vs 'xHI_z6.54',
                   so each chain's own columns are matched by position, not name)
    xhi_emu_path : if provided, compute xHI via this emulator instead of chain columns.
                   Either a single path (applied to every chain) or a list with
                   one entry per path_chain (use None for a chain that should
                   read its own stored xHI_z* columns instead).
    xhi_z        : redshifts to evaluate (defaults to SDC3b band centres)
    param_names  : parameter columns to pass to emulator (defaults to SDC3b params)
    title        : plot title
    truths       : optional list of true xHI values, one per panel, in the same
                   order as the auto-detected/passed z bands. Drawn as a vertical
                   line on each panel.
    truth_ranges : optional list of (lo, hi) pairs, one per panel, in the same
                   order as `truths`. Each band spans a finite bandwidth, so the
                   true simulated xHI is not constant across it but evolves from
                   a value at the band's high-z edge to a value at its low-z
                   edge; this is that evolution range, drawn as a grey span.
    chain_labels : optional list of labels, one per path_chain entry, prefixed
                   onto each chain's legend entries (e.g. ['Legacy', 'New']).
    """
    if xhi_z is None:
        xhi_z = _XHI_Z_SDC3b
    if param_names is None:
        param_names = ['zeta_eff', 'zeta_exp', 'rmfp', 'Vc']

    samples = [anesthetic.read_chains(root=chain + '/run') for chain in path_chain]

    # xhi_emu_path may be a single path (applied to every chain), a list (one
    # entry per chain, None meaning "use that chain's own stored columns"),
    # or None (use columns for every chain)
    if xhi_emu_path is None or isinstance(xhi_emu_path, str):
        emu_paths = [xhi_emu_path] * len(samples)
    else:
        emu_paths = list(xhi_emu_path)
        assert len(emu_paths) == len(samples), \
            "xhi_emu_path list must have one entry per path_chain"

    samples_prior = samples[0].prior()
    weights_prior = samples_prior.get_weights()

    def _detect_xhi_cols(sample):
        return [col[0] for col in sample.columns if 'xHI' in col[0]]

    # determine z label strings (and this chain's column list, for the prior panel)
    if emu_paths[0] is not None:
        z_strs = [f'{z:.2f}' for z in xhi_z]
    else:
        if xHI_columns is None:
            first_cols = _detect_xhi_cols(samples[0])
            print(f'Using default xHI columns: {first_cols}')
        else:
            first_cols = xHI_columns
        z_strs = [col.split('_z')[1] for col in first_cols]

    n_z = len(xhi_z)

    fig, axes = plt.subplots(n_z, 1, figsize=(8, 4 * n_z), sharex=True,
                             gridspec_kw={'hspace': 0})
    if n_z == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 100)

    # prior panels (from chain 0's source)
    if emu_paths[0] is not None:
        prior_params = samples_prior[param_names].values
        prior_xhi = _predict_xhi(emu_paths[0], prior_params, xhi_z)
        prior_xhi_cols = [prior_xhi[:, i] for i in range(n_z)]
    else:
        prior_xhi_cols = [samples_prior[col].values for col in first_cols]

    for ax, vals in zip(axes, prior_xhi_cols):
        h, edges = np.histogram(vals, bins=bins, weights=weights_prior)
        ax.hist(edges[:-1], edges, weights=h, color='grey', alpha=0.5,
                label='Prior', density=True)

    # posterior overlays
    ccb = sns.color_palette('colorblind')

    for i, sample in enumerate(samples):
        color = matplotlib.colors.rgb2hex(ccb[i])
        weights = sample.get_weights()

        if emu_paths[i] is not None:
            param_vals = sample[param_names].values
            xhi_arr = _predict_xhi(emu_paths[i], param_vals, xhi_z)
            xhi_cols = [xhi_arr[:, j] for j in range(n_z)]
        else:
            # match this chain's own xHI columns by position (not name) since
            # different chains may label the same z band with slightly
            # different precision, e.g. 'xHI_z6.53' vs 'xHI_z6.54'
            sample_cols = xHI_columns if xHI_columns is not None else _detect_xhi_cols(sample)
            xhi_cols = [sample[col].values for col in sample_cols]

        prefix = f'{chain_labels[i]}: ' if chain_labels is not None else ''

        for ax, vals, z_str in zip(axes, xhi_cols, z_strs):
            mean = np.average(vals, weights=weights)
            q95 = confidence_level(samples=vals, weights=weights,
                                   level=0.95, method='iso-probability')
            h, edges = np.histogram(vals, bins=bins, weights=weights)

            lbl = (f'{prefix}$x_{{\\mathrm{{HI}}, z={z_str}}}'
                   f'\\approx{mean:.2f}'
                   f'^{{+{q95[1]-mean:.2f}}}_{{-{mean-q95[0]:.2f}}}$')

            # ax.axvspan(q95[0], q95[1], color=color, alpha=0.2)

            ax.hist(edges[:-1], edges, weights=h, color=color, alpha=0.8,
                    density=True, label=lbl)

    if truth_ranges is not None:
        for ax, rng in zip(axes, truth_ranges):
            if rng is not None:
                ax.axvspan(min(rng), max(rng), color='grey', alpha=0.3,
                          zorder=0, label='True evolution range')

    if truths is not None:
        for ax, truth in zip(axes, truths):
            if truth is not None:
                ax.axvline(truth, color='k', linestyle='--', linewidth=1.5,
                          label=f'Truth $={truth:.2f}$')

    for ax in axes:
        ax.legend(loc='upper center', fontsize=plt.rcParams['font.size'] * 0.8)
        ax.set_ylabel('PDF')

    axes[-1].set_xlabel(r'$x_\mathrm{HI}$')
    axes[0].set_title(title)
    axes[0].set_xlim(0, 1)

    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()

    return fig, axes


if __name__ == '__main__':
    import argparse, re

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 18,
    })

    from sdc3b_xhi_truth import xhi_true, xhi_true_min_max

    # best xHI emulator from the seed scan (seed 34, avg_KL=1.21) — a chain's
    # own stored xHI_z* columns were computed during inference with whichever
    # emulator_xHI was configured at the time, which does not match this
    # best-seed emulator; recompute instead of trusting those columns for any
    # non-legacy chain (see analysis/plots/make_comparison_plots.py).
    # xHI_SDC3b_minmax is trained by reproduce_sdc3b.sh from
    # options/emulators/xHI_SDC3b.yml (seed=34) and is byte-identical to the
    # scan's seed-34 output (trained_emulators/xHI_SDC3b_scan_0034), so we
    # point here rather than at the untracked scan_best directory.
    BEST_XHI_EMU = 'trained_emulators/xHI_SDC3b_minmax/models/net_g_latest.pth'
    LEGACY_PREFIX = 'scripts/non-public/LikelihoodSDC3b_SDC3b_'

    parser = argparse.ArgumentParser()
    parser.add_argument('--PS', required=True, choices=['PS1', 'PS2'])
    parser.add_argument('--chains', nargs='+', default=None,
                        help='One or more chain root dirs (without "/run" suffix). '
                             'Defaults to the single legacy chain for --PS if omitted.')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='One label per --chains entry. Defaults to '
                             '"chain 1", "chain 2", ... if omitted.')
    parser.add_argument('--xhi_emu_path', nargs='+', default=None,
                        help='One entry per --chains: an emulator path, or the '
                             'literal "columns" to read that chain\'s own stored '
                             'xHI_z* columns instead. Defaults to the legacy '
                             'chain reading its own columns and every other '
                             'chain using the best-seed emulator.')
    args = parser.parse_args()

    PS = args.PS

    if args.chains is None:
        chain_dirs = [f'{LEGACY_PREFIX}{PS}']
        labels = ['Legacy']
    else:
        chain_dirs = args.chains
        if args.labels is not None:
            assert len(args.labels) == len(chain_dirs), \
                "--labels must have the same number of entries as --chains"
            labels = args.labels
        else:
            labels = [f'chain {i+1}' for i in range(len(chain_dirs))]

    if args.xhi_emu_path is not None:
        assert len(args.xhi_emu_path) == len(chain_dirs), \
            "--xhi_emu_path must have the same number of entries as --chains"
        emu_paths = [None if p.lower() == 'columns' else p for p in args.xhi_emu_path]
    else:
        emu_paths = [None if d.startswith(LEGACY_PREFIX) else BEST_XHI_EMU
                    for d in chain_dirs]

    # panel order is z~6.53, z~7.18, z~7.96; xhi_true's z1/z2/z3 run the
    # opposite way (z1 = highest z), see analysis/sdc3b_xhi_truth.py
    truths = [xhi_true['z3'][PS], xhi_true['z2'][PS], xhi_true['z1'][PS]]
    truth_ranges = [xhi_true_min_max['z3'][PS], xhi_true_min_max['z2'][PS],
                    xhi_true_min_max['z1'][PS]]

    if args.chains is None:
        plot_name = f'analysis/xHI_legacy_{PS}.png'
    else:
        slug = '_'.join(re.sub(r'\W+', '_', lbl.lower()).strip('_') for lbl in labels)
        plot_name = f'analysis/xHI_{PS}_{slug}.png'

    fig, axes = SDC3b_xHI_hist(
        path_chain=chain_dirs,
        chain_labels=labels if len(chain_dirs) > 1 else None,
        xhi_emu_path=emu_paths,
        plot_name=plot_name,
        title=f'SDC3b {PS}: Neutral fraction posteriors',
        truth_ranges=truth_ranges,
        truths=truths,
    )
    print(f"Saved → {plot_name}")

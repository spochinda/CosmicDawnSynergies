"""
Generate a paper-ready triangle plot for SDC3b chains, overlaying as many
chains as you like.

Usage:
    python analysis/plot_triangle_sdc3b.py --PS=PS1
    python analysis/plot_triangle_sdc3b.py --PS=PS1 \\
        --chains scripts/non-public/LikelihoodSDC3b_SDC3b_PS1 inferences/SDC3b_PS1/LikelihoodSDC3b \\
        --labels Legacy Reimplemented
    python analysis/plot_triangle_sdc3b.py --PS=PS1 \\
        --chains chain1 chain2 chain3 --labels lab1 lab2 lab3

With no --chains, defaults to the single legacy chain for the given PS, saved
to analysis/triangle_legacy_<PS>.png/.pdf (the filename referenced by the
paper). With --chains, saves to
analysis/triangle_<PS>_<label1>_vs_<label2>_..._.png/.pdf instead.
"""
import sys, os, argparse, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))

from anesthetic import read_chains, make_2d_axes

parser = argparse.ArgumentParser()
parser.add_argument('--PS', required=True, choices=['PS1', 'PS2'])
parser.add_argument('--chains', nargs='+', default=None,
                    help='One or more chain root dirs (without "/run" suffix), '
                         'relative to the repo root or absolute. Defaults to '
                         'the single legacy chain for --PS if omitted.')
parser.add_argument('--labels', nargs='+', default=None,
                    help='One label per --chains entry (same order/length). '
                         'Defaults to "chain 1", "chain 2", ... if omitted.')
args = parser.parse_args()

PS = args.PS

if args.chains is None:
    chain_dirs = [f'scripts/non-public/LikelihoodSDC3b_SDC3b_{PS}']
    labels = ['Legacy']
else:
    chain_dirs = args.chains
    if args.labels is not None:
        assert len(args.labels) == len(chain_dirs), \
            "--labels must have the same number of entries as --chains"
        labels = args.labels
    else:
        labels = [f'chain {i+1}' for i in range(len(chain_dirs))]


def _resolve(d):
    return d if os.path.isabs(d) else os.path.join(REPO, d)


CHAIN_ROOTS = [os.path.join(_resolve(d), 'run') for d in chain_dirs]

OUT_DIR = os.path.join(REPO, 'analysis')

PARAMS = ['zeta_eff', 'zeta_exp', 'rmfp', 'Vc', 'lognoise']
LABELS = {
    'zeta_eff':  r'$\zeta_\mathrm{eff}$',
    'zeta_exp':  r'$\zeta_\mathrm{exp}$',
    'rmfp':      r'$R_\mathrm{mfp}$ [Mpc]',
    'Vc':        r'$V_c$ [km/s]',
    'lognoise':  r'$\log_{10}\,\sigma_\mathrm{noise}$',
}
PRIOR_LIMS = {
    'zeta_eff':  (10.115, 99.860),
    'zeta_exp':  (0.5,    2.0),
    'rmfp':      (10.089, 99.988),
    'Vc':        (4.220,  99.732),
    'lognoise':  (-6.0,   -0.5),
}

COLORS = [matplotlib.colors.rgb2hex(c) for c in sns.color_palette('colorblind')]

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': 40,
    'axes.labelsize': 40,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
})

samples_list = []
for chain_root, label in zip(CHAIN_ROOTS, labels):
    print(f"Loading {label} chain from {chain_root}…")
    s = read_chains(chain_root)
    # legacy chains use 'noise' column name but same log scale — rename for plotting
    if 'noise' in s.columns and 'lognoise' not in s.columns:
        s = s.rename(columns={'noise': 'lognoise'})
    samples_list.append(s)

print("Generating triangle plot…")
fig, axes = make_2d_axes(PARAMS, labels=LABELS, upper=False, figsize=(13, 11))

for s, label, color in zip(samples_list, labels, COLORS):
    s.plot_2d(axes,
              kinds={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
              label=label, color=color, alpha=0.9)

# plot_2d re-applies each chain's own paramnames-derived labels, so force ours back on
axes.set_labels(LABELS, fontsize=22)

# make_2d_axes hardcodes tick labelsize='small' at creation; override it explicitly
axes.tick_params(axis='both', which='both', labelsize=16)

for i, p_row in enumerate(PARAMS):
    for j, p_col in enumerate(PARAMS):
        ax = axes.iloc[i, j]
        if ax is None:
            continue
        try:
            ax.set_xlim(*PRIOR_LIMS[p_col])
            if j != i:  # off-diagonal (lower and upper): y = p_row
                ax.set_ylim(*PRIOR_LIMS[p_row])
        except Exception:
            pass

title = f'SDC3b {PS}: Posterior on astrophysical parameters'
if len(samples_list) > 1:
    handles, labels_ = axes.iloc[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', fontsize=20, frameon=True,
              bbox_to_anchor=(0.98, 0.98), borderaxespad=0)
    # (chain labels are already shown in the legend, so left out of the title
    # to avoid the two colliding when there are several chains)
fig.suptitle(title, fontsize=22, y=0.9)
fig.tight_layout()

if args.chains is None:
    stem = f'triangle_legacy_{PS}'
else:
    slug = '_vs_'.join(re.sub(r'\W+', '_', lbl.lower()).strip('_') for lbl in labels)
    stem = f'triangle_{PS}_{slug}'

out_png = os.path.join(OUT_DIR, f'{stem}.png')
out_pdf = os.path.join(OUT_DIR, f'{stem}.pdf')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")

"""
Reproduce and cross-check the Cantabrigians team's SDC3b score (Tables 7 and 8,
Section 5.1.1 of the SDC3b paper) from the legacy inference chains.

Follows the exact methodology used to build the actual challenge submission
(CosmicDawnSynergies/scripts/analysis/SDC3b_posterior_cube.py): the xHI_z*
columns are sorted by *descending* redshift, so index 0/1/2 correspond to the
paper's z1/z2/z3 bins (z1 = highest z = 8.41-7.56, z2 = 7.56-6.85,
z3 = 6.85-6.25 = lowest z).

Metrics (Section 5.1.1):
    bias      = median(x) - x'                         (not printed, see Z)
    sigma     = Delta_68 / 2, Delta_68 = width of the 68% credible interval
    Z         = |median(x) - x'| / sigma
    S_SDC3b   = integral of the 3D posterior over the true evolution cuboid
                R = [a1',b1'] x [a2',b2'] x [a3',b3'], estimated two ways:
                  (a) binned exactly as the submitted FITS cube (100 bins/axis)
                  (b) unbinned: weighted fraction of posterior samples in R
    D_M       = Mahalanobis distance of the posterior median from x',
                using the (weighted) posterior covariance matrix C

Reproducibility note: of the metrics above, only S_SDC3b is exactly
reproducible from our own chains/cube, and it is — computed here three
independent ways (fresh binned histogram, unbinned sample fraction, and
directly from the actual submitted FITS cube), all agree with Tables 7/8 to
4 decimal places. Z and D_M come out close but not identical to the reported
values, for two different reasons:
  - Z depends on sigma = Delta_68/2, i.e. a 68% credible interval estimated
    from the posterior. The exact value is sensitive to how that interval is
    found (this script uses an iso-probability/shortest-interval search via
    CosmicDawnSynergies.utils.confidence_level); a different interval
    algorithm or binning gives a slightly different sigma and hence Z, even
    from the *same* underlying posterior.
  - D_M requires inverting the 3x3 covariance matrix C of (x1,x2,x3). Because
    x1/x2/x3 are strongly correlated (they all derive from the same 4
    astrophysical parameters — correlation coefficients ~0.6-0.96) and C is
    therefore moderately ill-conditioned (condition number ~1e2-4e2 here),
    small differences in how C is estimated get amplified substantially by
    the inversion. This is consistent with what we see: D_M is off by ~15%
    for PS1 (better-conditioned) and ~130% for PS2 (worse-conditioned),
    while S_SDC3b — which needs no matrix inversion — matches exactly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import anesthetic
from astropy.io import fits

from CosmicDawnSynergies.utils import confidence_level
from sdc3b_xhi_truth import xhi_true, xhi_true_min_max
from SDC3b_xHI import _predict_xhi, _XHI_Z_SDC3b

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CHAINS = {
    'PS1': 'scripts/non-public/LikelihoodSDC3b_SDC3b_PS1',
    'PS2': 'scripts/non-public/LikelihoodSDC3b_SDC3b_PS2',
}

# reimplemented/reproduced inference chains (see analysis/SDC3b_xHI.py __main__:
# their own stored xHI_z* columns use the wrong emulator seed, so xHI must be
# recomputed via the validated best-seed emulator instead)
REPRO_CHAINS = {
    'PS1': 'inferences/SDC3b_PS1/LikelihoodSDC3b',
    'PS2': 'inferences/SDC3b_PS2/LikelihoodSDC3b',
}
BEST_XHI_EMU = 'trained_emulators/xHI_SDC3b_scan_best/models/net_g_latest.pth'

# the actual FITS cubes submitted to the SDC3b organisers for scoring
SUBMITTED_FITS = {
    'PS1': '/Users/simonpochinda/venvs/cosmicdawn/lib/python3.12/site-packages/'
           'CosmicDawnSynergies/scripts/analysis/Cantabrigians_PS1.data.fits',
    'PS2': '/Users/simonpochinda/venvs/cosmicdawn/lib/python3.12/site-packages/'
           'CosmicDawnSynergies/scripts/analysis/Cantabrigians_PS2.data.fits',
}

# Tables 7 and 8 of the SDC3b paper, Cantabrigians row
REPORTED = {
    'PS1': {'Z_z1': 5.827, 'Z_z2': 3.96, 'Z_z3': 2.469, 'D_M': 8.362, 'S_SDC3b': 0.9995},
    'PS2': {'Z_z1': 0.092, 'Z_z2': 0.934, 'Z_z3': 0.27, 'D_M': 2.431, 'S_SDC3b': 1.0},
}


def get_sorted_xhi_columns(samples):
    """xHI columns sorted by descending redshift: index 0=z1 (highest z) ... 2=z3 (lowest z),
    matching the convention used to build the challenge submission cube."""
    cols = [col[0] for col in samples.columns if 'xHI' in col[0]]
    cols.sort(key=lambda c: float(c.split('_z')[-1]), reverse=True)
    return cols


def _score_from_xdf(x, weights, ps, n_bins=100):
    """Shared metric computation. x is a 3-column (weighted) DataFrame slice,
    ordered z1 (highest z) -> z3 (lowest z); weights is x's sample weights."""
    # true central values and evolution ranges, z1 (highest z) -> z3 (lowest z)
    x_true = np.array([xhi_true['z1'][ps], xhi_true['z2'][ps], xhi_true['z3'][ps]])
    ranges = [xhi_true_min_max['z1'][ps], xhi_true_min_max['z2'][ps], xhi_true_min_max['z3'][ps]]
    lo = np.array([min(r) for r in ranges])
    hi = np.array([max(r) for r in ranges])

    cols = list(x.columns.get_level_values(0)) if hasattr(x.columns, 'get_level_values') else list(x.columns)

    # --- per-parameter Z-scores ---
    x_median = x.median().values
    Z, sigma = {}, {}
    for i, key in enumerate(['z1', 'z2', 'z3']):
        d68 = confidence_level(samples=x[cols[i]].values, weights=weights,
                               level=0.68, method='iso-probability')
        sigma[key] = (d68[1] - d68[0]) / 2
        Z[key] = abs(x_median[i] - x_true[i]) / sigma[key]

    # --- SDC3b score: fraction of the posterior within cuboid R ---
    # (a) binned, exactly as the submitted FITS cube
    bins = np.linspace(0, 1, n_bins + 1)
    H, edges = np.histogramdd(x.values, bins=(bins, bins, bins), weights=weights)
    H = H / H.sum()
    centers = 0.5 * (bins[:-1] + bins[1:])
    in_range = [(centers >= lo[d]) & (centers <= hi[d]) for d in range(3)]
    mask = in_range[0][:, None, None] & in_range[1][None, :, None] & in_range[2][None, None, :]
    S_binned = H[mask].sum()

    # (b) unbinned: direct weighted fraction of samples inside R
    inside = np.all((x.values >= lo) & (x.values <= hi), axis=1)
    S_unbinned = weights[inside].sum() / weights.sum()

    # --- Mahalanobis distance ---
    C = x.cov().values
    diff = x_median - x_true
    D_M = float(np.sqrt(diff @ np.linalg.inv(C) @ diff))

    return {
        'Z_z1': Z['z1'], 'Z_z2': Z['z2'], 'Z_z3': Z['z3'],
        'D_M': D_M, 'S_SDC3b': S_binned, 'S_SDC3b_unbinned': S_unbinned,
        'x_median': x_median, 'x_true': x_true, 'sigma': sigma,
        'cols': cols,
    }


def compute_score(path_chain, ps, n_bins=100):
    """Score from a chain's own stored xHI_z* columns (the legacy chains)."""
    samples = anesthetic.read_chains(root=path_chain + '/run')
    weights = samples.get_weights()

    cols = get_sorted_xhi_columns(samples)
    assert len(cols) == 3, f"Expected 3 xHI columns, got {cols}"
    x = samples[cols]

    return _score_from_xdf(x, weights, ps, n_bins=n_bins)


def compute_score_reproduced(path_chain, ps, emu_path=BEST_XHI_EMU,
                             param_names=('zeta_eff', 'zeta_exp', 'rmfp', 'Vc'),
                             n_bins=100):
    """Score for a reimplemented/reproduced chain, recomputing xHI via the
    validated best-seed emulator applied to the posterior parameter samples —
    matching analysis/SDC3b_xHI.py and analysis/plots/make_comparison_plots.py,
    i.e. the xHI shown in legacy_vs_reproduced_xHI_PS{1,2}.png — rather than
    trusting the chain's own stored xHI_z* columns (wrong emulator seed)."""
    samples = anesthetic.read_chains(root=path_chain + '/run')
    weights = samples.get_weights()

    param_vals = samples[list(param_names)].values
    # z1 (highest z) -> z3 (lowest z), matching _score_from_xdf's convention
    xhi_z_desc = np.array(_XHI_Z_SDC3b[::-1])
    xhi_arr = _predict_xhi(emu_path, param_vals, xhi_z_desc)  # (N, 3), z1->z3

    cols = ['xHI_z1_emu', 'xHI_z2_emu', 'xHI_z3_emu']
    for i, c in enumerate(cols):
        samples[c] = xhi_arr[:, i]
    x = samples[cols]

    return _score_from_xdf(x, weights, ps, n_bins=n_bins)


def compute_S_from_fits(fits_path, ps):
    """S_SDC3b computed directly from the actual submitted posterior cube.
    Axis order is confirmed from the FITS header (AXIS1-FREQ z_c=7.96, AXIS2
    z_c=7.18, AXIS3 z_c=6.53), i.e. axis0=z1, axis1=z2, axis2=z3, and
    CRVAL/CDELT confirm 100 bins spanning [0,1] with centres at 0.005..0.995 —
    matching the (a) binned computation in compute_score exactly."""
    with fits.open(fits_path) as f:
        H = f[0].data.astype(np.float64)

    n_bins = H.shape[0]
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    ranges = [xhi_true_min_max['z1'][ps], xhi_true_min_max['z2'][ps], xhi_true_min_max['z3'][ps]]
    lo = np.array([min(r) for r in ranges])
    hi = np.array([max(r) for r in ranges])

    in_range = [(centers >= lo[d]) & (centers <= hi[d]) for d in range(3)]
    mask = in_range[0][:, None, None] & in_range[1][None, :, None] & in_range[2][None, None, :]
    return float(H[mask].sum())


if __name__ == '__main__':
    for ps, rel_path in CHAINS.items():
        result = compute_score(os.path.join(REPO, rel_path), ps)
        rep = REPORTED[ps]

        print(f"\n=== {ps} (columns, z1->z3: {result['cols']}) ===")
        print(f"median x = {np.round(result['x_median'], 4)}  "
              f"true x' = {np.round(result['x_true'], 4)}")
        print(f"{'metric':<12}{'computed':<12}{'reported':<12}{'diff':<10}")
        for key in ['Z_z1', 'Z_z2', 'Z_z3', 'D_M', 'S_SDC3b']:
            computed = result[key]
            reported = rep[key]
            print(f"{key:<12}{computed:<12.4f}{reported:<12.4f}{computed-reported:<+10.4f}")
        print(f"{'S (unbinned)':<12}{result['S_SDC3b_unbinned']:<12.4f}")

        if os.path.exists(SUBMITTED_FITS[ps]):
            S_fits = compute_S_from_fits(SUBMITTED_FITS[ps], ps)
            print(f"{'S (submitted FITS cube)':<24}{S_fits:<12.4f}"
                  f"{'(reported: '+str(rep['S_SDC3b'])+')'}")
        else:
            print(f"  [submitted FITS cube not found at {SUBMITTED_FITS[ps]}]")

    # --- reimplemented/reproduced chains, xHI via the best-seed emulator ---
    for ps, rel_path in REPRO_CHAINS.items():
        result = compute_score_reproduced(os.path.join(REPO, rel_path), ps)
        rep = REPORTED[ps]

        print(f"\n=== {ps} REPRODUCED (xHI via {os.path.basename(os.path.dirname(os.path.dirname(BEST_XHI_EMU)))}) ===")
        print(f"median x = {np.round(result['x_median'], 4)}  "
              f"true x' = {np.round(result['x_true'], 4)}  "
              f"(legacy reported S_SDC3b = {rep['S_SDC3b']})")
        print(f"{'metric':<12}{'computed':<12}")
        for key in ['Z_z1', 'Z_z2', 'Z_z3', 'D_M', 'S_SDC3b']:
            print(f"{key:<12}{result[key]:<12.4f}")
        print(f"{'S (unbinned)':<12}{result['S_SDC3b_unbinned']:<12.4f}")

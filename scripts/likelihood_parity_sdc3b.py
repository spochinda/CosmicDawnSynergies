"""Compare legacy torch LikelihoodSDC3b vs JAX LikelihoodSDC3b on random params."""
import sys
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/src')

# alias old import path for the frozen legacy module
import CosmicDawnSynergies.legacy.model as legacy_model
sys.modules['CosmicDawnSynergies.model'] = legacy_model
import CosmicDawnSynergies.legacy.likelihood as legacy_like

from CosmicDawnSynergies.likelihoods import build_likelihood
from CosmicDawnSynergies.utils import yaml_load

REPO = '/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies'
files = [
    [f'{REPO}/data/SDC3b/Pk_PS1_181.0_195.9.txt', f'{REPO}/data/SDC3b/Pk_PS_averaged_noise_181.0_195.9.txt', 181.0, 195.9],
    [f'{REPO}/data/SDC3b/Pk_PS1_166.0_180.9.txt', f'{REPO}/data/SDC3b/Pk_PS_averaged_noise_166.0_180.9.txt', 166.0, 180.9],
    [f'{REPO}/data/SDC3b/Pk_PS1_151.0_165.9.txt', f'{REPO}/data/SDC3b/Pk_PS_averaged_noise_151.0_165.9.txt', 151.0, 165.9],
]

# ---- legacy (torch)
legacy_kwargs = {
    'files': files,
    'kperp_file': f'{REPO}/data/SDC3b/bins_kper.txt',
    'kpar_file': f'{REPO}/data/SDC3b/bins_kpar.txt',
    'emulator': f'{REPO}/trained_emulators/Pk_SDC3b_MLP_minmax_extended/models/net_g_latest.pth',
    'emulator_xHI': f'{REPO}/trained_emulators/xHI_SDC3b_minmax/models/net_g_latest.pth',
    'lognoise': [-6.0, -0.5],
}
model_opt = yaml_load(f'{REPO}/trained_emulators/Pk_SDC3b_MLP_minmax_extended/Pk_minmax_extended_SDC3b.yml')
model_opt['is_train'] = False
model_opt['dist'] = False
model_opt['num_gpu'] = 0
model_opt['network_opt']['in_dim'] = 7
leg = legacy_like.LikelihoodSDC3b(legacy_kwargs, model_opt)

prior_dict = {}
for p in ['zeta_eff', 'zeta_exp', 'rmfp', 'Vc']:
    prior_dict[p] = [leg.model.param_stats[p]['min'], leg.model.param_stats[p]['max']]
prior_dict['lognoise'] = legacy_kwargs['lognoise']
leg.get_prior_indices(prior_dict)

# ---- JAX
jax_kwargs = dict(legacy_kwargs)
jax_kwargs['emulator'] = f'{REPO}/trained_emulators/Pk_SDC3b_MLP_minmax_extended_pth_jax'
jax_kwargs['emulator_xHI'] = f'{REPO}/trained_emulators/xHI_SDC3b_minmax_jax'
mod = build_likelihood('LikelihoodSDC3b', jax_kwargs)
loglik = jax.jit(mod.loglikelihood)
derived = jax.jit(mod.derived)

rng = np.random.default_rng(7)
names = list(prior_dict.keys())
lo = np.array([prior_dict[n][0] for n in names])
hi = np.array([prior_dict[n][1] for n in names])

max_dlogL, max_rel, max_dxhi = 0.0, 0.0, 0.0
for _ in range(50):
    theta = lo + (hi - lo) * rng.uniform(size=len(names))
    logL_leg, nDer = leg.computeLikelihood(theta)
    particle = {n: theta[i] for i, n in enumerate(names)}
    logL_jax = float(loglik(particle))
    d = derived(particle)
    xhi_jax = np.array([float(d[k]) for k in sorted(d)])
    xhi_leg = np.array(sorted_leg := nDer[1:])  # legacy: [logL, xHI...] in z order
    # legacy xHI order corresponds to bands; match by recomputing keys
    keys = [f'xHI_z{z:.2f}' for z in mod.z_xHI]
    xhi_jax_ordered = np.array([float(d[k]) for k in keys])
    dlogL = abs(logL_jax - logL_leg)
    rel = dlogL / abs(logL_leg)
    max_dlogL = max(max_dlogL, dlogL)
    max_rel = max(max_rel, rel)
    max_dxhi = max(max_dxhi, np.max(np.abs(xhi_jax_ordered - np.array(nDer[1:]))))

print(f'max |dlogL| = {max_dlogL:.4e}  (max rel {max_rel:.2e})')
print(f'max |dxHI|  = {max_dxhi:.4e}')
# rel tolerance reflects fp32 emulator forward noise (torch vs XLA reduction
# order, ~3e-5 rel on predictions) amplified through the Gaussian; the formula
# itself matches to 0.0 when fed identical predictions.
assert max_rel < 1e-3 and max_dxhi < 1e-4, 'LIKELIHOOD PARITY FAILED'
print('LIKELIHOOD PARITY PASSED')

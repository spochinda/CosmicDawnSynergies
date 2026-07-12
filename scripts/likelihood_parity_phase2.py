"""Compare legacy torch likelihoods vs JAX ports (Radio, XRB, SARAS3, HERA).

Requires torch + the legacy stack: pip install -e '.[convert]'
Points where the legacy code hit its guard values (-1e30) or produced
non-finite logL (log(0) tails, where the JAX port uses norm.logcdf) are
compared for agreement-of-rejection rather than value.
"""
import sys
from os import path as osp

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(REPO, 'src'))

import CosmicDawnSynergies.legacy.model as legacy_model
sys.modules['CosmicDawnSynergies.model'] = legacy_model
import CosmicDawnSynergies.legacy.likelihood as legacy_like

from CosmicDawnSynergies.likelihoods import build_likelihood
from CosmicDawnSynergies.utils import yaml_load


def legacy_module(cls_name, kwargs, yml, in_dim):
    model_opt = yaml_load(yml)
    model_opt['is_train'] = False
    model_opt['dist'] = False
    model_opt['num_gpu'] = 0
    model_opt['network_opt']['in_dim'] = in_dim
    return getattr(legacy_like, cls_name)(kwargs, model_opt)


def compare(name, leg, mod, prior_dict, n=40, seed=3, rel_tol=1e-3, stable_ref=None):
    """stable_ref(theta): legacy logL recomputed with scipy log_ndtr, used where
    the erf-based legacy code cancels to log(0) = -inf (P < ~1e-17 per point)."""
    loglik = jax.jit(mod.loglikelihood)
    names = list(prior_dict.keys())
    lo = np.array([prior_dict[k][0] for k in names])
    hi = np.array([prior_dict[k][1] for k in names])
    rng = np.random.default_rng(seed)

    max_rel, n_valid, n_stable = 0.0, 0, 0
    for _ in range(n):
        theta = lo + (hi - lo) * rng.uniform(size=len(names))
        out = leg.computeLikelihood(theta)
        logL_leg = out[0] if out is not None else np.nan
        logL_jax = float(loglik({k: theta[i] for i, k in enumerate(names)}))
        if not np.isfinite(logL_leg) or logL_leg <= -1e29:
            assert stable_ref is not None, f'{name}: legacy rejected but no stable_ref given'
            logL_leg = float(stable_ref(theta))
            n_stable += 1
            if logL_leg <= -1e29:  # genuine guard (e.g. XRB overflow): exact match required
                assert logL_jax <= -1e29, f'{name}: guard mismatch jax={logL_jax}'
                continue
        rel = abs(logL_jax - logL_leg) / max(abs(logL_leg), 1.0)
        max_rel = max(max_rel, rel)
        n_valid += 1
    status = 'PASSED' if max_rel < rel_tol else 'FAILED'
    print(f'{name:28s} max rel dlogL = {max_rel:.3e} over {n_valid} pts '
          f'({n_stable} via stable log_ndtr reference) {status}')
    assert max_rel < rel_tol, f'{name} parity FAILED'


def astro_prior(leg, n_data_dims):
    keys = list(leg.model.param_stats.keys())[n_data_dims:]
    return {k: [leg.model.param_stats[k]['min'], leg.model.param_stats[k]['max']] for k in keys}


# ---------------------------------------------------------------- Radio
from scipy.special import log_ndtr

kw = {'files': f'{REPO}/data/LWA1_ARCADE2/LWA1_with_err.npy',
      'emulator': f'{REPO}/trained_emulators/T_today_emulator/models/net_g_10000.pth'}
leg = legacy_module('LikelihoodRadioBackground', kw, f'{REPO}/trained_emulators/T_today_emulator/Radio.yml', 10)
prior = astro_prior(leg, 1)
leg.get_prior_indices(prior)
mod = build_likelihood('LikelihoodRadioBackground',
                       dict(kw, emulator=f'{REPO}/trained_emulators/T_today_emulator_jax'))


def radio_stable(theta):
    T_model = leg.predict(np.asarray(theta)[leg.prior_indices])
    t = (leg.T_obs - T_model) / np.sqrt(leg.dT_obs ** 2 + (0.05 * T_model) ** 2)
    return log_ndtr(t).sum()


compare('LikelihoodRadioBackground', leg, mod, prior, stable_ref=radio_stable)

# ---------------------------------------------------------------- XRB
import scipy.interpolate as sip

kw = {'emulator': f'{REPO}/trained_emulators/Xray_emulator/models/net_g_10000.pth'}
leg_xrb = legacy_module('LikelihoodXRB', kw, f'{REPO}/trained_emulators/Xray_emulator/Xray.yml', 10)
prior = astro_prior(leg_xrb, 1)
leg_xrb.get_prior_indices(prior)
mod = build_likelihood('LikelihoodXRB', dict(kw, emulator=f'{REPO}/trained_emulators/Xray_emulator_jax'))


def xrb_stable(theta):
    pred = leg_xrb.predict(np.asarray(theta)[leg_xrb.prior_indices])
    logpred_i = sip.interp1d(np.log10(leg_xrb.E_kev), np.log10(pred), kind='linear', fill_value='extrapolate')
    logL = 0.0
    for xmin, xmax, obs, std in leg_xrb.X_limits:
        E = np.geomspace(xmin, xmax, 100)
        logp = logpred_i(np.log10(E))
        if np.any(logp > 300) or np.any(~np.isfinite(10 ** logp)):
            return -1e30
        integral = np.trapezoid(10 ** logp, E * leg_xrb.keV_toHz) * leg_xrb.cm_toMpc ** 2 / leg_xrb.sr_todeg2
        if not np.isfinite(integral):
            return -1e30
        logL += log_ndtr((obs - integral) / np.sqrt(std ** 2 + (integral * 0.05) ** 2))
    return logL


compare('LikelihoodXRB', leg_xrb, mod, prior, stable_ref=xrb_stable)

# ---------------------------------------------------------------- SARAS3
poly = {'fg_a0': [3, 4], 'fg_a1': [-1, 1], 'fg_a2': [-0.1, 0.1], 'fg_a3': [-0.1, 0.1],
        'fg_a4': [-0.1, 0.1], 'fg_a5': [-0.1, 0.1], 'fg_a6': [-0.1, 0.1]}
kw = {'files': f'{REPO}/data/SARAS3/SARAS_3_averaged_spectrum.txt',
      'emulator': f'{REPO}/trained_emulators/T21_emulator/models/net_g_10000.pth',
      'poly_coeff': poly, 'lognoise': [-2, 0]}
leg = legacy_module('LikelihoodSARAS3', kw, f'{REPO}/trained_emulators/T21_emulator/T21.yml', 10)
prior = astro_prior(leg, 1)
prior.update(poly)
prior['lognoise'] = kw['lognoise']
leg.get_prior_indices(prior)
mod = build_likelihood('LikelihoodSARAS3', dict(kw, emulator=f'{REPO}/trained_emulators/T21_emulator_jax'))
compare('LikelihoodSARAS3', leg, mod, prior)

# ---------------------------------------------------------------- HERA
kw = {'files': [f'{REPO}/data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
                f'{REPO}/data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5'],
      'emulator': f'{REPO}/trained_emulators/Delta21_emulator2/models/net_g_400.pth',
      'decimate_data': False}
# Delta21_emulator2's yml predates the params_opt.normalization key (training
# default was norm_minmax); use the complete yml from Delta21_emulator instead
leg_h = legacy_module('LikelihoodHERA', kw, f'{REPO}/trained_emulators/Delta21_emulator/Delta21.yml', 11)
prior = astro_prior(leg_h, 2)
leg_h.get_prior_indices(prior)
mod = build_likelihood('LikelihoodHERA',
                       dict(kw, emulator=f'{REPO}/trained_emulators/Delta21_emulator2_jax'))


def hera_stable(theta):
    p = np.asarray(theta)[leg_h.prior_indices]
    logL = 0.0
    for band in leg_h.data.keys():
        d = leg_h.data[band]
        pred = d['wfn'] @ leg_h.predict(d['z'], d['k_mag'], p)
        t = (d['dsq'] - pred) / np.sqrt(d['std'] ** 2 + (0.2 * pred) ** 2)
        logL += log_ndtr(t).sum()
    return logL


compare('LikelihoodHERA', leg_h, mod, prior, stable_ref=hera_stable)

print('ALL PHASE-2 LIKELIHOOD PARITY CHECKS PASSED')

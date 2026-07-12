import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.constants import parsec, physical_constants

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register()
class LikelihoodXRB(BaseLikelihood):
    """X-ray background likelihood (Hickox & Markevitch 2006; Harrison+ 2016).

    The emulator predicts the spectrum on a fixed energy grid; each observed
    band integrates a log-log interpolation of that spectrum (jnp.interp +
    jnp.trapezoid replace scipy interp1d/trapezoid). Legacy early-return
    guards become a jnp.where(-1e30) penalty.
    """

    def extract_data(self):
        eV_toHz = physical_constants['electron volt-hertz relationship'][0]
        self.keV_toHz = eV_toHz * 1e3
        self.sr_todeg2 = (180 / np.pi) ** 2
        Mpc_tocm = 1e6 * parsec * 1e2
        self.cm_toMpc = 1 / Mpc_tocm

        # nu_min, nu_max, mean, std
        self.X_limits = np.array([
            [1, 2, 1.04e-12, 0.14e-12],                                  # Hickox & Markevitch (2006)
            [2, 8, 3.4e-12, 1.7e-12],                                    # Hickox & Markevitch (2006)
            [8, 24, 6.013e-8 / self.sr_todeg2, 0.14e-8 / self.sr_todeg2],   # Harrison et al. (2016)
            [20, 50, 6.56e-8 / self.sr_todeg2, 0.273e-8 / self.sr_todeg2],  # Harrison et al. (2016)
        ])

        minE, maxE = self.X_limits[:, 0].min(), self.X_limits[:, 1].max()
        E_kev = np.geomspace(minE, maxE, 100)
        self.block = jnp.asarray(E_kev)[:, None]
        self.logE_grid = jnp.asarray(np.log10(E_kev))

        # per-band static integration grids
        self.bands = []
        for xmin, xmax, obs, std in self.X_limits:
            E_band = np.geomspace(xmin, xmax, 100)
            self.bands.append({
                'logE': jnp.asarray(np.log10(E_band)),
                'E_Hz': jnp.asarray(E_band * self.keV_toHz),
                'obs': obs,
                'std': std,
            })

    def loglikelihood(self, particle):
        pred = self.predict(self.emulator_inputs(self.block, particle))
        logpred_grid = jnp.log10(pred)

        logL = 0.0
        bad = jnp.any(logpred_grid > 300) | jnp.any(~jnp.isfinite(logpred_grid))
        for band in self.bands:
            logpred = jnp.interp(band['logE'], self.logE_grid, logpred_grid)
            pred_band = 10 ** logpred
            integral = jnp.trapezoid(pred_band, band['E_Hz']) * self.cm_toMpc ** 2 / self.sr_todeg2
            bad = bad | ~jnp.isfinite(integral)
            sigma = jnp.sqrt(band['std'] ** 2 + (integral * 0.05) ** 2)
            logL += norm.logcdf((band['obs'] - integral) / sigma)
        return jnp.where(bad, -1e30, logL)

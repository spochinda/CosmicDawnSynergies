import jax.numpy as jnp
import numpy as np

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood, stats_arrays
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register()
class LikelihoodSARAS3(BaseLikelihood):
    """SARAS3 global-signal likelihood: log-polynomial foreground + emulated
    T21 with a 25% model-error floor and a sampled lognoise nuisance.

    Derived: the 5% quantile of the predicted T21 across the band (legacy
    nDerived output).
    """

    def extract_data(self):
        self.freq, self.T_SARAS, self.weights, self.fg_fit, self.fg_fit_T_resid = np.loadtxt(self.files).T
        log_freq = np.log10(self.freq)
        # foreground polynomial coordinate in [-1, 1]
        self.reduced_freq = jnp.asarray(2 * ((log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())) - 1)
        redshifts = 1420.0 / self.freq - 1

        log_z = self.model_opt['dataset']['data_dims']['z'].get('log', False)
        z_col = self.to_normed(redshifts, log_z)[:, None]
        dim_stats = stats_arrays(self.param_stats, list(self.param_stats.keys())[:self.n_data_dims])
        self.block_norm = jnp.asarray(np.asarray(self.normalize(jnp.asarray(z_col), dim_stats)))
        self.T_SARAS = jnp.asarray(self.T_SARAS)

        self.poly_names = list(self.opt['poly_coeff'].keys())
        self.astro_stats = stats_arrays(self.param_stats, self.astro_names)

    def _predict_T21(self, particle):
        theta_n = self.normalize(self.astro_vector(particle), self.astro_stats)
        inputs = jnp.concatenate(
            [self.block_norm, jnp.broadcast_to(theta_n, (self.block_norm.shape[0], theta_n.size))], axis=1)
        return self.unscale_target(self.model.net_g(inputs.astype(jnp.float32))) * 1e-3  # mK -> K

    def foreground(self, coeffs):
        powers = jnp.arange(len(self.poly_names))
        log10Tfg = jnp.sum(coeffs[:, None] * self.reduced_freq[None, :] ** powers[:, None], axis=0)
        return 10 ** log10Tfg

    def loglikelihood(self, particle):
        coeffs = jnp.array([particle[name] for name in self.poly_names])
        Tfg = self.foreground(coeffs)
        noise = 10 ** particle['lognoise']
        T21 = self._predict_T21(particle)

        var = noise ** 2 + (0.25 * T21) ** 2
        resid = self.T_SARAS - Tfg - T21
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * resid ** 2 / var)

    def derived(self, particle):
        return {'T21_q05': jnp.quantile(self._predict_T21(particle), 0.05)}

    @property
    def prior_bounds(self):
        bounds = super().prior_bounds
        for name, lims in self.opt['poly_coeff'].items():
            bounds[name] = tuple(lims)
        bounds['lognoise'] = tuple(self.opt['lognoise'])
        return bounds

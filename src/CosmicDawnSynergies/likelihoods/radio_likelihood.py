import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood, stats_arrays
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register()
class LikelihoodRadioBackground(BaseLikelihood):
    """LWA1/ARCADE2 radio-background upper-limit likelihood
    (data from Table 2 of Dowell & Taylor 2018).

    Legacy semantics: logL = sum log Phi((T_obs - T_model) / sqrt(dT_obs^2 +
    (0.05 T_model)^2)) with Phi the normal CDF (log-CDF used here for
    stability; identical wherever the legacy log(P) was finite).
    """

    def extract_data(self):
        assert self.files is not None, 'No data files provided'
        if not self.files.endswith('.npy'):
            raise NotImplementedError('Only .npy data files are supported in LikelihoodRadioBackground')
        nu_obs, T_obs, dT_obs = np.load(self.files, allow_pickle=True)

        log_nu = self.model_opt['dataset']['data_dims']['nu'].get('log', False)
        nu_col = self.to_normed(nu_obs, log_nu)[:, None]
        dim_stats = stats_arrays(self.param_stats, list(self.param_stats.keys())[:self.n_data_dims])
        self.block_norm = jnp.asarray(np.asarray(self.normalize(jnp.asarray(nu_col), dim_stats)))
        self.T_obs = jnp.asarray(np.asarray(T_obs, dtype=float))
        self.dT_obs = jnp.asarray(np.asarray(dT_obs, dtype=float))
        self.astro_stats = stats_arrays(self.param_stats, self.astro_names)

    def loglikelihood(self, particle):
        theta_n = self.normalize(self.astro_vector(particle), self.astro_stats)
        inputs = jnp.concatenate(
            [self.block_norm, jnp.broadcast_to(theta_n, (self.block_norm.shape[0], theta_n.size))], axis=1)
        T_model = self.unscale_target(self.model.net_g(inputs.astype(jnp.float32)))
        dT_model = T_model * 0.05
        t = (self.T_obs - T_model) / jnp.sqrt(self.dT_obs ** 2 + dT_model ** 2)
        return jnp.sum(norm.logcdf(t))

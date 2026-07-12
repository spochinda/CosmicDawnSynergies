from collections import OrderedDict

import jax.numpy as jnp
import numpy as np

from CosmicDawnSynergies.models.mlp_model import MLPModel


def norm_minmax(params, stats, invert=False):
    minimum, maximum = stats['min'], stats['max']
    if invert:
        return params * (maximum - minimum) + minimum
    return (params - minimum) / (maximum - minimum)


def norm_standard(params, stats, invert=False):
    mean, std = stats['mean'], stats['std']
    if invert:
        return params * std + mean
    return (params - mean) / std


def norm_minmax_extended(params, stats, invert=False):
    minimum, maximum = stats['min'], stats['max']
    if invert:
        return (params + 1) / 2 * (maximum - minimum) + minimum
    return (params - minimum) / (maximum - minimum) * 2 - 1


NORMALIZATIONS = {
    'norm_minmax': norm_minmax,
    'norm_standard': norm_standard,
    'norm_minmax_extended': norm_minmax_extended,
}


def stats_arrays(param_stats, keys=None):
    """param_stats (per-name dicts) -> column vectors of min/max/mean/std."""
    keys = list(param_stats.keys()) if keys is None else list(keys)
    return {
        'min': jnp.array([param_stats[k]['min'] for k in keys]),
        'max': jnp.array([param_stats[k]['max'] for k in keys]),
        'mean': jnp.array([param_stats[k]['mean'] for k in keys]),
        'std': jnp.array([param_stats[k]['std'] for k in keys]),
    }


class BaseLikelihood:
    """Binds one observational dataset to an emulator.

    ``extract_data`` runs host-side numpy at construction; ``loglikelihood``
    must be pure JAX (a function of one particle dict, jit/vmap-safe) so it
    can drive blackjax nested sampling. ``derived`` (optional) maps a particle
    dict to a dict of derived quantities, vmapped over the posterior after
    the run.
    """

    def __init__(self, opt):
        self.opt = opt
        self.files = opt.get('files')
        self.output_names = list(opt.get('output_names', []))

        # Load the emulator (standard emulator dir layout)
        self.emulator_dir = opt['emulator']
        self.model = MLPModel.from_emulator_dir(self.emulator_dir, which=opt.get('which', 'best'),
                                                use_ema=opt.get('use_ema', False))
        self.model_opt = self.model.opt

        self.target_log = self.model_opt['dataset']['targets_opt'].get('log', False)
        self.target_offset = self.model_opt['dataset']['targets_opt'].get('offset', 0.0)
        params_norm = self.model_opt['dataset']['params_opt'].get('normalization', 'norm_minmax')
        self.normalize = NORMALIZATIONS[params_norm]

        self.param_stats = self.model.param_stats
        self.n_data_dims = len(self.model_opt['dataset']['data_dims'])
        self.astro_names = list(self.param_stats.keys())[self.n_data_dims:]
        self.stats = stats_arrays(self.param_stats)

        self.extract_data()

    # ------------------------------------------------------------------ api
    def extract_data(self):
        """Host-side data loading/preprocessing (numpy)."""
        pass

    def loglikelihood(self, particle):
        """Pure-JAX logL of one particle dict {param_name: scalar}."""
        raise NotImplementedError

    def derived(self, particle):
        """Pure-JAX derived quantities of one particle dict, or None."""
        return None

    @property
    def prior_bounds(self):
        """Uniform prior bounds for this module's sampled parameters."""
        bounds = OrderedDict()
        for name in self.astro_names:
            bounds[name] = (self.param_stats[name]['min'], self.param_stats[name]['max'])
        return bounds

    # -------------------------------------------------------------- helpers
    def astro_vector(self, particle):
        """Particle dict -> raw astro-parameter vector in emulator order."""
        return jnp.array([particle[name] for name in self.astro_names])

    def unscale_target(self, pred):
        if self.target_log:
            pred = 10 ** pred
        if self.target_offset > 0:
            pred = pred - self.target_offset
        return pred

    @staticmethod
    def to_normed(values, stat):
        """Host-side log10 helper for data-dimension values."""
        return np.log10(values) if stat else np.asarray(values)

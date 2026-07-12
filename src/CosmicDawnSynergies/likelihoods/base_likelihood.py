from collections import OrderedDict

import jax.numpy as jnp

from CosmicDawnSynergies.models.mlp_model import MLPModel


class BaseLikelihood:
    """Binds one observational dataset to an emulator.

    ``extract_data`` runs host-side numpy at construction; ``loglikelihood``
    must be pure JAX (a function of one particle dict, jit/vmap-safe) so it
    can drive blackjax nested sampling. ``derived`` (optional) maps a particle
    dict to a dict of derived quantities, vmapped over the posterior after
    the run.

    Likelihoods work in raw physical units: ``self.predict`` (from
    MLPModel.make_predict_fn) owns the emulator's input log-transforms,
    normalization and target unscaling. Likelihood code never normalizes.
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
        self.predict = self.model.make_predict_fn()

        self.param_stats = self.model.param_stats
        self.n_data_dims = len(self.model_opt['dataset']['data_dims'])
        self.astro_names = list(self.param_stats.keys())[self.n_data_dims:]

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
        """Uniform prior bounds for this module's sampled parameters.

        Note: param_stats of log-transformed astro params (e.g. log10fX) are
        stats of the transformed column, so bounds are in sampled space.
        """
        bounds = OrderedDict()
        for name in self.astro_names:
            bounds[name] = (self.param_stats[name]['min'], self.param_stats[name]['max'])
        return bounds

    # -------------------------------------------------------------- helpers
    def astro_vector(self, particle):
        """Particle dict -> raw astro-parameter vector in emulator order."""
        return jnp.array([particle[name] for name in self.astro_names])

    def emulator_inputs(self, data_block, particle):
        """Stack a static data-dimension block (n, n_data_dims) with the
        particle's astro parameters into raw (n, in_dim) emulator inputs."""
        theta = self.astro_vector(particle)
        return jnp.concatenate(
            [data_block, jnp.broadcast_to(theta, (data_block.shape[0], theta.size))], axis=1)

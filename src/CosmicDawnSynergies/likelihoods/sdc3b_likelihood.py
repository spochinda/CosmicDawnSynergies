import jax.numpy as jnp
import numpy as np

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood, NORMALIZATIONS, stats_arrays
from CosmicDawnSynergies.models.mlp_model import MLPModel
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register()
class LikelihoodSDC3b(BaseLikelihood):
    """SDC3b 2D power-spectrum likelihood (Gaussian with 10% model-error floor),
    with optional xHI emulator for derived neutral fractions per band.

    Semantics match the legacy torch implementation: per band,
    logL = sum[ -0.5 log(2π(σ²+ (0.1 P)²)) - 0.5 (P_obs - P_err - P)²/(σ²+(0.1 P)²) ]
    with σ = 10^lognoise sampled as a nuisance parameter.
    """

    def __init__(self, opt):
        super().__init__(opt)

        self.emulator_xHI = self.opt.get('emulator_xHI', False)
        if self.emulator_xHI:
            self.init_emulator_xHI()

    # ------------------------------------------------------------- data prep
    def extract_data(self):
        data_dims = self.model_opt['dataset']['data_dims']
        log_z = data_dims['z'].get('log', False)
        log_kperp = data_dims['kperp'].get('log', False)
        log_kpar = data_dims['kpar'].get('log', False)

        kperp = np.loadtxt(self.opt['kperp_file'])
        kpar = np.loadtxt(self.opt['kpar_file'])
        kperp_n = self.to_normed(kperp, log_kperp)
        kpar_n = self.to_normed(kpar, log_kpar)
        # (n_kpar*n_kperp, 2) grid, matching legacy meshgrid(kperp, kpar) ordering
        kcoord = np.array(np.meshgrid(kperp_n, kpar_n)).reshape(2, -1).T

        self.bands = []
        for Pk_obs_file, Pk_err_file, lower_freq, upper_freq in self.files:
            Pk_obs = np.loadtxt(Pk_obs_file)
            Pk_err = np.loadtxt(Pk_err_file)
            mid_z = (1420.0 / ((lower_freq + upper_freq) / 2)) - 1
            z_n = np.log10(mid_z) if log_z else mid_z

            # static [z, kperp, kpar] block, pre-normalized column-wise with the
            # emulator's data-dim stats (normalization is elementwise)
            block = np.column_stack([np.full(len(kcoord), z_n), kcoord[:, 0], kcoord[:, 1]])
            dim_stats = stats_arrays(self.param_stats, list(self.param_stats.keys())[:3])
            block_n = np.asarray(self.normalize(jnp.asarray(block), dim_stats))

            self.bands.append({
                'z': mid_z,
                'block_norm': jnp.asarray(block_n),
                'Pk_obs_minus_err': jnp.asarray((Pk_obs - Pk_err).reshape(-1)),
            })

        self.astro_stats = stats_arrays(self.param_stats, self.astro_names)

    def init_emulator_xHI(self):
        self.model_xHI = MLPModel.from_emulator_dir(self.emulator_xHI, which=self.opt.get('which', 'best'))
        norm_name = self.model_xHI.opt['dataset']['params_opt'].get('normalization', 'norm_minmax')
        self.normalize_xHI = NORMALIZATIONS[norm_name]

        self.z_xHI = np.array([band['z'] for band in self.bands])
        self.output_names = self.output_names + [f'xHI_z{z:.2f}' for z in self.z_xHI]

        # Legacy quirk kept for parity: the xHI inputs are normalized with the
        # *Pk* emulator's param_stats (kperp/kpar entries removed), not the xHI
        # emulator's own stats.
        keys = [k for i, k in enumerate(self.param_stats.keys()) if i not in (1, 2)]
        self.stats_xHI = stats_arrays(self.param_stats, keys)

    # ------------------------------------------------------------ likelihood
    def loglikelihood(self, particle):
        theta = self.astro_vector(particle)
        theta_n = self.normalize(theta, self.astro_stats)
        noise = 10 ** particle['lognoise']

        logL = 0.0
        for band in self.bands:
            inputs = jnp.concatenate(
                [band['block_norm'], jnp.broadcast_to(theta_n, (band['block_norm'].shape[0], theta_n.size))],
                axis=1)
            pred = self.unscale_target(self.model.net_g(inputs.astype(jnp.float32)))
            var = noise ** 2 + (0.10 * pred) ** 2
            resid = band['Pk_obs_minus_err'] - pred
            logL += jnp.sum(-0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * resid ** 2 / var)
        return logL

    def derived(self, particle):
        if not self.emulator_xHI:
            return None
        theta = self.astro_vector(particle)
        inputs = jnp.column_stack([
            jnp.asarray(self.z_xHI),
            jnp.broadcast_to(theta, (len(self.z_xHI), theta.size)),
        ])
        inputs = self.normalize_xHI(inputs, self.stats_xHI)
        xHI = self.model_xHI.net_g(inputs.astype(jnp.float32))
        return {f'xHI_z{z:.2f}': xHI[i] for i, z in enumerate(self.z_xHI)}

    @property
    def prior_bounds(self):
        bounds = super().prior_bounds
        bounds['lognoise'] = tuple(self.opt['lognoise'])
        return bounds

from collections import OrderedDict

import jax.numpy as jnp
import numpy as np

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood
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
        kperp = np.loadtxt(self.opt['kperp_file'])
        kpar = np.loadtxt(self.opt['kpar_file'])
        # (n_kpar*n_kperp, 2) raw grid, matching legacy meshgrid(kperp, kpar) ordering
        kcoord = np.array(np.meshgrid(kperp, kpar)).reshape(2, -1).T

        self.bands = []
        for Pk_obs_file, Pk_err_file, lower_freq, upper_freq in self.files:
            Pk_obs = np.loadtxt(Pk_obs_file)
            Pk_err = np.loadtxt(Pk_err_file)
            mid_z = (1420.0 / ((lower_freq + upper_freq) / 2)) - 1

            # static raw [z, kperp, kpar] block; the model's predict_fn owns
            # log-transforms and normalization
            block = np.column_stack([np.full(len(kcoord), mid_z), kcoord[:, 0], kcoord[:, 1]])
            self.bands.append({
                'z': mid_z,
                'block': jnp.asarray(block),
                'Pk_obs_minus_err': jnp.asarray((Pk_obs - Pk_err).reshape(-1)),
            })

    def init_emulator_xHI(self):
        self.model_xHI = MLPModel.from_emulator_dir(self.emulator_xHI, which=self.opt.get('which', 'best'))

        self.z_xHI = np.array([band['z'] for band in self.bands])
        self.output_names = self.output_names + [f'xHI_z{z:.2f}' for z in self.z_xHI]

        # Legacy quirk kept for parity: the xHI inputs are normalized with the
        # *Pk* emulator's param_stats (kperp/kpar entries removed), not the xHI
        # emulator's own stats — passed as an explicit stats override.
        stats_override = OrderedDict(
            (k, v) for i, (k, v) in enumerate(self.param_stats.items()) if i not in (1, 2))
        self.predict_xHI = self.model_xHI.make_predict_fn(param_stats=stats_override)
        self.z_block_xHI = jnp.asarray(self.z_xHI)[:, None]

    # ------------------------------------------------------------ likelihood
    def loglikelihood(self, particle):
        noise = 10 ** particle['lognoise']

        logL = 0.0
        for band in self.bands:
            pred = self.predict(self.emulator_inputs(band['block'], particle))
            var = noise ** 2 + (0.10 * pred) ** 2
            resid = band['Pk_obs_minus_err'] - pred
            logL += jnp.sum(-0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * resid ** 2 / var)
        return logL

    def derived(self, particle):
        if not self.emulator_xHI:
            return None
        xHI = self.predict_xHI(self.emulator_inputs(self.z_block_xHI, particle))
        return {f'xHI_z{z:.2f}': xHI[i] for i, z in enumerate(self.z_xHI)}

    @property
    def prior_bounds(self):
        bounds = super().prior_bounds
        bounds['lognoise'] = tuple(self.opt['lognoise'])
        return bounds

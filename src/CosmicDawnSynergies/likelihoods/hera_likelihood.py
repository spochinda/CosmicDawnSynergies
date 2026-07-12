import os

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from CosmicDawnSynergies.likelihoods.base_likelihood import BaseLikelihood, stats_arrays
from CosmicDawnSynergies.utils.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register()
class LikelihoodHERA(BaseLikelihood):
    """HERA power-spectrum upper-limit likelihood over UVPSpec bands.

    Data extraction (hera_pspec, masking, decimation, window functions) is a
    verbatim numpy port of the legacy implementation; per band the model is
    Delta^2 -> window-function convolution -> one-sided Gaussian with a 20%
    model-error floor (log normal-CDF).
    """

    def __init__(self, opt):
        self.warnings = opt.get('warnings', False)
        self.set_negative_to_zero = opt.get('set_negative_to_zero', True)
        self.decimate_data = opt.get('decimate_data', True)
        self.mask_to_emulator_range = opt.get('mask_to_emulator_range', True)
        super().__init__(opt)

    def extract_data(self):
        import hera_pspec as hp  # heavy dependency, only needed by this module

        data_dims = self.model_opt['dataset']['data_dims']
        log_z = data_dims['z'].get('log', False)
        log_k = data_dims['k'].get('log', False)

        if self.mask_to_emulator_range:
            zmin, zmax = data_dims['z']['lims_nsample'][:-1]
            kmin, kmax = data_dims['k']['lims_nsample'][:-1]

        dim_stats = stats_arrays(self.param_stats, list(self.param_stats.keys())[:self.n_data_dims])

        self.bands = []
        for file in self.files:
            fn = os.path.basename(file)
            uvp = hp.UVPSpec()
            uvp.read_hdf5(file)
            for band, blpair, polpair in uvp.get_all_keys():
                spw_index = uvp.spw_array[band]
                freq_start, freq_end, Nfreqs, Ndlys = uvp.get_spw_ranges()[band]
                z = uvp.cosmo.f2z(np.mean([freq_start, freq_end]))
                if self.mask_to_emulator_range and (z < zmin or z > zmax):
                    print(f'Skipping z={z:.2f} outside of zmin={zmin} and zmax={zmax} for file {fn}')
                    continue
                k_para = uvp.get_kparas(spw_index)
                k_perp = uvp.get_kperps(spw_index)
                k_mag = np.sqrt(k_perp ** 2 + k_para ** 2)
                dsq = uvp.get_data((band, blpair, polpair))[0].real.copy()
                assert uvp.norm_units == 'h^-3 Mpc^3 k^3 / (2pi^2)', (
                    f'Units are {uvp.norm_units}. Maybe need to use uvp.convert_to_deltasq()?')
                try:
                    wfn = uvp.get_window_function((band, blpair, polpair))[0]
                except AttributeError:
                    print(f'AttributeError: Setting window function to identity matrix for z={z:.2f} in {fn}',
                          flush=True)
                    wfn = np.identity(dsq.shape[0])
                try:
                    var = uvp.get_cov((band, blpair, polpair))[0].diagonal().real.copy()
                except AttributeError:
                    print(f'AttributeError: Getting variance from stats for z={z:.2f} in {fn}', flush=True)
                    var = uvp.get_stats('P_SN', (band, blpair, polpair)).real[0]
                std = np.sqrt(var)

                if self.set_negative_to_zero:
                    dsq[dsq < 0] = 0

                if self.mask_to_emulator_range:
                    mask = np.logical_and(k_mag >= kmin, k_mag <= kmax)
                    mask = np.logical_and(mask, std > 0)
                    k_mag = k_mag[mask]
                    dsq = dsq[mask]
                    std = std[mask]
                    wfn = wfn[mask][:, mask]

                if self.decimate_data:
                    print(f'Decimating data for z={z:.2f} in file {fn}')
                    k_mag, dsq, std, wfn = self.decimate(k_mag, dsq, std, wfn)
                else:
                    print(f'Not decimating data for z={z:.2f} in file {fn}')

                # static [z, k] block, pre-normalized with the emulator's stats
                z_n = np.log10(z) if log_z else z
                k_n = self.to_normed(k_mag, log_k)
                block = np.column_stack([np.full(len(k_n), z_n), k_n])
                block_n = np.asarray(self.normalize(jnp.asarray(block), dim_stats))

                self.bands.append({
                    'z': z, 'file': fn,
                    'block_norm': jnp.asarray(block_n),
                    'dsq': jnp.asarray(dsq),
                    'std': jnp.asarray(std),
                    'wfn': jnp.asarray(wfn),
                })

        self.astro_stats = stats_arrays(self.param_stats, self.astro_names)

    def decimate(self, k_mag, dsq, std, wfn):
        idx = np.argmin(dsq + 2 * std)
        is_odd = idx % 2
        mask = np.arange(len(k_mag)) % 2 == is_odd
        return k_mag[mask], dsq[mask], std[mask], wfn[mask][:, mask]

    def loglikelihood(self, particle):
        theta_n = self.normalize(self.astro_vector(particle), self.astro_stats)

        logL = 0.0
        for band in self.bands:
            inputs = jnp.concatenate(
                [band['block_norm'], jnp.broadcast_to(theta_n, (band['block_norm'].shape[0], theta_n.size))],
                axis=1)
            pred = self.unscale_target(self.model.net_g(inputs.astype(jnp.float32)))
            pred = band['wfn'] @ pred
            t = (band['dsq'] - pred) / jnp.sqrt(band['std'] ** 2 + (0.2 * pred) ** 2)
            logL += jnp.sum(norm.logcdf(t))
        return logL

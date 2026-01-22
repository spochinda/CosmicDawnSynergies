import os
import sys
import glob
import numpy as np
import torch
import copy

from scipy.constants import parsec, physical_constants
import hera_pspec as hp

import scipy.special as ssp
import CosmicDawnSynergies.itamar.radio_cutoff_calc as rad

import astropy.constants as c
import astropy.units as u
from scipy.interpolate import interp1d

from CosmicDawnSynergies.model import MLPModel
from CosmicDawnSynergies.utils import yaml_load


class LikelihoodBase:
    def __init__(self, 
                 opt,
                 model_opt, 
                 ):
        self.opt = opt
        self.model_opt = model_opt

        self.files = self.opt.get("files")
        self.output_names = self.opt.get("output_names", [self.__class__.__name__,])

        self.target_log = self.model_opt['dataset']['targets_opt'].get('log', False)
        self.target_offset = self.model_opt['dataset']['targets_opt'].get('offset', 0.0)
        self.params_norm = self.model_opt['dataset']['params_opt'].get('normalization', False)

        # Load trained weights and param_stats
        self.model = MLPModel(self.model_opt)
        self.emulator_path = self.opt.get('emulator')
        self.model.load_network(self.model.net_g, self.emulator_path, strict=True, param_key='params')

        # self.extract_data()
        # Load xHI (SDC3b)
        # self.nDerived = len(self.output_names)

    def extract_data(self):
        pass        

    def computeLikelihood(self, params):
        pass

    def predict(self, params):
        pass

    def norm_minmax(self, params, invert=False, **kwargs):
        minimum = kwargs.get("min", None)
        maximum = kwargs.get("max", None)
        if invert:
            return params * (maximum - minimum) + minimum
        else:
            return (params - minimum) / (maximum - minimum)

    def norm_standard(self, param, invert=False, **kwargs):
        mean = kwargs.get("mean", None)
        std = kwargs.get("std", None)
        if invert:
            return param * std + mean
        else:
            return (param - mean) / std
    
    def norm_minmax_extended(self, params, invert=False, **kwargs):
        minimum = kwargs.get("min", None)
        maximum = kwargs.get("max", None)
        if invert:
            return (params + 1) / 2 * (maximum - minimum) + minimum
        else:
            return (params - minimum) / (maximum - minimum) * 2 - 1
        
    def get_prior_indices(self, prior_dict):
        prior_keys = list(prior_dict.keys())
        emulator_keys = list(self.model.param_stats.keys())
        n_data_dims = len(self.model_opt['dataset']['data_dims'])
        self.prior_indices = []
        for key in emulator_keys[n_data_dims:]: #find non-data_dim indices of emulator_keys in prior_keys
            self.prior_indices.append(prior_keys.index(key))


class LikelihoodRadioBackground(LikelihoodBase):
    def __init__(self, 
                 opt,
                 model_opt, 
                 **kwargs
                 ):
        super().__init__(opt, model_opt)
        """
        Likelihood module for the LWA1/ARCADE2 constraints 
        (data from Table 2 of Dowell & Taylor (2018))
        """
        self.extract_data()

        self.nDerived = len(self.output_names)
        
    def extract_data(self, **kwargs):
        # Precompute normalization dict
        self.norm_dict = {}
        self.norm_dict['min'] = np.array([item[1]['min'] for item in self.model.param_stats.items()])
        self.norm_dict['max'] = np.array([item[1]['max'] for item in self.model.param_stats.items()])
        self.norm_dict['mean'] = np.array([item[1]['mean'] for item in self.model.param_stats.items()])
        self.norm_dict['std'] = np.array([item[1]['std'] for item in self.model.param_stats.items()])

        assert self.files is not None, "No data files provided"
        if self.files.endswith('.npy'):
            self.nu_obs, self.T_obs, self.dT_obs = np.load(self.files, allow_pickle=True)
        else:
            raise NotImplementedError("Only .npy data files are supported in LikelihoodRadioBackground")
        
        self.nu_obs_norm = np.log10(self.nu_obs) if self.model_opt['dataset']["data_dims"]["nu"]["log"] else self.nu_obs
        #nu_key = "log10nu" if self.model_opt['dataset']["data_dims"]["nu"]["log"] else "nu"
        #nu_min = self.model.param_stats[nu_key]['min']
        #nu_max = self.model.param_stats[nu_key]['max']
        #self.nu_obs_norm = (self.nu_obs_norm - nu_min) / (nu_max - nu_min)

    def computeLikelihood(self, params):
        params = np.array(params)
        params = params[self.prior_indices]

        T_model = self.predict(params=params)
        dT_model = T_model*0.05

        P = 0.5 * (1 + ssp.erf( (self.T_obs - T_model) / np.sqrt(2) / np.sqrt(self.dT_obs**2+dT_model**2))) 
        logL = np.log(P).sum()
        logL = float(logL)
        return logL, [logL,]

    def predict(self, params):
        params = np.array([np.nan, *params])
        params=np.tile(params, (len(self.nu_obs_norm), 1))
        params[:,0] = self.nu_obs_norm
        params = getattr(self, self.params_norm)(params, invert=False, **self.norm_dict)

        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model.net_g(params)
            pred = pred.detach().cpu().numpy()

            if self.target_log:
                pred = 10**pred
            if self.target_offset>0:
                pred = pred - self.target_offset
        return pred

    def get_T_radio_today_Jiten(self, zs, sfr, frad=1):
        '''
        Returns the T_radio today by integrating the redshifting SED across all zs,
        scaled by the SFR at the given z and frad.
        Note: modified from Itamar's code above to be much more efficient (x2000 faster)
        '''

        nu_today = np.logspace(-2, 1.1, 100)*1e9*u.Hz
        log_nu, log_sed = rad.get_radio_sed('power_law')
        log_sed_interp = interp1d(log_nu, log_sed, kind='linear')

        constants = 1/(8*np.pi*c.k_B) * (c.c**3/nu_today**2)
        dz = abs(zs[1] - zs[0])
        A = 1/(rad.Hubble_const(zs)*u.km/u.s/u.Mpc) * 1/(1+zs) * dz # dz/(H*(1+z))
        redshifted_sed = 10**log_sed_interp(np.log10(np.outer(1+zs,nu_today.value))) * (u.W/u.Hz) 

        T_at_z = constants[None, :] * A[:, None] * frad * redshifted_sed * sfr[:, None]/(u.Mpc**3)
        T_today = np.sum(T_at_z,axis=0).to_value(u.K)
            
        return nu_today, T_today


class LikelihoodXRB(LikelihoodBase):
    def __init__(self, 
                 opt,
                 model_opt, 
                 **kwargs
                 ):
        super().__init__(opt, model_opt)
        """
        Likelihood module for the Chandra constraints 
        (data from Hickox+2006 and Harrison+2016)
        """
        self.extract_data() 

        self.nDerived = len(self.output_names)

    def extract_data(self):
        # Precompute normalization dict
        self.norm_dict = {}
        self.norm_dict['min'] = np.array([item[1]['min'] for item in self.model.param_stats.items()])
        self.norm_dict['max'] = np.array([item[1]['max'] for item in self.model.param_stats.items()])
        self.norm_dict['mean'] = np.array([item[1]['mean'] for item in self.model.param_stats.items()])
        self.norm_dict['std'] = np.array([item[1]['std'] for item in self.model.param_stats.items()])

        eV_toHz = physical_constants['electron volt-hertz relationship'][0]
        self.keV_toHz = eV_toHz*1e3
        self.sr_todeg2 = (180/np.pi)**2
        Mpc_tom = 1e6 * parsec
        Mpc_tocm = Mpc_tom * 1e2
        self.cm_toMpc = 1/Mpc_tocm
        
        self.units = self.keV_toHz * self.cm_toMpc**2 / self.sr_todeg2

        ###Data###
        self.X_limits = np.array([ #nu_min, nu_max, mean, std 
            #[0.5, 2, 8.15*1e-12, 0.58*1e-12], #Lehmer+2012 
            [1, 2, 1.04*1e-12, 0.14*1e-12], #Hickox & Markevitch (2006)
            [2, 8, 3.4*1e-12, 1.7*1e-12], #Hickox & Markevitch (2006)
            [8, 24, 6.013*1e-8/self.sr_todeg2, 0.14*1e-8/self.sr_todeg2], #Harrison et al. (2016) #6.773*1e-8/sr_todeg2, 0.348*1e-8/sr_todeg2
            [20, 50, 6.56*1e-8/self.sr_todeg2, 0.273*1e-8/self.sr_todeg2], #Harrison et al. (2016) #6.205*1e-8/sr_todeg2, 0.17*1e-8/sr_todeg2
            ])
        
        minE, maxE = min(self.X_limits[:,0]), max(self.X_limits[:,1])
        self.E_kev = np.geomspace(minE, maxE, 100)
        self.E_kev_norm = np.log10(self.E_kev) if self.model_opt['dataset']["data_dims"]["E_kev"]["log"] else self.E_kev
        #E_kev_key = "log10E_kev" if self.model_opt['dataset']["data_dims"]["E_kev"]["log"] else "E_kev"
        #E_kev_min = self.model.param_stats[E_kev_key]['min']
        #E_kev_max = self.model.param_stats[E_kev_key]['max']
        #self.E_kev_norm = (self.E_kev_norm - E_kev_min) / (E_kev_max - E_kev_min)

    def computeLikelihood(self, params):
        params = np.array(params)
        params = params[self.prior_indices]
        pred = self.predict(params=params)

        # integrate in log-log space
        logE_kev = np.log10(self.E_kev)
        logpred = np.log10(pred)
        logpred_interpolator = interp1d(logE_kev, logpred, kind='linear', fill_value='extrapolate')

        logL = 0
        for xmin,xmax,obs,std in self.X_limits: #integrate model from xmin to xmax and compare to data to get logL
            E_kev = np.geomspace(xmin, xmax, 100)
            logE_kev = np.log10(E_kev)
            logpred = logpred_interpolator(logE_kev)

            # Check interpolated values
            if np.any(logpred > 300):
                return -1e30, [-1e30,]

            E_kev = 10**logE_kev
            pred = 10**logpred

            # Check for overflow/invalid after exponentiation
            if np.any(~np.isfinite(pred)):
                return -1e30, [-1e30,]

            try:
                pred_integral = np.trapezoid(pred, E_kev*self.keV_toHz) * self.cm_toMpc**2 / self.sr_todeg2
            except Exception as e:
                pred_integral = np.trapz(pred, E_kev*self.keV_toHz) * self.cm_toMpc**2 / self.sr_todeg2

            # Check if integral is finite
            if not np.isfinite(pred_integral):
                return -1e30, [-1e30,]

            P = 0.5 * (1 + ssp.erf( (obs - pred_integral) / np.sqrt(2) / np.sqrt(std**2+(pred_integral*0.05)**2) ))

            # Ensure P is valid
            if not np.isfinite(P) or P <= 0:
                return -1e30, [-1e30,]

            logL += np.log(P)

        logL = float(logL)
        return logL, [logL,]
        
    def predict(self, params):
        params = np.array([np.nan, *params])
        params=np.tile(params, (len(self.E_kev_norm), 1))
        params[:,0] = self.E_kev_norm
        params = getattr(self, self.params_norm)(params, invert=False, **self.norm_dict)

        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model.net_g(params)
            pred = pred.detach().cpu().numpy()

            if self.target_log:
                pred = 10**pred
            if self.target_offset>0:
                pred = pred - self.target_offset
        return pred
        

class LikelihoodHERA(LikelihoodBase):
    def __init__(self, 
                 opt,
                 model_opt, 
                 **kwargs
                 ):
        super().__init__(opt, model_opt)
        self.warnings = self.opt.get("warnings", False)
        self.set_negative_to_zero = self.opt.get("set_negative_to_zero", True)
        self.decimate_data = self.opt.get("decimate_data", True)
        self.mask_to_emulator_range = self.opt.get("mask_to_emulator_range", True)

        self.extract_data()

        self.nDerived = len(self.output_names)

    def computeLikelihood(self, params):
        try:
            # Important: model must take k as h/cMpc!
            params = np.array(params)
            params = params[self.prior_indices]

            logL = []
            for i,band in enumerate(self.data.keys()):
                dsq = self.data[band]["dsq"]
                std = self.data[band]["std"]
                wfn = self.data[band]["wfn"]
                z = self.data[band]["z"]
                k_mag = self.data[band]["k_mag"]
                dsq_pred = self.predict(z, k_mag, params)
                dsq_pred = wfn @ dsq_pred #theory=model for diag(wfn), @=matrix multiplication
                assert np.shape(dsq_pred) == np.shape(dsq), "Shape mismatch"
                r = dsq - dsq_pred
                logL_ = np.sum(np.log(0.5 * (1 + ssp.erf(r / np.sqrt(2) / np.sqrt(std**2+(0.2*dsq_pred)**2)))))
                logL.append(logL_)
            logL = np.sum(logL)
            logL = float(logL)

            
            nDerived_params = [logL,] #self.get_log10Trad_log10Ts(z=self.redshifts, p=params)
            return logL, [*nDerived_params]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)
    
    def predict(self, z, karr, params):
        try:
            dim_log_0 = self.model_opt['dataset']["data_dims"]["z"]["log"]
            dim_log_1 = self.model_opt['dataset']["data_dims"]["k"]["log"]

            z = np.log10(z) if dim_log_0 else z
            karr = np.log10(karr) if dim_log_1 else karr

            params = np.array([z, np.nan, *params])
            params=np.tile(params, (len(karr), 1))
            params[:,1] = karr
            params = getattr(self, self.params_norm)(params, invert=False, **self.norm_dict)

            with torch.no_grad():
                params = torch.from_numpy(params).to(dtype=torch.float32)
                pred = self.model.net_g(params)
                pred = pred.detach().cpu().numpy()

                if self.target_log:
                    pred = 10**pred
                if self.target_offset>0:
                    pred = pred - self.target_offset

            return pred
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        
    def extract_data(self, **kwargs):
        # Precompute normalization dict
        self.norm_dict = {}
        self.norm_dict['min'] = np.array([item[1]['min'] for item in self.model.param_stats.items()])
        self.norm_dict['max'] = np.array([item[1]['max'] for item in self.model.param_stats.items()])
        self.norm_dict['mean'] = np.array([item[1]['mean'] for item in self.model.param_stats.items()])
        self.norm_dict['std'] = np.array([item[1]['std'] for item in self.model.param_stats.items()])

        if self.mask_to_emulator_range:
            zlim = self.model_opt['dataset']['data_dims']['z']['lims_nsample'][:-1]
            zmin, zmax = zlim
            klim = self.model_opt['dataset']['data_dims']['k']['lims_nsample'][:-1]
            kmin, kmax = klim

        self.data = dict()
        i=0
        for file in self.files:
            fn = os.path.basename(file)
            uvp = hp.UVPSpec()
            uvp.read_hdf5(file)
            band_keys = uvp.get_all_keys()
            for band,blpair,polpair in band_keys:
                spw_index = uvp.spw_array[band]
                freq_start, freq_end, Nfreqs, Ndlys = uvp.get_spw_ranges()[band]
                z = uvp.cosmo.f2z(np.mean([freq_start, freq_end]))
                if self.mask_to_emulator_range:
                    if z < zmin or z > zmax:
                        print(f"Skipping z={z:.2f} outside of zmin={zmin} and zmax={zmax} for file {fn}")
                        continue
                k_para = uvp.get_kparas(spw_index)
                k_perp = uvp.get_kperps(spw_index)
                k_mag = np.sqrt(k_perp**2 + k_para**2)
                dsq = uvp.get_data((band,blpair,polpair))[0].real.copy()
                assert uvp.norm_units == "h^-3 Mpc^3 k^3 / (2pi^2)", f"Units are {uvp.norm_units}. Maybe need to use uvp.convert_to_deltasq()?"
                #uvp.convert_to_deltasq() # if not already in delta^2. However all data products so far have been delta^2
                #dsq = uvp.get_data((band,blpair,polpair))[0].real.copy()
                try:
                    wfn = uvp.get_window_function((band,blpair,polpair))[0]
                except AttributeError:
                    print(f"AttributeError: Setting window funciton to identity matrix for z={z:.2f} in file {fn}", flush=True)
                    wfn = np.identity(dsq.shape[0])
                try:
                    var = uvp.get_cov((band,blpair,polpair))[0].diagonal().real.copy()
                except AttributeError:
                    print(f"AttributeError: Getting variance from stats for z={z:.2f} in file {fn}", flush=True)
                    var = uvp.get_stats('P_SN', (band,blpair,polpair)).real[0]
                std = np.sqrt(var)

                #mask: keep values within emulator range and with non-zero std, and set negative dsq values to zero                
                if self.set_negative_to_zero:
                    dsq[dsq < 0] = 0
                
                if self.mask_to_emulator_range:
                    mask = np.logical_and(k_mag >= kmin, k_mag <= kmax)
                    mask = np.logical_and(mask, std > 0)
                    k_mag = k_mag[mask]
                    dsq = dsq[mask]
                    std = std[mask]
                    wfn = wfn[mask][:,mask]
                
                #decimate around 2 sigma minimum
                print(file)
                if self.decimate_data: #"H1C_IDR3" in file:
                    print(f"Decimating data for z={z:.2f} in file {os.path.basename(file)}")
                    k_mag, dsq, std, wfn = self.decimate(k_mag, dsq, std, wfn)
                else:
                    print(f"Not decimating data for z={z:.2f} in file {os.path.basename(file)}")

                self.data[i] = {"k_mag": k_mag, "dsq": dsq, "std": std, "wfn": wfn, "z": z, "file": fn}
                i+=1

                if self.warnings:
                    if len(k_perp) != len(k_para):
                        print(f"k_perp shape {len(k_perp)} != k_para shape {len(k_para)}")
                    if len(k_perp) == 1:
                        print(f"Only one k_perp value")
                        if np.isclose(k_perp, 0):
                            print(f"k_perp is close to 0, k_mag approx k_para")

    
    def decimate(self, k_mag, dsq, std, wfn):
        idx = np.argmin(dsq+2*std)
        is_odd = idx % 2
        mask = np.arange(len(k_mag)) % 2 == is_odd
        k_mag = k_mag[mask]
        dsq = dsq[mask]
        std = std[mask]
        wfn = wfn[mask][:,mask]
        return k_mag, dsq, std, wfn
    
    def plot_data(self, axes, **kwargs):
        xlabel = kwargs.get("xlabel", r"$k$ [$h$/Mpc]")
        ylabel = kwargs.get("ylabel", r"$\Delta^2(k)$")
        xmin = kwargs.get("xmin", None)
        xmax = kwargs.get("xmax", None)
        ymin = kwargs.get("ymin", None)
        ymax = kwargs.get("ymax", None)
        yscale = kwargs.get("yscale", None)
        xscale = kwargs.get("xscale", None)

        rows, cols = np.shape(axes)
        axes_flattened = axes.flatten()
        for i,band in enumerate(self.data.keys()):
            data = self.data[band]
            dsq = data["dsq"]
            std = data["std"]
            k_mag = data["k_mag"]
            z = data["z"]
            axes_flattened[i].errorbar(k_mag, dsq, yerr=std, fmt='o', label=f"z={z:.2f}")
            axes_flattened[i].set_xscale(xscale)
            axes_flattened[i].set_yscale(yscale)
            axes_flattened[i].set_xlabel(xlabel)
            axes_flattened[i].set_ylabel(ylabel)
            axes_flattened[i].legend()
        axes = axes_flattened.reshape((rows,cols))
        return axes
    

class LikelihoodSARAS3(LikelihoodBase):
    def __init__(self, 
                 opt,
                 model_opt, 
                 **kwargs
                 ):
        super().__init__(opt, model_opt)
        """
        Likelihood module for the SARAS3 constraints.
        """
        self.extract_data() 
        
        self.nDerived = len(self.output_names)

    def extract_data(self):
        # Precompute normalization dict
        self.norm_dict = {}
        self.norm_dict['min'] = np.array([item[1]['min'] for item in self.model.param_stats.items()])
        self.norm_dict['max'] = np.array([item[1]['max'] for item in self.model.param_stats.items()])
        self.norm_dict['mean'] = np.array([item[1]['mean'] for item in self.model.param_stats.items()])
        self.norm_dict['std'] = np.array([item[1]['std'] for item in self.model.param_stats.items()])

        self.freq, self.T_SARAS, self.weights, self.fg_fit, self.fg_fit_T_resid = np.loadtxt(self.files).T
        log_freq = np.log10(self.freq)
        self.reduced_freq = 2 * ((log_freq - log_freq.min()) / (log_freq.max()-log_freq.min())) - 1
        self.redshifts = 1420/self.freq - 1

        self.redshifts_norm = np.log10(self.redshifts) if self.model_opt['dataset']["data_dims"]["z"]["log"] else self.redshifts
        #redshifts_key = "log10z" if self.model_opt['dataset']["data_dims"]["z"]["log"] else "z"
        #z_min = self.model.param_stats[redshifts_key]['min']
        #z_max = self.model.param_stats[redshifts_key]['max']
        #self.redshifts_norm = (self.redshifts_norm - z_min) / (z_max - z_min)

    def foreground(self, params):
        log10Tfg = np.sum(np.array([a_i * self.reduced_freq**i for i,a_i in enumerate(params)]), axis=0)
        Tfg = 10**log10Tfg
        return Tfg

    def computeLikelihood(self, params):
        params = np.array(params)

        Tfg = self.foreground(params[self.poly_coeff_indices])
        lognoise = params[self.noise_indices]
        noise = 10**lognoise

        T21_pred = self.predict(params[self.prior_indices]) * 1e-3  #mK to K

        logL = (
            -0.5*np.log(2*np.pi*(noise**2+(0.25*T21_pred)**2)) 
            - 0.5 * (self.T_SARAS - Tfg - T21_pred)**2
            /(noise**2+(0.25*T21_pred)**2)
            ).sum()
        logL = float(logL)

        return logL, [float(np.quantile(T21_pred,0.05)),]
    
    def predict(self, params):
        params = np.array([np.nan, *params])
        params=np.tile(params, (len(self.redshifts_norm), 1))
        params[:,0] = self.redshifts_norm
        params = getattr(self, self.params_norm)(params, invert=False, **self.norm_dict)

        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model.net_g(params)
            pred = pred.detach().cpu().numpy()

            if self.target_log:
                pred = 10**pred
            if self.target_offset>0:
                pred = pred - self.target_offset
        return pred
    
    def get_prior_indices(self, prior_dict):
        prior_keys = list(prior_dict.keys())
        emulator_keys = list(self.model.param_stats.keys())
        n_data_dims = len(self.model_opt['dataset']['data_dims'])
        self.prior_indices = []
        for key in emulator_keys[n_data_dims:]: #find non-data_dim indices of emulator_keys in prior_keys
            self.prior_indices.append(prior_keys.index(key))

        self.poly_coeff_indices = []
        self.noise_indices = []
        for key in prior_keys:
            if "fg_a" in key:
                self.poly_coeff_indices.append(prior_keys.index(key))
            if "noise" in key:
                self.noise_indices.append(prior_keys.index(key))


class LikelihoodPowerSpectrum:
    def __init__(self, 
                 prior_dict,
                 emulator,
                 files, 
                 data_dims = ["z", "log10k",],
                 output_names = [r"\log L_\mathrm{dsq}",],
                 **kwargs
                 ):
        assert False, "Deprecated likelihood class."
        self.set_negative_to_zero = kwargs.get("set_negative_to_zero", True)
        self.decimate_data = kwargs.get("decimate_data", False)
        self.mask_to_emulator_range = kwargs.get("mask_to_emulator_range", True)
        self.k_name_in_scale_opt = kwargs.get("k_name_in_scale_opt", "log10k")
        
        self.emulator = emulator
        self.prior_dict = prior_dict
        self.files = files
        self.output_names = output_names
        self.nDerived = len(self.output_names)
        self.scaler = Scaler(self.emulator.scale_opt)        

        prior_keys = list(self.prior_dict.keys())
        emulator_keys = list(self.emulator.scale_opt.keys())
        self.prior_indices = []
        for key in emulator_keys: #find non-data_dim indices of emulator_keys in prior_keys
            if key not in data_dims:
                self.prior_indices.append(prior_keys.index(key))

        self.extract_data()


    def computeLikelihood(self, params):
        # Important: model must take k as h/cMpc!
        params = np.array(params)
        params = params[self.prior_indices]

        logL = []
        for i,band in enumerate(self.data.keys()):
            dsq = self.data[band]["dsq"]
            std = self.data[band]["std"]
            z = self.data[band]["z"]
            k_mag = self.data[band]["k_mag"]
            dsq_pred = self.predict(z, k_mag, params)
            assert np.shape(dsq_pred) == np.shape(dsq), "Shape mismatch"
            r = dsq - dsq_pred
            logL_ = np.sum(np.log(0.5 * (1 + ssp.erf(r / np.sqrt(2) / np.sqrt(std**2+(0.2*dsq_pred)**2)))))
            logL.append(logL_)
        logL = np.sum(logL)
        logL = float(logL)
        return logL, [logL,]
    
    def predict(self, z, karr, params):
        is_log_z = self.emulator.data_opt["data_dims"][0]["z"]["log"]
        is_log_k = self.emulator.data_opt["data_dims"][1]["k"]["log"]
        data_log = self.emulator.data_opt["data_log"]

        z = np.log10(z) if is_log_z else z
        karr = np.log10(karr) if is_log_k else karr

        params = np.array([z, np.nan, *params])
        params=np.tile(params, (len(karr), 1))
        params[:,1] = karr
        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if data_log:
                pred = 10**pred
        return pred
        
    def extract_data(self, **kwargs):        
        if self.mask_to_emulator_range:
            k_index = list(self.emulator.scale_opt.keys()).index(self.k_name_in_scale_opt)
            kmin, kmax = self.emulator.data_opt["data_dims"][k_index]["k"]["lims"]

        self.data = dict()
        for i,file in enumerate(self.files):
            loaded_data = np.load(file, allow_pickle=True).item()
            z = np.array(loaded_data["z"])
            k_mag = np.array(loaded_data["k_mag"])
            dsq = np.array(loaded_data["dsq"])
            std = np.array(loaded_data["std"])
            
            if self.mask_to_emulator_range:
                k_mask = np.logical_and(k_mag >= kmin, k_mag <= kmax)
                k_mag = k_mag[k_mask]
                dsq = dsq[k_mask]
                std = std[k_mask]
            
            if self.set_negative_to_zero:
                dsq[dsq < 0] = 0
            
            if self.decimate_data:
                k_mag, dsq, std = self.decimate(k_mag, dsq, std)
            
            self.data[i] = {"k_mag": k_mag, "dsq": dsq, "std": std, "z": z}
        return self.data

    def decimate(self, k_mag, dsq, std):
        idx = np.argmin(dsq+2*std)
        is_odd = idx % 2
        mask = np.arange(len(k_mag)) % 2 == is_odd
        k_mag = k_mag[mask]
        dsq = dsq[mask]
        std = std[mask]
        return k_mag, dsq, std


class LikelihoodSDC3b(LikelihoodBase):
    def __init__(self, 
                 opt,
                 model_opt, 
                 **kwargs
                 ):
        super().__init__(opt, model_opt)
        
        self.extract_data() 

        # Load xHI emulator if provided
        self.emulator_xHI = self.opt.get("emulator_xHI", False)
        if self.emulator_xHI:
            self.init_emulator_xHI()

        self.nDerived = len(self.output_names)
    
    def init_emulator_xHI(self):
        self.model_opt_xHI = os.path.abspath(os.path.join(self.emulator_xHI, os.pardir, os.pardir))
        self.model_opt_xHI = glob.glob(os.path.join(self.model_opt_xHI, "*.yml"))[0]
        self.model_opt_xHI = yaml_load(self.model_opt_xHI)
        self.model_opt_xHI['is_train'] = False
        self.model_opt_xHI['dist'] = False
        self.model_opt_xHI['num_gpu'] = 0
        self.model_opt_xHI['network_opt']['in_dim'] = len(self.model_opt_xHI['dataset']['params_opt']['names']) + len(self.model_opt_xHI['dataset']['data_dims'].items())
        self.model_xHI = MLPModel(self.model_opt_xHI)
        self.model_xHI.load_network(self.model_xHI.net_g, self.emulator_xHI, strict=True, param_key='params')
        
        self.params_norm_xHI = self.model_opt_xHI['dataset']['params_opt']['normalization']
        self.z_xHI = np.array([])
        for band in self.data['bands'].keys():
            self.z_xHI = np.append(self.z_xHI, self.data['bands'][band]["z"])
        output_names_xHI = [f"xHI_z{z:.2f}" for z in self.z_xHI]
        self.output_names.extend(output_names_xHI)

        #remove kperp and kpar normalization info from norm_dict_xHI for xHI prediction
        self.norm_dict_xHI = copy.deepcopy(self.norm_dict)
        self.norm_dict_xHI['min'] = np.delete(self.norm_dict_xHI['min'],[1,2])
        self.norm_dict_xHI['max'] = np.delete(self.norm_dict_xHI['max'],[1,2])
        self.norm_dict_xHI['mean'] = np.delete(self.norm_dict_xHI['mean'],[1,2])
        self.norm_dict_xHI['std'] = np.delete(self.norm_dict_xHI['std'],[1,2])


    def predict_xHI(self, params):
        params = np.array([np.nan, *params])
        params=np.tile(params, (len(self.z_xHI), 1))
        params[:,0] = self.z_xHI
        params = getattr(self, self.params_norm_xHI)(params, invert=False, **self.norm_dict_xHI)
        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model_xHI.net_g(params)
            pred = pred.detach().cpu().numpy().tolist()
        return pred

    def extract_data(self, **kwargs):
        self.data = {'bands': {}}
        for i,(Pk_obs_file,Pk_err_file,lower_freq,upper_freq) in enumerate(self.files):
            self.data['bands'][f"band_{i+1}"] = {}
            
            Pk_obs = np.loadtxt(Pk_obs_file)
            Pk_err = np.loadtxt(Pk_err_file)
            
            mid_freq = (lower_freq + upper_freq) / 2
            mid_z = (1420/mid_freq) - 1
            log_z = self.model_opt['dataset']["data_dims"]["z"].get("log", False)
            self.data['bands'][f"band_{i+1}"][f"z"] = mid_z
            z_norm = np.log10(mid_z) if log_z else mid_z
            #z_key = "log10z" if log_z else "z"
            #z_norm = getattr(self, self.params_norm)(z_norm, invert=False, **self.model.param_stats[z_key])
            
            self.data['bands'][f"band_{i+1}"][f"z_norm"] = z_norm
            self.data['bands'][f"band_{i+1}"][f"Pk_obs"] = Pk_obs
            self.data['bands'][f"band_{i+1}"][f"Pk_err"] = Pk_err
            self.data['bands'][f"band_{i+1}"][f"Pk_obs_minus_err"] = (Pk_obs - Pk_err).reshape(-1)
        self.data["kperp"] = np.loadtxt(self.opt.get("kperp_file"))
        self.data["kpar"] = np.loadtxt(self.opt.get("kpar_file"))
        self.data["kcoord"] = np.array(np.meshgrid(self.data["kperp"], self.data["kpar"]))
        self.data["kcoord"] = self.data["kcoord"].reshape(2, -1).T
        
        self.kperp_norm = np.log10(self.data['kperp']) if self.model_opt['dataset']["data_dims"]["kperp"]["log"] else self.data['kperp']
        #self.kperp_key = "log10kperp" if self.model_opt['dataset']["data_dims"]["kperp"]["log"] else "kperp"
        #self.kperp_norm = getattr(self, self.params_norm)(self.kperp_norm, invert=False, **self.model.param_stats[self.kperp_key])
        
        self.kpar_norm = np.log10(self.data['kpar']) if self.model_opt['dataset']["data_dims"]["kpar"]["log"] else self.data['kpar']
        #self.kpar_key = "log10kpar" if self.model_opt['dataset']["data_dims"]["kpar"]["log"] else "kpar"
        #self.kpar_norm = getattr(self, self.params_norm)(self.kpar_norm, invert=False, **self.model.param_stats[self.kpar_key])
        self.kcoord_norm = np.array(np.meshgrid(self.kperp_norm, self.kpar_norm))
        self.kcoord_norm = self.kcoord_norm.reshape(2, -1).T

        self.norm_dict = {}
        self.norm_dict['min'] = np.array([item[1]['min'] for item in self.model.param_stats.items()])
        self.norm_dict['max'] = np.array([item[1]['max'] for item in self.model.param_stats.items()])
        self.norm_dict['mean'] = np.array([item[1]['mean'] for item in self.model.param_stats.items()])
        self.norm_dict['std'] = np.array([item[1]['std'] for item in self.model.param_stats.items()])
        
    def computeLikelihood(self, params):
        try:
            params = np.array(params)

            lognoise = params[self.noise_indices]
            noise = 10**lognoise

            logL = 0
            for i,band in enumerate(self.data['bands'].keys()):
                Pk_obs_minus_err = self.data['bands'][band][f"Pk_obs_minus_err"]
                z = self.data['bands'][band]["z_norm"]
                pred = self.predict(z, params[self.prior_indices])

                logL += (
                    -0.5*np.log(2*np.pi*(noise**2+(0.10*pred)**2)) 
                    - 0.5 * ( Pk_obs_minus_err - pred)**2
                    /(noise**2+(0.10*pred)**2)
                    ).sum()
            logL = float(logL)
            
            nDerived = [logL,]

            if self.emulator_xHI:
                xHI_pred = self.predict_xHI(params[self.prior_indices])
                nDerived += xHI_pred

            return logL, nDerived
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)
    
    def predict(self, z, params):
        kperp = self.kcoord_norm[:,0]
        kpar = self.kcoord_norm[:,1]
        params = np.array([z, np.nan, np.nan, *params])
        params=np.tile(params, (len(kperp), 1))
        params[:,1] = kperp
        params[:,2] = kpar
        #standardize or minmax normalize
        params = getattr(self, self.params_norm)(params, invert=False, **self.norm_dict)
        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model.net_g(params)
            pred = pred.detach().cpu().numpy()

            if self.target_log:
                pred = 10**pred
            if self.target_offset>0:
                pred = pred - self.target_offset
        return pred
    
    def get_prior_indices(self, prior_dict):
        prior_keys = list(prior_dict.keys())
        emulator_keys = list(self.model.param_stats.keys())
        n_data_dims = len(self.model_opt['dataset']['data_dims'])
        self.prior_indices = []
        for key in emulator_keys[n_data_dims:]: #find non-data_dim indices of emulator_keys in prior_keys
            self.prior_indices.append(prior_keys.index(key))

        self.noise_indices = []
        for key in prior_keys:
            if "noise" in key:
                self.noise_indices.append(prior_keys.index(key))
        


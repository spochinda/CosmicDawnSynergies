import os
import numpy as np

from scipy.io import loadmat
from scipy.constants import parsec, physical_constants
import hera_pspec as hp

import scipy.special as ssp
import CosmicDawnSynergies.itamar.radio_cutoff_calc as rad
from CosmicDawnSynergies.train_tools import Scaler
from .emulator_poweremu import *

#from tensorflow import keras
#from globalemu.eval import evaluate

import astropy.constants as c
import astropy.units as u
from scipy.interpolate import interp1d


def emulatorModel2d(emu, z, karr, params):
    z = np.log10(z) if emu.data_opt["data_dims_log"][0] else z
    karr = np.log10(karr) if emu.data_opt["data_dims_log"][1] else karr

    params = np.array([z, np.NaN, *params])
    params=np.tile(params, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return emu.predict(params)

def emulatorModel1d(emu, arr, params):
    arr = np.log10(arr) if emu.data_opt["data_dims_log"] else arr
    params = np.array([np.NaN, *params])
    params=np.tile(params, (len(arr), 1))
    params[:,0] = arr
    with torch.no_grad():
        params = torch.from_numpy(params).to(dtype=torch.float32)
        pred = emu.model(params)
        pred = pred.detach().cpu().numpy()
        if emu.data_opt["data_log"]:
            pred = 10**pred
        if emu.data_opt["data_dims_log"]:
            arr = 10**arr
    return arr, pred

class LikelihoodRadioBackground:
    def __init__(self, 
                 emulator,
                 prior_dict,
                 data_dims = ["log10nu_today",],
                 datapath='codes/itamar/LWA1_with_err.npy',
                 output_names = [r"\log L_\mathrm{Radio}",],
                 **kwargs
                 ):
        """
        Likelihood module for the LWA1/ARCADE2 constraints 
        (data from Table 2 of Dowell & Taylor (2018))
        """
        self.emulator = emulator
        self.prior_dict = prior_dict
        self.nu_obs, self.T_obs, self.dT_obs = np.load(datapath)
        self.output_names = output_names
        self.nDerived = len(self.output_names)
        self.scaler = Scaler(self.emulator.scale_opt)


        prior_keys = list(self.prior_dict.keys())
        emulator_keys = list(self.emulator.scale_opt.keys())
        self.prior_indices = []
        for key in emulator_keys: #find non-data_dim indices of emulator_keys in prior_keys
            if key not in data_dims:
                self.prior_indices.append(prior_keys.index(key))        
    
    def computeLikelihood(self, params):
        params = np.array(params)
        params = params[self.prior_indices]
        arr = self.nu_obs
        arr, T_model = emulatorModel1d(emu=self.emulator, arr=arr, params=params)
        dT_model = T_model*0.05

        P = 0.5 * (1 + ssp.erf( (self.T_obs - T_model) / np.sqrt(2) / np.sqrt(self.dT_obs**2+dT_model**2))) 
        logL = np.log(P).sum()
        return logL, [logL,]
    
    def predict(self, nu_today, params):
        nu_today = np.log10(nu_today) if self.emulator.data_opt["data_dims_log"] else nu_today
        nu_today = self.scaler.standardize(nu_today, **self.emulator.scale_opt["log10nu_today"]["stats"])

        params = np.array([np.NaN, *params])
        params=np.tile(params, (len(nu_today), 1))
        params[:,0] = nu_today
        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
            if self.emulator.data_opt["data_dims_log"]:
                nu_today = 10**nu_today
        return nu_today, pred

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


class LikelihoodXRB:
    def __init__(self, 
                 emulator,
                 prior_dict,
                 data_dims = ["log10E_min",],
                 output_names = [r"\log L_\mathrm{CXB,tot}",],
                 **kwargs    
                 ):
        """
        Likelihood module for the Chandra constraints 
        (data from Hickox+2006 and Harrison+2016)
        """
        self.emulator = emulator
        self.prior_dict = prior_dict
        self.output_names = output_names
        self.nDerived = len(self.output_names)
        self.scaler = Scaler(self.emulator.scale_opt)
        self.pre_compute()
        

        prior_keys = list(self.prior_dict.keys())
        emulator_keys = list(self.emulator.scale_opt.keys())
        self.prior_indices = []
        for key in emulator_keys: #find non-data_dim indices of emulator_keys in prior_keys
            if key not in data_dims:
                self.prior_indices.append(prior_keys.index(key))        

    def pre_compute(self):
        """precomputed variables not necessary to put inside and slow down computeLikelihood"""
        eV_toHz = physical_constants['electron volt-hertz relationship'][0]
        keV_toHz = eV_toHz*1e3
        sr_todeg2 = (180/np.pi)**2
        Mpc_tom = 1e6 * parsec
        Mpc_tocm = Mpc_tom * 1e2
        cm_toMpc = 1/Mpc_tocm
        
        self.units = keV_toHz * cm_toMpc**2 / sr_todeg2

        ###Data###
        self.X_limits = np.array([ #nu_min, nu_max, mean, std 
            #[0.5, 2, 8.15*1e-12, 0.58*1e-12], #Lehmer+2012 
            [1, 2, 1.04*1e-12, 0.14*1e-12], #Hickox & Markevitch (2006)
            [2, 8, 3.4*1e-12, 1.7*1e-12], #Hickox & Markevitch (2006)
            [8, 24, 6.013*1e-8/sr_todeg2, 0.14*1e-8/sr_todeg2], #Harrison et al. (2016) #6.773*1e-8/sr_todeg2, 0.348*1e-8/sr_todeg2
            [20, 50, 6.56*1e-8/sr_todeg2, 0.273*1e-8/sr_todeg2], #Harrison et al. (2016) #6.205*1e-8/sr_todeg2, 0.17*1e-8/sr_todeg2
            ])
        
        minE, maxE = min(self.X_limits[:,0]), max(self.X_limits[:,1])
        self.E_min = np.logspace(minE, maxE, 100)
  
    
    def computeLikelihood(self,params):
        params = np.array(params)
        params = params[self.prior_indices]

        E_min, XRB_pred = self.predict(E_min=self.E_min, params=params)
        E_min = np.log10(E_min) if self.emulator.data_opt["data_dims_log"] else E_min
        XRB_pred = np.log10(XRB_pred) if self.emulator.data_opt["data_log"] else XRB_pred
        XRB_pred = interp1d(E_min, XRB_pred, kind='linear', fill_value='extrapolate') #model interpolator

        logL = 0
        for xmin,xmax,XRB,std in self.X_limits: #integrate model from xmin to xmax and compare to data to get logL
            E_min = np.logspace(xmin, xmax, 100)
            E_min = np.log10(E_min) if self.emulator.data_opt["data_dims_log"] else E_min
            XRB_pred_ = XRB_pred(E_min)
            XRB_pred_ = np.trapz(XRB_pred_, E_min)
            P = 0.5 * (1 + ssp.erf( (XRB - XRB_pred_) / np.sqrt(2) / np.sqrt(std**2+(XRB_pred_*0.05)**2) ))
            logL += np.log(P)
        return logL, [logL,]
    
    def predict(self, E_min, params):
        E_min = np.log10(E_min) if self.emulator.data_opt["data_dims_log"] else E_min
        E_min = self.scaler.standardize(E_min, **self.emulator.scale_opt["log10E_min"]["stats"])

        params = np.array([np.NaN, *params])
        params=np.tile(params, (len(E_min), 1))
        params[:,0] = E_min
        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
            if self.emulator.data_opt["data_dims_log"]:
                E_min = 10**E_min
        return E_min, pred








class LikelihoodHERA:
    def __init__(self, 
                 emulator,
                 prior_dict,
                 files, 
                 data_dims = ["z", "log10k",],
                 output_names = [r"\log L_\mathrm{HERA}",],
                 **kwargs
                 ):
        self.warnings = kwargs.get("warnings", False)
        self.set_negative_to_zero = kwargs.get("set_negative_to_zero", True)
        self.decimate_data = kwargs.get("decimate_data", True)
        self.mask_to_emulator_range = kwargs.get("mask_to_emulator_range", True)
        self.k_name_in_scale_opt = kwargs.get("k_name_in_scale_opt", "log10k")
        self.z_name_in_scale_opt = kwargs.get("z_name_in_scale_opt", "z")
        
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
        return logL, [logL,]
    
    def predict(self, z, karr, params):
        z = np.log10(z) if self.emulator.data_opt["data_dims_log"][0] else z
        z = self.scaler.standardize(z, **self.emulator.scale_opt["z"]["stats"])
        karr = np.log10(karr) if self.emulator.data_opt["data_dims_log"][1] else karr
        karr = self.scaler.standardize(karr, **self.emulator.scale_opt["log10k"]["stats"])

        params = np.array([z, np.NaN, *params])
        params=np.tile(params, (len(karr), 1))
        params[:,1] = karr
        with torch.no_grad():
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
        return pred
        
    def extract_data(self, **kwargs):        
        if self.mask_to_emulator_range:
            z_index = list(self.emulator.scale_opt.keys()).index(self.z_name_in_scale_opt)
            z_opt = self.emulator.data_opt["lims_nsample"][z_index]
            zmin, zmax, _ = z_opt
            k_index = list(self.emulator.scale_opt.keys()).index(self.k_name_in_scale_opt)
            k_opt = self.emulator.data_opt["lims_nsample"][k_index]
            kmin, kmax, _ = k_opt
            kmin, kmax

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
                    print(f"AttributeError: Setting window funciton to identity matrix for z={z:.2f} in file {fn}")
                    wfn = np.identity(dsq.shape[0])
                try:
                    var = uvp.get_cov((band,blpair,polpair))[0].diagonal().real.copy()
                except AttributeError:
                    print(f"AttributeError: Getting variance from stats for z={z:.2f} in file {fn}")
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
                if self.decimate_data:
                    k_mag, dsq, std, wfn = self.decimate(k_mag, dsq, std, wfn)

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


class LikelihoodSARAS3:
    def __init__(self, 
                 emulator,
                 prior_dict,
                 data_dims = ["log10E_min",],
                 output_names = [r"\log L_\mathrm{SARAS3}"],
                 **kwargs
                 ):
        """
        Likelihood module for the SARAS3 constraints.
        """
        self.output_names = output_names
        self.datapath = datapath
        self.emupath = emupath
        #reduce_data
        self.freq, self.T_SARAS, self.weights, self.fg_fit, self.fg_fit_T_resid = np.loadtxt(datapath).T
        log_freq = np.log10(self.freq)
        self.reduced_freq = 2*((log_freq - log_freq.min())/ \
                          (log_freq.max()-log_freq.min())) - 1
        self.z = 1420/self.freq-1
        #setup
        self.model = keras.models.load_model(self.emupath + 'model.h5', compile=False)
        #self.predictor = evaluate(base_dir=self.emupath, model=self.model, z=self.z, gc=False, logs=[]) #0,1,2,3,7], )
        self.nDerived = len(self.output_names.items()) 

    def foreground(self, p):
        log_fit = 10**(np.sum([
                        p[i] * self.reduced_freq**i
                        for i in range(len(p))],
                        axis=0))
        return log_fit

    def computeLikelihood(self,p):
        fg = self.foreground(p[9:-1])
        noise = p[-1]
        
        assert False, "Need to fix this"
        signal = self.predictor(p[:9])[0]/1000 #select signal and convert to K

        logL = (
            -0.5*np.log(2*np.pi*(noise**2+(0.25*signal)**2)) 
            - 0.5 * (self.T_SARAS - fg - signal)**2
            /(noise**2+(0.25*signal)**2)
            ).sum()
        return logL#, [None]


class LikelihoodNeutralFraction:
    def __init__(self, 
                 #datapath='codes/itamar/LWA1_with_err.npy',
                 emupath='data/globalemu/xHI_emulator1/results/', 
                 output_names = {"logL_xHI": r"\log L_\mathrm{x_{HI}}"}
                 ):
        self.output_names = output_names
        #self.datapath = datapath
        self.emupath = emupath
        #data        
        #setup
        #self.model = keras.models.load_model(self.emupath + 'model.h5', compile=False)
        #self.predictor = evaluate(base_dir=self.emupath, model=self.model, gc=False, logs=[], z=[6.0, 7.0, 7.6]) #0,1,2,3,7], )
        self.nDerived = len(self.output_names.items()) 

            

    def computeLikelihood(self,p):
        assert False, "Need to fix this"
        xHI_emu, zs = self.predictor(p[:9]) 
        xHI_emu_sigma = 0.05
        xHI=np.array([0.09, 0.57, 0.855]) #[approx McGreer+15, Mason+18, Hoag+19]0.06
        xHI_sigma=np.array([0.08, 0.13, 0.075])#0.05

        logL = np.sum(-0.5*np.log(2*np.pi*(xHI_sigma**2+xHI_emu_sigma**2)) - 0.5 * (xHI - xHI_emu)**2/(xHI_sigma**2+xHI_emu_sigma**2))
            
        return logL#, [None]









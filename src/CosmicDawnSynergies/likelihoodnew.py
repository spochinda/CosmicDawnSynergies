import os
import sys
import numpy as np
import torch

from scipy.io import loadmat
from scipy.constants import parsec, physical_constants
import hera_pspec as hp

import scipy.special as ssp
import CosmicDawnSynergies.itamar.radio_cutoff_calc as rad
#from .emulator_poweremu import *
from CosmicDawnSynergies.train_tools import Scaler, poweremu_torch


#from tensorflow import keras
#from globalemu.eval import evaluate

import astropy.constants as c
import astropy.units as u
from scipy.interpolate import interp1d


def emulatorModel2d(emu, z, karr, params):
    z = np.log10(z) if emu.data_opt["data_dims_log"][0] else z
    karr = np.log10(karr) if emu.data_opt["data_dims_log"][1] else karr

    params = np.array([z, np.nan, *params])
    params=np.tile(params, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return emu.predict(params)

def emulatorModel1d(emu, arr, params):
    key = list(emu.data_opt["data_dims"][0].keys())[0]
    dim_log = emu.data_opt["data_dims"][0][key]["log"]
    data_log = emu.data_opt["data_log"]

    arr = np.log10(arr) if dim_log else arr
    params = np.array([np.nan, *params])
    params=np.tile(params, (len(arr), 1))
    params[:,0] = arr
    with torch.no_grad():
        params = torch.from_numpy(params).to(dtype=torch.float32)
        pred = emu.model(params)
        pred = pred.detach().cpu().numpy()
        if data_log:
            pred = 10**pred
        if dim_log:
            arr = 10**arr
    return arr, pred

class LikelihoodRadioBackground:
    def __init__(self, 
                 prior_dict,
                 emulator,
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

        nu_today, T_model = self.predict(nu_today=self.nu_obs, params=params)
        dT_model = T_model*0.05

        P = 0.5 * (1 + ssp.erf( (self.T_obs - T_model) / np.sqrt(2) / np.sqrt(self.dT_obs**2+dT_model**2))) 
        logL = np.log(P).sum()
        logL = float(logL)
        return logL, [logL,]

    def predict(self, nu_today, params):

        nu_today = np.log10(nu_today) if self.emulator.data_opt["data_dims"][0]["nu_today"]["log"] else nu_today

        params = np.array([np.nan, *params])
        params=np.tile(params, (len(nu_today), 1))
        params[:,0] = nu_today
        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
            if self.emulator.data_opt["data_dims"][0]["nu_today"]["log"]:
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
                 prior_dict,
                 emulator,
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
        self.preprocessing()
        

        prior_keys = list(self.prior_dict.keys())
        emulator_keys = list(self.emulator.scale_opt.keys())
        self.prior_indices = []
        for key in emulator_keys: #find non-data_dim indices of emulator_keys in prior_keys
            if key not in data_dims:
                self.prior_indices.append(prior_keys.index(key))        

    def preprocessing(self):
        """precomputed variables not necessary to put inside and slow down computeLikelihood"""
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
        self.E_min = np.geomspace(minE, maxE, 100)
  
    
    def computeLikelihood(self, params):
        try:
            params = np.array(params)
            params = params[self.prior_indices]

            E_min, XRB_pred = self.predict(E_min=self.E_min, params=params)
            E_min = np.log10(E_min) if self.emulator.data_opt["data_dims"][0]["E_min"]["log"] else E_min
            XRB_pred = np.log10(XRB_pred) if self.emulator.data_opt["data_log"] else XRB_pred
            XRB_pred = interp1d(E_min, XRB_pred, kind='linear', fill_value='extrapolate') #model interpolator

            logL = 0
            for xmin,xmax,XRB,std in self.X_limits: #integrate model from xmin to xmax and compare to data to get logL
                E_min = np.geomspace(xmin, xmax, 100)
                E_min = np.log10(E_min) if self.emulator.data_opt["data_dims"][0]["E_min"]["log"] else E_min
                XRB_pred_ = XRB_pred(E_min)
                
                E_min = 10**E_min if self.emulator.data_opt["data_dims"][0]["E_min"]["log"] else E_min
                XRB_pred_ = 10**XRB_pred_ if self.emulator.data_opt["data_log"] else XRB_pred_
                XRB_pred_ = XRB_pred_
                
                try:
                    XRB_pred_ = np.trapezoid(XRB_pred_, E_min*self.keV_toHz) * self.cm_toMpc**2 / self.sr_todeg2
                except Exception as e:
                    XRB_pred_ = np.trapz(XRB_pred_, E_min*self.keV_toHz) * self.cm_toMpc**2 / self.sr_todeg2
                P = 0.5 * (1 + ssp.erf( (XRB - XRB_pred_) / np.sqrt(2) / np.sqrt(std**2+(XRB_pred_*0.05)**2) ))
                logL += np.log(P)
            logL = float(logL)
            return logL, [logL,]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            assert False
    
    def predict(self, E_min, params):
        E_min = np.log10(E_min) if self.emulator.data_opt["data_dims"][0]["E_min"]["log"] else E_min
        E_min = self.scaler.standardize(E_min, **self.emulator.scale_opt["log10E_min"]["stats"])

        params = np.array([np.nan, *params])
        params=np.tile(params, (len(E_min), 1))
        params[:,0] = E_min
        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
            if self.emulator.data_opt["data_dims"][0]["E_min"]["log"]:
                E_min = 10**E_min
        return E_min, pred


class LikelihoodHERA:
    def __init__(self, 
                 prior_dict,
                 emulator,
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
        
        if self.emulator is not None:
            self.scaler = Scaler(self.emulator.scale_opt)

        if self.prior_dict is not None:
            prior_keys = list(self.prior_dict.keys())
            emulator_keys = list(self.emulator.scale_opt.keys())
            self.prior_indices = []
            for key in emulator_keys: #find non-data_dim indices of emulator_keys in prior_keys
                if key not in data_dims:
                    self.prior_indices.append(prior_keys.index(key))

        self.extract_data()
        
        
        
        


        #self.redshifts = np.unique(np.array([self.data[i]["z"] for i in range(len(self.data))]))
        self.redshifts = np.loadtxt("/cosma/apps/dp140/dc-poch1/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies/data/redshifts.txt")
        self.Trad_emu = poweremu_torch()
        self.Trad_emu.load_network("/cosma/apps/dp140/dc-poch1/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies/data/trained_emulators_poweremu/Trad_Arad_emu.pth")
        self.scaler_Trad = Scaler(scale_opt=self.Trad_emu.scale_opt)
        self.Ts_emu = poweremu_torch()
        self.Ts_emu.load_network("/cosma/apps/dp140/dc-poch1/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies/data/trained_emulators_poweremu/Ts_Arad_emu2.pth")
        self.scaler_Ts = Scaler(scale_opt=self.Ts_emu.scale_opt)
        
        self.output_names = [f"log10Trad_z{z:.2f}" for z in self.redshifts] + [f"log10Ts_z{z:.2f}" for z in self.redshifts]
        self.nDerived = len(self.output_names)
    
    @torch.no_grad()
    def get_log10Trad_log10Ts(self, z, p):
        try:
            #z_Trad = self.scaler_Trad.normalize(z, mean=self.scaler_Trad.scale_opt["z"]["stats"]["mean"], std=self.scaler_Trad.scale_opt["z"]["stats"]["std"])
            p_Trad = np.array([np.nan, *p])
            p_Trad=np.tile(p_Trad, (len(z), 1))
            p_Trad[:,0] = z
            p_Trad = self.scaler_Trad.transform(p_Trad, use_scale_opt=True)
            p_Trad = torch.from_numpy(p_Trad).to(dtype=torch.float32)
            log10Trad = self.Trad_emu.model(p_Trad)
            log10Trad = log10Trad.detach().cpu().numpy()
            log10Trad = log10Trad#[0]
            if self.Trad_emu.data_opt.get("data_log", False):
                log10Trad = 10**log10Trad
            if self.Trad_emu.data_opt.get("offset", False):
                log10Trad = log10Trad - self.Trad_emu.data_opt["offset"]
            if self.Trad_emu.data_opt.get("data_log", False):
                log10Trad = np.log10(log10Trad)

            
            #z_Ts = self.scaler_Ts.normalize(z, mean=self.scaler_Ts.scale_opt["z"]["stats"]["mean"], std=self.scaler_Ts.scale_opt["z"]["stats"]["std"])
            p_Ts = np.array([np.nan, *p])
            p_Ts=np.tile(p_Ts, (len(z), 1))
            p_Ts[:,0] = z
            p_Ts = self.scaler_Ts.transform(p_Ts, use_scale_opt=True)
            p_Ts = torch.from_numpy(p_Ts).to(dtype=torch.float32)
            log10Ts = self.Ts_emu.model(p_Ts)
            log10Ts = log10Ts.detach().cpu().numpy()
            log10Ts = log10Ts#[0]
            if self.Ts_emu.data_opt.get("data_log", False):
                log10Ts = 10**log10Ts
            if self.Ts_emu.data_opt.get("offset", False):
                log10Ts = log10Ts - self.Ts_emu.data_opt["offset"]
            if self.Ts_emu.data_opt.get("data_log", False):
                log10Ts = np.log10(log10Ts)
            

            nDerived_params = [*log10Trad, *log10Ts]
            
            return nDerived_params
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)


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

            
            nDerived_params = self.get_log10Trad_log10Ts(z=self.redshifts, p=params)
            return logL, [*nDerived_params]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)
    
    def predict(self, z, karr, params):
        try:
            dim_log_0 = self.emulator.data_opt["data_dims"][0]["z"]["log"]
            dim_log_1 = self.emulator.data_opt["data_dims"][1]["k"]["log"]
            data_log = self.emulator.data_opt["data_log"]
            offset = self.emulator.data_opt.get("offset", False)

            z = np.log10(z) if dim_log_0 else z
            karr = np.log10(karr) if dim_log_1 else karr

            params = np.array([z, np.nan, *params])
            params=np.tile(params, (len(karr), 1))
            params[:,1] = karr
            with torch.no_grad():
                params = self.scaler.transform(params, use_scale_opt=True)
                params = torch.from_numpy(params).to(dtype=torch.float32)
                pred = self.emulator.model(params)
                pred = pred.detach().cpu().numpy()

                if dim_log_1:
                    karr = 10**karr
                if data_log:
                    pred = 10**pred
                if offset:
                    pred = pred - offset
                
                factor = 1. #karr**3/(2*np.pi**2)
                pred =  factor * pred
            return pred
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        
    def extract_data(self, **kwargs):        
        if self.mask_to_emulator_range:
            z_index = list(self.emulator.scale_opt.keys()).index(self.z_name_in_scale_opt)
            zmin, zmax = self.emulator.data_opt["data_dims"][z_index]["z"]["lims"]
            k_index = list(self.emulator.scale_opt.keys()).index(self.k_name_in_scale_opt)
            kmin, kmax = self.emulator.data_opt["data_dims"][k_index]["k"]["lims"]

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
                if "H1C_IDR3" in file: #self.decimate_data:
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



class LikelihoodSARAS3:
    def __init__(self, 
                 prior_dict,
                 emulator,
                 file,
                 data_dims,
                 poly_coeff,
                 noise,
                 output_names = [r"\log L_\mathrm{SARAS3}"],
                 **kwargs
                 ):
        """
        Likelihood module for the SARAS3 constraints.
        """
        self.convert_mK_to_K = kwargs.get("convert_mK_to_K", True)
        self.emulator = emulator
        self.prior_dict = prior_dict
        self.file = file
        self.data_dims = data_dims
        self.output_names = output_names
        self.nDerived = len(self.output_names)
        self.scaler = Scaler(self.emulator.scale_opt)

        prior_keys = list(self.prior_dict.keys())
        emulator_keys = list(self.emulator.scale_opt.keys())
        self.astro_indices = []
        self.emulator_indices = []
        self.data_dims_indices = []
        for i,key in enumerate(emulator_keys): #find non-data_dim indices of emulator_keys in prior_keys
            if key in prior_keys:
                self.astro_indices.append(prior_keys.index(key))
                self.emulator_indices.append(i)
            else:
                self.data_dims_indices.append(i)
        
        self.poly_coeff_indices = [prior_keys.index(name) for name in poly_coeff]
        self.noise_indices = [prior_keys.index(name) for name in noise]

        self.preprocessing()
    
    def preprocessing(self):
        self.freq, self.T_SARAS, self.weights, self.fg_fit, self.fg_fit_T_resid = np.loadtxt(self.file).T
        log_freq = np.log10(self.freq)
        self.reduced_freq = 2 * ((log_freq - log_freq.min()) / (log_freq.max()-log_freq.min())) - 1
        self.redshifts = 1420/self.freq - 1

    def foreground(self, params):
        log10Tfg = np.sum(np.array([a_i * self.reduced_freq**i for i,a_i in enumerate(params)]), axis=0)
        Tfg = 10**log10Tfg
        return Tfg

    def computeLikelihood(self, params):
        params = np.array(params)

        Tfg = self.foreground(params[self.poly_coeff_indices])
        std = params[self.noise_indices]

        redshifts, T21_pred = self.predict(params[self.astro_indices])

        logL = (
            -0.5*np.log(2*np.pi*(std**2+(0.25*T21_pred)**2)) 
            - 0.5 * (self.T_SARAS - Tfg - T21_pred)**2
            /(std**2+(0.25*T21_pred)**2)
            ).sum()
        logL = float(logL)
        return logL, [logL,]
    
    def predict(self, params):
        redshifts = np.log10(self.redshifts) if self.emulator.data_opt["data_dims"][0]["z"]["log"] else self.redshifts

        #params = np.array([np.nan, *params])
        #params=np.tile(params, (len(redshifts), 1))
        #params[:,0] = redshifts
        params_ = np.empty((redshifts.size, params.size + len(self.data_dims)))
        params_[:,self.data_dims_indices] = redshifts[:,None]
        params_[:,self.emulator_indices] = params
        params = params_

        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.emulator.data_opt["data_log"]:
                pred = 10**pred
            if self.emulator.data_opt["data_dims"][0]["z"]["log"]:
                redshifts = 10**redshifts
            if self.convert_mK_to_K:
                pred *= 1e-3
        return redshifts, pred

class LikelihoodPowerSpectrum:
    def __init__(self, 
                 prior_dict,
                 emulator,
                 files, 
                 data_dims = ["z", "log10k",],
                 output_names = [r"\log L_\mathrm{dsq}",],
                 **kwargs
                 ):
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



class LikelihoodSDC3b:
    def __init__(self, 
                 prior_dict,
                 emulator,
                 files,
                 averaged_noise_files,
                 noise,
                 xHI_file = None,  
                 data_dims = ["z", "kperp", "kpar"],
                 output_names = [r"\log L_\mathrm{dsq}",],
                 **kwargs
                 ):
        
        self.emulator = emulator
        self.prior_dict = prior_dict
        self.files = files
        self.averaged_noise_files = averaged_noise_files
        self.xHI_file = xHI_file
        self.output_names = output_names
        self.nDerived = len(self.output_names)
        if self.emulator is not None:
            self.scaler = Scaler(self.emulator.scale_opt)

        if self.prior_dict is not None:
            prior_keys = list(self.prior_dict.keys())
            emulator_keys = list(self.emulator.scale_opt.keys())
            self.astro_indices = []
            self.emulator_indices = []
            self.data_dims_indices = []
            for i,key in enumerate(emulator_keys): #find non-data_dim indices of emulator_keys in prior_keys
                if key in prior_keys:
                    self.astro_indices.append(prior_keys.index(key))
                    self.emulator_indices.append(i)
                else:
                    self.data_dims_indices.append(i)
            
            self.noise_indices = [prior_keys.index(name) for name in noise]
        
        self.extract_data()

        if self.xHI_file is not None:
            self.xHI_emulator = poweremu_torch()
            self.xHI_emulator.load_network(xHI_file)
            self.xHI_scaler = Scaler(scale_opt=self.xHI_emulator.scale_opt)
            self.z_xHI = np.array([(1420/((196.+181.)/2))-1, (1420/((181.+166.)/2))-1, (1420/((166.+151.)/2))-1])
            self.output_names = [f"xHI_z{z:.2f}" for z in self.z_xHI]
            self.nDerived = len(self.output_names)

            self.xHI_astro_indices = []
            emulator_keys = list(self.xHI_emulator.scale_opt.keys())
            for i,key in enumerate(emulator_keys):
                if key in prior_keys:
                    self.xHI_astro_indices.append(prior_keys.index(key))

    
    def predict_xHI(self, params):
        z = np.log10(self.z_xHI) if self.xHI_emulator.data_opt["data_dims"][0]["z"]["log"] else self.z_xHI
        params = np.array([np.nan, *params])
        params = np.tile(params, (len(z), 1))
        params[:,0] = z
        with torch.no_grad():
            params = self.xHI_scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.xHI_emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if self.xHI_emulator.data_opt["data_log"]:
                pred = 10**pred
        return pred.tolist()

    def computeLikelihood(self, params):
        params = np.array(params)

        std = 10**params[self.noise_indices]

        logL = 0
        for i in range(len(self.files)):
            for j in range(len(self.files[i])):
                Pk = self.data[f"PS{i+1}"].get(f"Pk{j}", None)
                z = self.data[f"PS{i+1}"][f"z{j}"]
                pred = self.predict(params[self.astro_indices], z)
                residual = Pk - pred

                logL += (
                    -0.5*np.log(2*np.pi*(std**2+(0.10*pred)**2)) 
                    - 0.5 * (residual)**2
                    /(std**2+(0.10*pred)**2)
                    ).sum()
        logL = float(logL)
        
        if self.xHI_file is not None:
            nDerived = self.predict_xHI(params[self.xHI_astro_indices]) # xHI
        else:
            nDerived = [logL,]

        return logL, nDerived
    
    def predict(self, params, z):
        is_log_z = self.emulator.data_opt["data_dims"][0]["z"]["log"]
        is_log_kperp = self.emulator.data_opt["data_dims"][1]["kperp"]["log"]
        is_log_kpar = self.emulator.data_opt["data_dims"][2]["kpar"]["log"]
        data_log = self.emulator.data_opt["data_log"]

        kcoord = self.data["kcoord"]
        kperp = kcoord[:,0]
        kpar = kcoord[:,1]
        
        z = np.log10(z) if is_log_z else z
        kperp = np.log10(kperp) if is_log_kperp else kperp
        kpar = np.log10(kpar) if is_log_kpar else kpar

        #params_xHI = np.array([z, np.nan, np.nan, *params])
        params = np.array([z, np.nan, np.nan, *params])
        params=np.tile(params, (len(kperp), 1))
        params[:,1] = kperp
        params[:,2] = kpar
        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.emulator.model(params)
            pred = pred.detach().cpu().numpy()
            if data_log:
                pred = 10**pred
        #if self.xHI_file
        return pred
        
    def extract_data(self, **kwargs):
        self.data = dict()
        for i,PS in enumerate(self.files):
            self.data[f"PS{i+1}"] = {}
            for j,(file,avg_noise_file) in enumerate(zip(PS, self.averaged_noise_files)):
                f2,f1 = os.path.basename(os.path.splitext(file)[0]).split("Pk_PS")[-1][2:].split("_")
                f2, f1 = float(f2), float(f1)
                f = (f2 + f1) / 2
                z = (1420/f) - 1
                averaged_noise = np.loadtxt(avg_noise_file)
                self.data[f"PS{i+1}"][f"z{j}"] = z
                self.data[f"PS{i+1}"][f"Pk{j}"] = np.loadtxt(file)
                self.data[f"PS{i+1}"][f"Pk{j}"] -= averaged_noise
                self.data[f"PS{i+1}"][f"Pk{j}"] = self.data[f"PS{i+1}"][f"Pk{j}"].reshape(-1)
        self.data["kperp"] = np.loadtxt("/home/sp2053/rds/rds-uksrc-eElmlMT25pY/yl871/SKA_SDC3b/PS1_PS2_Data/bins_kper.txt")
        self.data["kpar"] = np.loadtxt("/home/sp2053/rds/rds-uksrc-eElmlMT25pY/yl871/SKA_SDC3b/PS1_PS2_Data/bins_kpar.txt")
        self.data["kcoord"] = np.array(np.meshgrid(self.data["kperp"], self.data["kpar"]))
        self.data["kcoord"] = self.data["kcoord"].reshape(2, -1).T

    
if __name__ == "__main__":
    #try LikelihoodSDC3b
    import os
    data_path = "/home/sp2053/rds/rds-uksrc-eElmlMT25pY/yl871/SKA_SDC3b/PS1_PS2_Data/"
    likelihood_kwargs = {
        "files": [[os.path.join(data_path, "Pk_PS1_181.0_195.9.txt"), os.path.join(data_path, "Pk_PS1_166.0_180.9.txt"), os.path.join(data_path, "Pk_PS1_151.0_165.9.txt")],
                    [os.path.join(data_path, "Pk_PS2_181.0_195.9.txt"), os.path.join(data_path, "Pk_PS2_166.0_180.9.txt"), os.path.join(data_path, "Pk_PS2_151.0_165.9.txt")]],
        "averaged_noise_files": [os.path.join(data_path, "Pk_PS_averaged_noise_181.0_195.9.txt"), os.path.join(data_path, "Pk_PS_averaged_noise_166.0_180.9.txt"), os.path.join(data_path, "Pk_PS_averaged_noise_151.0_165.9.txt")],
        "emulator": None,#"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/trained_emulators_poweremu/powerspec3.pth",
        "data_dims": ["z", "kperp", "kpar"],
        "noise": {"noise": [0.01, 0.02]},
        }

    likesdc3b = LikelihoodSDC3b(prior_dict=None, **likelihood_kwargs)
    print(f"likesdc3b data: {likesdc3b.data}, keys {likesdc3b.data.keys()}")
    #meshgrid of self.data["kperp"] and self.data["kpar"] with shape (2,kperp.size,kpar.size)
    kperp = likesdc3b.data["kperp"]
    kpar = likesdc3b.data["kpar"]
    grid = np.array(np.meshgrid(kperp, kpar))
    print(f"grid shape: {grid.shape}")
    grid = grid.reshape(2, -1).T
    print(f"grid shape: {grid.shape}")
    Pk = likesdc3b.data["PS1"]["Pk1"]
    for k,v in likesdc3b.data["PS1"].items():
        try:
            print(f"{k}: {v.shape}, min: {v.min()}, max: {v.max()}")
        except:
            pass


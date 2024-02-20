import numpy as np

from scipy.io import loadmat
from scipy.constants import parsec, physical_constants

import scipy.special as ssp
import codes.itamar.radio_cutoff_calc as rad
from codes.emulator_poweremu import *
from margarine.maf import MAF

from tensorflow import keras
from globalemu.eval import evaluate

import astropy.constants as c
import astropy.units as u


def emulatorModel2d(emu, z, karr, p):
    par0 = np.array([z, np.NaN, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return emu.predict(params)

def emulatorModel1d(emu, arr, p):
    par0 = np.array([np.NaN, *p])
    params=np.tile(par0, (len(arr), 1))
    params[:,0] = arr
    return emu.predict(params)

class LikelihoodRadioBackground:
    """Likelihood module for the LWA1/ARCADE2 constraints 
    (data from Table 2 of Dowell & Taylor (2018))"""
    def __init__(self, 
                 z_cutoff=6.01,
                 datapath='codes/itamar/LWA1_with_err.npy',
                 emupath="data/trained_emulators_poweremu/SFR_emu_fullz_n70_l50505050_t0.0001_o0.pkl", #'data/trained_emulators_poweremu/SFR1_emu_n400_l80808080_t1e-05_o0.pkl', 
                 mafpath='data/margarine/LWA_MAF.pkl',
                 maf_logZ = -0.05650523374863248,
                 use_MAFs=False,
                 output_names = {"logL_RadioBackground": r"\log L_\mathrm{Radio}"}
                 ):
        self.output_names = output_names
        self.datapath = datapath
        self.emupath = emupath
        self.mafpath = mafpath

        self.maf_logZ = maf_logZ
        self.use_MAFs = use_MAFs
        self.z_cutoff = z_cutoff
        self.z_array = np.arange(6, 40, 0.1) #hardcoded redshift array 6-40

        self.SFR_emu = poweremu(loadfile=self.emupath, preprocesss_log_x=False, tol=1e-5, offset=0)
        self.Tradio_emu = poweremu(loadfile="data/trained_emulators_poweremu/Tradio_emu_n200_l50505050_t0.0001_o0.pkl", preprocesss_log_x=False, preprocess_y=False, tol=1e-4, offset=0)
        self.maf = MAF.load(self.mafpath)
        self.nu_obs, self.T_obs, self.dT_obs = np.load(self.datapath)
        self.nDerived = len(self.output_names.items()) #len(self.nu_obs) + 1 if not self.use_MAFs else 1 #Tradios + logLLWA     

    def T_radio_today(self,z_dense, sfr_dense, cut_sfr=True):
        nu_today = np.logspace(-2, 1.1, 100)*10**9 * u.Hz # Hz
        lambda_today  = c.c/nu_today
        log_nu, log_sed = rad.get_radio_sed('power_law')
        dz = abs(z_dense[1] - z_dense[0])
        T = np.zeros_like(nu_today.value)
        
        if cut_sfr: 
            z_dense = z_dense[sfr_dense > 10**(-7)]
            sfr_dense = sfr_dense[sfr_dense > 10**(-7)]

        for t_idx, (ldba, nu) in enumerate(zip(lambda_today, nu_today)):
            factor = (  ldba**2/(2*c.k_B)  )*(  c.c/(4*np.pi)  )
            for z_idx, z in enumerate(z_dense):
                sfr = sfr_dense[z_idx]
                Hz = rad.Hubble_const(z) * u.km/u.s/u.Mpc
                A = (1/Hz)*(1/(1+z)) * dz
                log_nu_emmit = np.log10(nu.to(u.Hz).value*(1 + z))
                log_sed_interp = np.interp(log_nu_emmit, log_nu, log_sed)
                val = A*factor*10**(log_sed_interp)*(u.W  /u.Hz) *sfr /(u.Mpc)**3
                #T[z_idx:, t_idx] += val.to(u.K).value
                T[t_idx] += val.to(u.K).value

        return nu_today, T

    def computeLikelihood_old(self, p):
        if not self.use_MAFs:
            fr = 10**p[7] #index for fradio
            z_dense = np.arange(6, 40, 0.1)
            SFR_predict = emulatorModel1d(emu=self.SFR_emu, arr=self.z_array, p=p)
            sfr_dense = 10**(np.interp(z_dense, self.z_array, 
                                       np.log10(SFR_predict)  #interpolate in logspace
                                       ))

            nu_today, T_today = self.T_radio_today(z_dense[::-1], sfr_dense[::-1]) #rad.get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
            #T_model = np.mean(T_today, axis=0) * fr
            T_model = T_today * fr 
            T_model_interp = np.interp(self.nu_obs, nu_today.value, T_model)
            dT_model_interp = T_model_interp*0.25 #25 percent error

            P = 0.5 * (1 + ssp.erf( (self.T_obs - T_model_interp) / np.sqrt(2) / np.sqrt(self.dT_obs**2+dT_model_interp**2))) 
            if 0 in P:
                logL=-np.inf
            else:    
                logL = np.log(P).sum()
        else:
            T_model_interp = None #np.zeros(len(self.T_obs))
            logL = self.maf.log_like(p[:9], self.maf_logZ) #insert real evidence
        return logL#, T_model_interp
    
    def computeLikelihood(self, p):
        if not self.use_MAFs:
            nu_today = np.logspace(-2, 1.1, 100)#*10**9 * u.Hz # Hz
            T_model = 10**emulatorModel1d(emu=self.Tradio_emu, arr=np.log10(nu_today), p=p)

            T_model_interp = np.interp(self.nu_obs, nu_today.value, T_model)
            dT_model_interp = T_model_interp*0.05 #25 percent error

            P = 0.5 * (1 + ssp.erf( (self.T_obs - T_model_interp) / np.sqrt(2) / np.sqrt(self.dT_obs**2+dT_model_interp**2))) 
            if 0 in P:
                logL=-np.inf
            else:    
                logL = np.log(P).sum()
        else:
            T_model_interp = None #np.zeros(len(self.T_obs))
            logL = self.maf.log_like(p[:9], self.maf_logZ) #insert real evidence
        return logL#, T_model_interp




class LikelihoodXRB:
    """Likelihood module for the Chandra constraints 
    (data from Hickox+2006 and Harrison+2016)"""
    def __init__(self, 
                 #datapath='codes/itamar/LWA1_with_err.npy',
                 nu_lims = [0.4, 55],
                 nu_keV_path ="data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_nu_mat.mat",
                 emupath='data/trained_emulators_poweremu/CXBlog10_emu_n400_l50505050_t1e-05_o0.pkl', 
                 mafpath='data/margarine/Chandra_MAF.pkl',
                 maf_logZ = -0.09300543633401404,
                 use_MAFs=False,
                 output_names = {"logL_XRB": r"\log L_\mathrm{Chandra,tot}"}
                 ):
        self.output_names = output_names
        self.nu_lims = nu_lims
        self.nu_keV_path = nu_keV_path #loadmat(nu_keV_path)["nu_keV"][0]
        self.emupath = emupath
        self.mafpath = mafpath
        self.maf_logZ = maf_logZ
        self.use_MAFs = use_MAFs
        self.XRB_emu = poweremu(loadfile=self.emupath, preprocesss_log_x=False, tol=1e-5, offset=0)
        self.maf = MAF.load(self.mafpath)
        
        self.pre_compute()
        self.nDerived = len(self.output_names.items()) #len(self.X_limits) + 1 if not self.use_MAFs else 1 #CXBs + logLChandra


        #nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.1)#55)
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

        nu_keV = loadmat(self.nu_keV_path)["nu_keV"][0]
        nu_mask = (nu_keV >self.nu_lims[0]) & (nu_keV <self.nu_lims[1])
        self.nu_keV = nu_keV[nu_mask]

        self.numin_index = np.array([np.where(self.nu_keV > numin_obs_keV)[0][0] for numin_obs_keV in self.X_limits[:,0]], dtype=int)
        self.numax_index = np.array([np.where(self.nu_keV < numax_obs_keV)[0][-1] for numax_obs_keV in self.X_limits[:,1]], dtype=int)

        self.deltanu_obs = np.array([
            self.nu_keV[indEmin_obs + 1 : indEmax_obs + 1] - self.nu_keV[indEmin_obs : indEmax_obs] for indEmin_obs,indEmax_obs in zip(self.numin_index, self.numax_index)# X_limits[:,0:2].astype(int)
            ], dtype=object)
  
    
    def computeLikelihood(self,p):
        if not self.use_MAFs:
            logL = 0
            XRB_pred0 = emulatorModel1d(emu=self.XRB_emu,arr=np.log10(self.nu_keV), p=p[:9])
            sum_XRB_pred =np.array([
                np.sum(XRB_pred0[indEmin_obs:indEmax_obs] * deltanuobs * self.units) for indEmin_obs,indEmax_obs,deltanuobs in zip(self.numin_index, self.numax_index, self.deltanu_obs)
                ])
            P = 0.5 * (1 + ssp.erf( (self.X_limits[:,2] - sum_XRB_pred) / np.sqrt(2) / np.sqrt(self.X_limits[:,3]**2+(sum_XRB_pred*0.05)**2) ))
            if 0 in P:
                logL=-np.inf
            else:    
                logL = np.log(P).sum()
        else:
            sum_XRB_pred = None #np.zeros(len(self.X_limits))
            logL = self.maf.log_like(p, self.maf_logZ) #insert real evidence
        return logL#, sum_XRB_pred



class LikelihoodSARAS3:
    """Likelihood module for the Chandra constraints 
    (data from Hickox+2006 and Harrison+2016)"""
    def __init__(self, 
                 datapath='data/SARAS3/SARAS_3_averaged_spectrum.txt',
                 emupath='data/globalemu/emulator14/results/', 
                 mafpath='data/margarine/SARAS_MAF.pkl',
                 maf_logZ = -158.8493155641061,
                 use_MAFs=False,
                 output_names = {"logL_SARAS": r"\log L_\mathrm{SARAS3}", "filler": "filler"}
                 ):
        self.output_names = output_names
        self.datapath = datapath
        self.emupath = emupath
        self.mafpath = mafpath
        self.maf_logZ = maf_logZ
        self.use_MAFs = use_MAFs
        #reduce_data
        self.freq, self.T_SARAS, self.weights, self.fg_fit, self.fg_fit_T_resid = np.loadtxt(datapath).T
        log_freq = np.log10(self.freq)
        self.reduced_freq = 2*((log_freq - log_freq.min())/ \
                          (log_freq.max()-log_freq.min())) - 1
        self.z = 1420/self.freq-1
        #setup
        self.model = keras.models.load_model(self.emupath + 'model.h5', compile=False)
        self.predictor = evaluate(base_dir=self.emupath, model=self.model, z=self.z, gc=False, logs=[]) #0,1,2,3,7], )
        self.maf = MAF.load(self.mafpath)
        self.nDerived = len(self.output_names.items()) #2 if not self.use_MAFs else 1#logLSARAS
        
        #self.pre_compute()
        
    
    def pre_compute(self):
        """precomputed variables not necessary to put inside and slow down computeLikelihood"""

    def foreground(self, p):
        log_fit = 10**(np.sum([
                        p[i] * self.reduced_freq**i
                        for i in range(len(p))],
                        axis=0))
        return log_fit

    def computeLikelihood(self,p):
        if not self.use_MAFs:
            fg = self.foreground(p[9:-1])
            noise = p[-1]
            
            signal = self.predictor(p[:9])[0]/1000 #select signal and convert to K

            logL = (
                -0.5*np.log(2*np.pi*(noise**2+(0.25*signal)**2)) 
                - 0.5 * (self.T_SARAS - fg - signal)**2
                /(noise**2+(0.25*signal)**2)
                ).sum()
        else:
            logL = self.maf.log_like(p, -158.8493155641061 )
        return logL#, [None]


class LikelihoodNeutralFraction:
    def __init__(self, 
                 #datapath='codes/itamar/LWA1_with_err.npy',
                 emupath='data/globalemu/xHI_emulator1/results/', 
                 #mafpath='data/margarine/LWA_MAF.pkl',
                 #maf_logZ = -0.05650523374863248,
                 use_MAFs=False,
                 output_names = {"logL_xHI": r"\log L_\mathrm{x_{HI}}"}
                 ):
        self.output_names = output_names
        #self.datapath = datapath
        self.emupath = emupath
        #self.mafpath = mafpath
        #self.maf_logZ = maf_logZ
        self.use_MAFs = use_MAFs
        #data        
        #setup
        self.model = keras.models.load_model(self.emupath + 'model.h5', compile=False)
        self.predictor = evaluate(base_dir=self.emupath, model=self.model, gc=False, logs=[], z=[6.0, 7.0, 7.6]) #0,1,2,3,7], )
        self.nDerived = len(self.output_names.items()) #2 if not self.use_MAFs else 1#logLSARAS

        if self.use_MAFs!=False:
            assert "MAFs not supported at the moment. use_MAFs must be False"
            

    def computeLikelihood(self,p):
        if not self.use_MAFs:
            xHI_emu, zs = self.predictor(p[:9]) 
            xHI_emu_sigma = 0.05
            xHI=np.array([0.09, 0.57, 0.855]) #[approx McGreer+15, Mason+18, Hoag+19]0.06
            xHI_sigma=np.array([0.08, 0.13, 0.075])#0.05

            logL = np.sum(-0.5*np.log(2*np.pi*(xHI_sigma**2+xHI_emu_sigma**2)) - 0.5 * (xHI - xHI_emu)**2/(xHI_sigma**2+xHI_emu_sigma**2))
            
        else:
            pass
            #logL = self.maf.log_like(p, -158.8493155641061 )
        return logL#, [None]









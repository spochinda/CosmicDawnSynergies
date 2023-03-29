path = "/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

import numpy as np
from scipy.constants import parsec, physical_constants
from codes.emulator_poweremu import *
from codes.likelihood_hera import *
from scipy.io import loadmat
import scipy.special as ssp
import codes.itamar.radio_cutoff_calc as rad
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

paramNames = [
             "log10fstarII",
             "log10fstarIII",
             "log10Vc",
             "log10fX",
             "alpha",
             "nu_0",
             #"zeta",
             "tau",
             "log10fradio",
             "pop",
             #"feed",
             #"delay"
             ]

texDict = {"log10fstarII": r"$\log_{10} f_{\rm star, II}$",
           "log10fstarIII": r"$\log_{10} f_{\rm star, III}$",
           "log10Vc": r"$V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "alpha": r"$\alpha$",
           "nu_0": r"$\nu_{\rm 0}$",
           "tau": r"$\tau$",
           "log10fradio": r"$\log_{10} f_{\rm r}$",
           "pop": r"$\rm pop$",
           #"feed": r"$\rm feed$",
           #"delay": r"$\rm delay$",
           }

priorDict_HERA4 = {
             "log10fstarII": np.log10([1e-3, 0.5]).tolist(),
             "log10fstarIII": np.log10([1e-3, 0.5]).tolist(),
             "log10Vc": np.log10([4.2, 100]).tolist(),
             "log10fX": np.log10([1e-3, 1e2]).tolist(),
             "alpha": [-0.5, 2.5],#[-1, -1.3, -1.5],
             "nu_0": [-0.5, 16.5],#[100:100:1500, 2000, 3000],
             #"zeta": [4.3641, 3369.6118],
             #"tau": [0.02351, 0.10198],#??
             "tau": [0.054-3*0.007, 0.054+3*0.007],#??
             "log10fradio": np.log10([1e1, 1e5]).tolist(),##############
             "pop": [-0.5, 2.5],#[231, 232, 233],#??
             #"feed": [-0.5, 1.5],#[0, 1]
             #"delay": [-0.5, 1.5], #[0, 0.75]
             }

discrete_params = {
            "alpha": [1, 1.3, 1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            #"feed": [0, 1],
            #"delay": [0, 0.75]

}


# Load emulators
P = poweremu(loadfile=path+"data/trained_emulators_poweremu/Deltasq_emu_PL9_n1000_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
XRB_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/XRB_emu_PL9_n60_l40404040_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)
SFR_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/SFR_emu_PL9_n60_l40404040_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)
TS_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_Ts_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_TK_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_Trad_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)

def TS_TK_Trad_from_emulators(p, z=8):
    par0 = np.array([z, *p])
    TS = TS_emu.predict(par0)
    TK = TK_emu.predict(par0)
    TR = TR_emu.predict(par0)
    return TS, TK, TR

def emulatorModel2d(z, karr, p):
    par0 = np.array([z, np.NaN, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return P.predict(params)

def emulatorModel1d(emu, arr, p):
    par0 = np.array([np.NaN, *p])
    params=np.tile(par0, (len(arr), 1))
    params[:,0] = arr
    return emu.predict(params)

eV_toHz = physical_constants['electron volt-hertz relationship'][0]
keV_toHz = eV_toHz*1e3
sr_todeg2 = (180/np.pi)**2
Mpc_tom = 1e6 * parsec
Mpc_tocm = Mpc_tom * 1e2
cm_toMpc = 1/Mpc_tocm
XRB_limits = {
    '1 - 2 keV': {
        'mean': 1.04*1e-12,
        'std': 0.14*1e-12,
        }, #Hickox & Markevitch (2006)
    '2 - 8 keV': {
        'mean': 3.4*1e-12,
        'std': 1.7*1e-12,
        }, #Hickox & Markevitch (2006)
    '20 - 50 keV': {
        'mean': 6.205*1e-8/sr_todeg2,
        'std': 0.17*1e-8/sr_todeg2,
        }, #Harrison et al. (2016)
    '8 - 24 keV': {
        'mean': 6.773*1e-8/sr_todeg2,
        'std': 0.348*1e-8/sr_todeg2,
        }, #Harrison et al. (2016)
}

nu_keV = loadmat(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_nu_mat.mat")["nu_keV"][0]
nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.1)#55)
nu_keV = nu_keV[nu_mask]

def like_Chandra(p):
    logL = 0
    for i,key in enumerate(XRB_limits.keys()):
        numin_obs_keV, numax_obs_keV = np.array(key[:-4].split(" - "),dtype=float)
        indEmin_obs = np.where(nu_keV > numin_obs_keV)[0][0]
        indEmax_obs = np.where(nu_keV < numax_obs_keV)[0][-1]
        deltanu_obs = nu_keV[indEmin_obs + 1 : indEmax_obs + 1] - nu_keV[indEmin_obs : indEmax_obs]
        XRB_pred = emulatorModel1d(emu=XRB_emu,arr=np.log(nu_keV), p=p)
        sum_XRB_pred = np.sum(XRB_pred[indEmin_obs:indEmax_obs] * deltanu_obs * keV_toHz * cm_toMpc**2 / sr_todeg2) #(erg / s / Hz / cm^2 /sr) * (keV) * (Hz/keV) / (deg^2/sr)= erg / s /cm^2 / deg^2
        P = 0.5 * (1 + ssp.erf( (XRB_limits[key]["mean"] - sum_XRB_pred) / np.sqrt(2) / np.sqrt(XRB_limits[key]["std"]**2+(sum_XRB_pred*0.1)**2)))
        if P==0:
            logL += -np.inf
        else:
            logL += np.log(P)
    
    return logL

[nu_obs, T_obs, dT_obs] = np.load(path+'codes/itamar/LWA1_with_err.npy')
def like_LWA(p, z_cutoff=7):
    fr = 10**p[7] #index 7 is logfradio
    z_dense = np.linspace(z_cutoff-0.01, z_cutoff+0.01,2)
    sfr_dense = 10**(np.interp(z_dense, [6,7,8], np.log10( 
        emulatorModel1d(emu=SFR_emu, arr=[6,7,8], p=p)
    ) )) 

    nu_today, T_today = rad.get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
    T_model = np.mean(T_today, axis=0) * fr
    
    T_model_interp = np.interp(nu_obs, nu_today.value, T_model)
    dT_model_interp = T_model_interp*0.1

    P = 0.5 * (1 + ssp.erf( (T_obs - T_model_interp) / np.sqrt(2) / np.sqrt(dT_obs**2+dT_model_interp**2))) 
    if 0 in P:
        logL=-np.inf
    else:    
        logL = np.log(P).sum()
    
    return logL

def loglikelihood(p):
    try:
        p=np.copy(p)
        for i,(key,val) in enumerate( zip(paramNames,p) ):
            if key in discrete_params.keys():
                j = round(val)
                p[i] =  discrete_params[key][j]
        #HERA logL
        m = lambda z,karr,p=p: emulatorModel2d(z, karr, p) 
        logL_HERA, individual_loglikes = like_hera.loglike(m, return_individual_loglikes=True) 
        #T_emus = np.array([TS_TK_Trad_from_emulators(p,z=zi) for zi in redshifts]).flatten().tolist() #flatten nested tuples and add *extra
        
        #Chandra logL
        logL_Chandra = like_Chandra(p)

        #LWA1/ARCADE2 logL
        logL_LWA = like_LWA(p)

        logL = logL_HERA + logL_Chandra + logL_LWA 
        #print(logL_Chandra, logL) 
        return logL, [logL_LWA, logL_Chandra, *individual_loglikes]#, *T_emus]#[*T8, *T10, *extra]
    except Exception as e:
        print(e, flush=True)

# Polychord ingredients
_priorBounds = np.array([priorDict_HERA4[p] for p in paramNames])
prior = UniformPrior(_priorBounds.T[0], _priorBounds.T[1])

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)


IDR = [
    path+'data/idr4_screenshot_data/pspec_h4c_idr4_fields.npy',
    #path+'data/observations_HERA_IDR2/pspec_h1c_idr2_field{}.h5',
    #'data/idr4_screenshot_data/pspec_h6c_idr6_fields.npy'
    ]


for datapath in IDR:
    selections = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}} if "idr2" in datapath else None
    decimation_factor = 2 if "idr2" in datapath else None

    like_hera = likelihood(
        datapath=datapath,
        decimation_factor=decimation_factor,
        selections=selections
                    )
    name = like_hera.datapath.split("/")[-1].split("_")[2]

    bandsNfields = 0
    redshifts=[]
    for band in like_hera.data.keys():
        for field in like_hera.data[band].keys():
            redshifts.append(like_hera.data[band][field]["z"])
            bandsNfields = bandsNfields+1
    redshifts = np.unique(redshifts)
    nDims = len(paramNames)


    nDerived = bandsNfields+1+1#+3*len(redshifts) #2*9 + 3*9 # (selections, number of bands*fields, +6 temperature outputs) # idr4=(9bands*2fields+3temps*9bands), idr2=(2bands*1fields+3temps*9redshifts(AKA bands))
    settings = PolyChordSettings(nDims, nDerived)
    settings.nlive = 10000#00 #2000
    settings.base_dir = 'non-public/{0}_Chandra_LWA_nlive_{1}'.format(name,settings.nlive)
    settings.file_root = 'run_' + name
    settings.do_clustering = True
    settings.read_resume = True

    if True:
        #comm.barrier()
        output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)

        redshifts_str = [str(z) for z in redshifts]
        polychordnames = []
        for p in paramNames:
            polychordnames.append((p, texDict[p][1:-1]))
        #for z in redshifts_str:#["8", "10"]:
        #    for T in ["TS", "TK", "TR"]:
        #        polychordnames.append((T+z, r"T_"+T[1]+"\,(z="+z+")"))
        for i in range( nDerived ):#- 3*len(redshifts_str) ):
            polychordnames.append(("logL"+str(i), r"\log L"+str(i)))
        output.make_paramnames_files(polychordnames)

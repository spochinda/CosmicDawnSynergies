import os
path = os.path.dirname(os.getcwd()) #"/home/sp2053/rds/hpc-work/powerspectra_analysis/" #"/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

import numpy as np
from scipy.constants import parsec, physical_constants
from codes.emulator_poweremu import *
from codes.likelihood_hera import *

from margarine.maf import MAF
from scipy.io import loadmat
import scipy.special as ssp#
import codes.itamar.radio_cutoff_calc as rad
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior,LogUniformPrior
from globalemu.eval import evaluate
from tensorflow import keras
from itertools import product

import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#size = comm.Get_size()

if rank==0:
    print("Running script",flush=True)



"""
paramNames = [
            "log10fstarII", #0
            "log10fstarIII",#1
            "log10Vc",#2
            "log10fX",#3
            "alpha",#4
            "nu_0",#5
            #"zeta",
            "tau",#6
            "log10fradio",#7
            "pop",#8
            #"feed",
            #"delay",
            ]
"""
texDict = {"log10fstarII": r"$\log_{10} \left(f_{\rm star, II}\right)$",
        "log10fstarIII": r"$\log_{10} \left(f_{\rm star, III}\right)$",
        "log10Vc": r"$\log_{10}\left(V_c\right)$",
        "log10fX": r"$\log_{10}\left( f_{\rm X}\right)$",
        "alpha": r"$\alpha$",
        "nu_0": r"$\nu_{\rm 0}$",
        "tau": r"$\tau$",
        "log10fradio": r"$\log_{10} \left(f_{\rm r}\right)$",
        "pop": r"$\rm pop$",
        }

texDict_SARAS3 = {
    "a0": r"$a_{0}$",
    "a1": r"$a_{1}$",
    "a2": r"$a_{2}$",
    "a3": r"$a_{3}$",
    "a4": r"$a_{4}$",
    "a5": r"$a_{5}$",
    "a6": r"$a_{6}$",
    "std21": r"$\sigma_{21}$",
    }

priorDict = {
            "log10fstarII": np.log10([1e-3, 0.5]).tolist(),
            "log10fstarIII": np.log10([1e-3, 0.5]).tolist(),
            "log10Vc": np.log10([4.2, 100]).tolist(),
            "log10fX": np.log10([1e-3, 1e3]).tolist(),
            "alpha": [-0.5, 2.5],
            "nu_0": [-0.5, 16.5],#[100:100:1500, 2000, 3000],
            "tau": [0.054-3*0.007, 0.054+3*0.007],#??
            "log10fradio": np.log10([1e-1, 99990.]).tolist(),##############
            "pop": [-0.5, 2.5],#[231, 232, 233],#??
            }

priorDict_SARAS3 = {
    "a0": [-10,10],
    "a1": [-10,10],
    "a2": [-10,10],
    "a3": [-10,10],
    "a4": [-10,10],
    "a5": [-10,10],
    "a6": [-10,10],
    "std21": [1e-2, 0.5],
}



    
discrete_params = {
            "alpha": [1, 1.3, 1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            #"feed": [0, 1],
            #"delay": [0, 0.75]
}

# Emulators
P = poweremu(loadfile=path+"data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
XRB_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/CXBlog10_emu_n400_l50505050_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)
SFR_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/SFR1_emu_n400_l80808080_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)







#Likelihoods

def TS_TK_Trad_from_emulators(p, z=8):
    par0 = np.array([z, *p])
    TS = TS_emu.predict(par0)
    TK = TK_emu.predict(par0)
    TR = TR_emu.predict(par0)
    return TS, TK, TR

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

from codes.likelihood import *
#########Chandra#########
"""
eV_toHz = physical_constants['electron volt-hertz relationship'][0]
keV_toHz = eV_toHz*1e3
sr_todeg2 = (180/np.pi)**2
Mpc_tom = 1e6 * parsec
Mpc_tocm = Mpc_tom * 1e2
cm_toMpc = 1/Mpc_tocm

nu_keV = loadmat(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_nu_mat.mat")["nu_keV"][0]
nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.1)#55)
nu_keV = nu_keV[nu_mask]

X_limits = np.array([ #nu_min, nu_max, mean, std
    #[0.5, 2, 8.15*1e-12, 0.58*1e-12], #Lehmer+2012 
    [1, 2, 1.04*1e-12, 0.14*1e-12], #Hickox & Markevitch (2006)
    [2, 8, 3.4*1e-12, 1.7*1e-12], #Hickox & Markevitch (2006)
    [8, 24, 6.773*1e-8/sr_todeg2, 0.348*1e-8/sr_todeg2], #Harrison et al. (2016)
    [20, 50, 6.205*1e-8/sr_todeg2, 0.17*1e-8/sr_todeg2], #Harrison et al. (2016)
])
numin_index = np.array([np.where(nu_keV > numin_obs_keV)[0][0] for numin_obs_keV in X_limits[:,0]], dtype=int)
numax_index = np.array([np.where(nu_keV < numax_obs_keV)[0][-1] for numax_obs_keV in X_limits[:,1]], dtype=int)

deltanu_obs = np.array([
    nu_keV[indEmin_obs + 1 : indEmax_obs + 1] - nu_keV[indEmin_obs : indEmax_obs] for indEmin_obs,indEmax_obs in zip(numin_index,numax_index)# X_limits[:,0:2].astype(int)
    ], dtype=object)

def like_Chandra(p):
    if not use_MAFs:
        logL = 0
        XRB_pred0 = emulatorModel1d(emu=XRB_emu,arr=np.log10(nu_keV), p=p)
        sum_XRB_pred =np.array([
            np.sum(XRB_pred0[indEmin_obs:indEmax_obs] * deltanuobs * keV_toHz * cm_toMpc**2 / sr_todeg2) for indEmin_obs,indEmax_obs,deltanuobs in zip(numin_index, numax_index, deltanu_obs)
            ])
        P = 0.5 * (1 + ssp.erf( (X_limits[:,2] - sum_XRB_pred) / np.sqrt(2) / np.sqrt(X_limits[:,3]**2+(sum_XRB_pred*0.05)**2) ))
        if 0 in P:
            logL=-np.inf
        else:    
            logL = np.log(P).sum()
    else:
        sum_XRB_pred = np.zeros(len(X_limits))
        logL = Chandra_maf.log_like(p, -0.09300543633401404) #insert real evidence
    return logL, sum_XRB_pred
"""

#########LWA#########
"""[nu_obs, T_obs, dT_obs] = np.load(path+'codes/itamar/LWA1_with_err.npy')
def like_LWA(p, z_cutoff=6.01):
    if not use_MAFs:
        fr = 10**p[7] #change to params
        z_dense = np.linspace(z_cutoff-0.01, z_cutoff+0.01,2)
        sfr_dense = 10**(np.interp(z_dense, [6,7,8], np.log10( 
            emulatorModel1d(emu=SFR_emu, arr=[6,7,8], p=p)
        ) )) 
        nu_today, T_today = rad.get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
        T_model = np.mean(T_today, axis=0) * fr
        T_model_interp = np.interp(nu_obs, nu_today.value, T_model)
        dT_model_interp = T_model_interp*0.05

        P = 0.5 * (1 + ssp.erf( (T_obs - T_model_interp) / np.sqrt(2) / np.sqrt(dT_obs**2+dT_model_interp**2))) 
        if 0 in P:
            logL=-np.inf
        else:    
            logL = np.log(P).sum()
    else:
        T_model_interp = np.zeros(len(T_obs))
        logL = LWA_maf.log_like(p, -0.05650523374863248) #insert real evidence
    return logL, T_model_interp"""




#########SARAS3#########
"""
#model = keras.models.load_model(path+'data/globalemu/emulator{0}/results/model.h5'.format(7+i), compile=False)
#predictor = evaluate(base_dir=path+'data/globalemu/emulator{0}/results/'.format(7+i), model=model, z=z, gc=False, logs=[]) #0,1,2,3,7], ) 
freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt(path+"data/SARAS3/SARAS_3_averaged_spectrum.txt").T
log_freq = np.log10(freq)
reduced_freq = 2*((log_freq - log_freq.min())/ \
    (log_freq.max()-log_freq.min())) - 1
z = 1420/freq-1
model = keras.models.load_model(path+'data/globalemu/emulator14/results/model.h5', compile=False)
predictor = evaluate(base_dir=path+'data/globalemu/emulator14/results/', model=model, z=z, gc=False, logs=[]) #0,1,2,3,7], )
def foreground(p, freq):
    log_fit = 10**(np.sum([
                    p[i] * reduced_freq**i
                    for i in range(len(p))],
                    axis=0))
    return log_fit
def like_SARAS(p):
    if not use_MAFs:
        fg = foreground(p[9:-1], freq)
        noise = p[-1]
        
        #signal = emulatorModel1d(emu=T21_emu, arr=z, p=p[:9])/1000 #globalemu_emulator(p[7:-1])
        signal, redshifts = predictor(p[:9]) 
        signal = signal/1000. #to K
        #print("Stack: ", np.stack((z,redshifts, z==redshifts),axis=1))
        logL = (
            -0.5*np.log(2*np.pi*(noise**2+(0.25*signal)**2)) 
            - 0.5 * (T_SARAS - fg - signal)**2
            /(noise**2+(0.25*signal)**2)
            ).sum()
    else:
        logL = saras_maf.log_like(p, -158.8493155641061 )
    return logL
"""

#########HERA#########
#datapath = path+'data/observations_H1C_IDR2/pspec_h1c_idr2_field{}.h5'
    #[
    #path+'data/observations_H4C_IDR2/pspec2cuthalf_h4c_idr4_fields.npy',#pspec2half_h4c_idr4_fields.npy'
    #path+'data/observations_H4C_IDR2/pspec_h4c_idr4_fields.npy',
    #path+'data/observations_H1C_IDR2/pspec_h1c_idr2_field{}.h5',
    #'data/observations_H4C_IDR2/pspec_h6c_idr6_fields.npy'
    #]
#selection_idr2 = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}
#selections = selection_idr2 if "h1c" in datapath else None

#selections = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}
#decimation_factor = 2 #if "h1c" in datapath else None

#like_hera = likelihood(
#    datapath=datapath,
#    decimation_factor=decimation_factor,
#    selections=selections
#    )




output_names_HERA = {"logL_HERA": r"\log L_\mathrm{HERA}", 
                        #"logL_HERA_B1": r"\log L_\mathrm{HERA,1}", "logL_HERA_B2": r"\log L_\mathrm{HERA,2}", 
                        }

output_names_Chandra = {"logL_Chandra": r"\log L_\mathrm{Chandra,tot}", 
                        #"S_{1-2}": r"S(1-2 keV)", "S_{2-8}": r"S(2-8 keV)", "S_{8-24}": r"S(8-24 keV)", "S_{20-50}": r"S(20-50 keV)"
                        }

output_names_LWA = {"logL_LWA": r"\log L_\mathrm{LWA}"}

output_names_SARAS3 = {"logL_SARAS": r"\log L_\mathrm{SARAS3}"}

output_names_xHI = {"logL_xHI": r"\log L_\mathrm{x_{HI}}"}

#nu_obs, T_obs, dT_obs = np.load('codes/itamar/LWA1_with_err.npy')
#for nu in nu_obs: 
#    output_names_LWA["T_Radio_z0_{0:.3f}".format(nu/1e9)] =  r"T_Radio_{0:.3f} (z=0)".format(nu/1e9)


    

selection = [{"1": {"D": {"kstart":0.36}}}, {"2": {"C": {"kstart":0.34}}}]

dpath = [
    'data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
    'data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5']

like_hera = likelihood(
    datapath=dpath,
    decimation_factor=2,
    selections=selection,
    return_individual_loglikes=False,
    output_names = output_names_HERA
)

#m = lambda z,karr,p: emulatorModel2d(z, karr, p[:9]) #need to change to pspec_likelihood for IDR4



#Setup sampling

class UniformDiscretePrior:
    def __init__(self, a):
        self.a = a
        self.b = len(a)
        #self.b = b

    def __call__(self, x):
        return self.a[ np.floor(self.b * x).astype(int) ] #+ (self.b-self.a) * x




use_MAFs = False
hera_maf = MAF.load("data/margarine/HERA_MAF.pkl")
Chandra_maf = MAF.load("data/margarine/Chandra_MAF.pkl")
LWA_maf = MAF.load("data/margarine/LWA_MAF.pkl")
saras_maf = MAF.load("data/margarine/SARAS_MAF.pkl") 

mafs = [hera_maf, saras_maf]
theta_lims = np.stack(
    [np.max([maf.theta_min for maf in mafs],axis=0), 
    np.min([maf.theta_max for maf in mafs],axis=0)],
    axis=1)
#maf_param_indices = [0,1,2,3,6,7]






def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)



#constraints = np.array(list(product([0,1], repeat=4))) #repeats=number of constraints
#constraints = np.unique(constraints, axis=0)
#constraints = np.array([set for set in constraints.tolist() if set!=[0,0,0,0]]) #remove 0 constraints



LikelihoodModules = np.array([like_hera, 
                        LikelihoodXRB(use_MAFs=use_MAFs, output_names = output_names_Chandra), 
                        LikelihoodRadioBackground(use_MAFs=use_MAFs, output_names = output_names_LWA), 
                        LikelihoodSARAS3(use_MAFs=use_MAFs, output_names = output_names_SARAS3),
                        LikelihoodNeutralFraction(use_MAFs=use_MAFs, output_names = output_names_xHI)])

#constraints = np.array([(1,1,1,1), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]).astype(bool) #HERA, Chandra, LWA, SARAS
constraints = np.array([(1,1,1,1,0)]).astype(bool) #HERA, Chandra, LWA, SARAS, xHI
nlives = [2]

for i,(nlive,(HERA,Chandra,LWA,SARAS,xHI)) in enumerate(zip(nlives,constraints)):
    priorDict_ = priorDict.copy() if not SARAS else {**priorDict.copy(), **priorDict_SARAS3.copy()}
    texDict_ = texDict.copy() if not SARAS else {**texDict.copy(), **texDict_SARAS3.copy()}

    if use_MAFs:
        pop_params = list(discrete_params.keys()) if not SARAS else [*list(discrete_params.keys()), *list(priorDict_SARAS3.keys())]
        [priorDict_.pop(p) for p in pop_params]
        [texDict_.pop(p) for p in pop_params]
    
    paramNames = list(priorDict_.keys())



    def prior(cube):
        theta = np.zeros_like(cube)
        if not use_MAFs:
            for i,(name,bounds) in enumerate(priorDict_.items()):
                if name in discrete_params.keys():
                    theta[i] = UniformDiscretePrior(discrete_params[name])(cube[i])
                else:
                    if name!="std21":
                        theta[i] = UniformPrior(*bounds)(cube[i])    
                    else:
                        theta[i] = LogUniformPrior(*bounds)(cube[i])
        else:
            for i,t in enumerate(theta_lims):
                theta[i] = UniformPrior(*t)(cube[i]) #edit bounds with maf.theta_min, maf.theta_max
        return theta


    def loglikelihood(p, include_HERA=HERA, include_Chandra=Chandra,  include_LWA=LWA, include_SARAS=SARAS):
        #try:
        #HERA logL p[:9] filters out foreground parameters
        if False:
            m = lambda z,karr,p=p[:9]: emulatorModel2d(P, z, karr, p[:9]) #need to change to pspec_likelihood for IDR4
            if not use_MAFs:
                logL_HERA, individual_loglikes = like_hera.loglike(m, return_individual_loglikes=return_individual_loglikes) if include_HERA else (0,0) #need to change to pspec_likelihood for IDR4
                logL_Chandra, sum_XRB_pred = LikelihoodXRB().computeLikelihood(p[:9]) if include_Chandra else (0.,0.) #logL_Chandra, sum_XRB_pred = like_Chandra(p[:9]) if include_Chandra else (0.,0.)
                logL_LWA, T_model_interp = LikelihoodRadioBackground().computeLikelihood(p[:9]) if include_LWA else (0.,0.) #like_LWA(p[:9]) if include_LWA else (0.,0.)
                logL_SARAS = LikelihoodSARAS3().computeLikelihood(p) if include_SARAS else 0. #like_SARAS(p) if include_SARAS else 0.
            else:
                logL_HERA, individual_loglikes = (hera_maf.log_like(p, -16.778600948827645 ),0) if True else (0,0) #need to change to pspec_likelihood for IDR4
                logL_Chandra, sum_XRB_pred = LikelihoodXRB(use_MAFs=use_MAFs).computeLikelihood(p[:9]) if include_Chandra else (0.,0.) #logL_Chandra, sum_XRB_pred = like_Chandra(p[:9]) if include_Chandra else (0.,0.)
                logL_LWA, T_model_interp = LikelihoodRadioBackground(use_MAFs=use_MAFs).computeLikelihood(p[:9]) if include_LWA else (0.,0.) #like_LWA(p[:9]) if include_LWA else (0.,0.)
                logL_SARAS = LikelihoodSARAS3(use_MAFs=use_MAFs).computeLikelihood(p) if include_SARAS else 0. #like_SARAS(p) if include_SARAS else 0.

        #m = lambda z,karr,p=p: emulatorModel2d(z, karr, p[:9]) #need to change to pspec_likelihood for IDR4
        #logL_HERA, individual_loglikes = like_hera.loglike(p) if include_HERA else (0,0) #need to change to pspec_likelihood for IDR4

        logL_nDerived = [likelihood.computeLikelihood(p) for likelihood in LikelihoodModules[constraints[i]]]
        logL = np.sum([item[0] if isinstance(item,list) else item for item in logL_nDerived])

        
        #if True:
        logL_nDerived_flattened = np.array([])
        for item in logL_nDerived:
            if isinstance(item,list):
                for sub_item in item:
                    logL_nDerived_flattened = np.append(logL_nDerived_flattened, sub_item)
            else:
                logL_nDerived_flattened = np.append(logL_nDerived_flattened, item)

            #nDerived3 = np.array(nDerived3).flatten()

        #print(logL,"\nnderived2:\n",logL_nDerived,"\nnderived3:\n",logL_nDerived_flattened, flush=True)
        
        #nDerived_2 = [item[0],*item[1] if item[1] is not None else item[0] for item in nDerived_2]

        #print("logLs: HERA={0:.2f}, SARAS={1:.2f}".format(logL_HERA, logL_SARAS))
        
        #s_derived = time.time()
        
        #logL = logL_HERA + logL_SARAS + logL_Chandra + logL_LWA + logL_SARAS
        
    
        #nderived_params = np.array([])
        #nderived_params = np.append(nderived_params, individual_loglikes) if include_HERA else nderived_params
        #nderived_params = np.append(nderived_params, [logL_Chandra, *sum_XRB_pred]) if include_Chandra else nderived_params
        #nderived_params = np.append(nderived_params, [logL_LWA, *T_model_interp]) if include_LWA else nderived_params
        #nderived_params = np.append(nderived_params, [logL_SARAS,]) if include_SARAS else nderived_params
    
        #e_derived = time.time() - s_derived
        #print("Time spent on each likelihood calculation:\n{0} HERA\n{1} Chandra\n{2} LWA\n{3} SARAS\n{4} derived".format(e_HERA,e_Chandra,e_LWA,e_SARAS,e_derived))
        return logL, logL_nDerived_flattened#, *T_emus]
        #except Exception as e:
        #    exc_type, exc_obj, exc_tb = sys.exc_info()
        #    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #   print(exc_type, fname, exc_tb.tb_lineno)
        #   print(e, flush=True)


    

    #name = "idr3"#"_".join( like_hera.datapath.split("/")[-1].split("_")[1:3] ) if HERA else "no_idr"

    bandsNfields = 0
    redshifts=[]
    for band in like_hera.data.keys():
        for field in like_hera.data[band].keys():
            redshifts.append(like_hera.data[band][field]["z"])
            bandsNfields = bandsNfields+1
    temp_redshifts, index = np.unique(redshifts, return_index=True)
    index.sort()
    redshifts = np.array(redshifts)[index] #preserve redshift order
    nDims = len(paramNames) #if not use_MAFs else len(np.array(paramNames)[maf_param_indices])
    #print("nderived: ", [d.nDerived for d in LikelihoodModules])
    #print("3: ", (bandsNfields*HERA-1)*0 , LikelihoodXRB().nDerived , LikelihoodRadioBackground().nDerived , LikelihoodSARAS3().nDerived)
    nDerived = np.sum([likelihood.nDerived for likelihood in LikelihoodModules[constraints[i]]]) #2 #(bandsNfields*HERA-1)*0 + LikelihoodXRB(use_MAFs=use_MAFs).nDerived + LikelihoodRadioBackground(use_MAFs=use_MAFs).nDerived + LikelihoodSARAS3(use_MAFs=use_MAFs).nDerived #if not use_MAFs else 2 #+3*len(redshifts) #2*9 + 3*9 # (selections, number of bands*fields, +6 temperature outputs) # idr4=(9bands*2fields+3temps*9bands), idr2=(2bands*1fields+3temps*9redshifts(AKA bands)) #2
    settings = PolyChordSettings(nDims, nDerived)
    settings.nlive = nlive #00 #2000
    settings.base_dir = path+'scripts/non-public/{0}HERA_{1}Chandra_{2}LWA_{3}SARAS_{4}xHI_globalemu315emu14testmainbranch_nlive_{5}'.format(*constraints[i].astype(int),settings.nlive)
    settings.file_root = 'run'
    settings.do_clustering = True
    settings.read_resume = False    
    
    if rank==0:
        start_time = time.time()
        print("Redshifts: ", redshifts, flush=True) 
        print("Constraints, nDerived:", HERA,Chandra,LWA,SARAS, nDerived, flush=True) 
        print("Starting sampling. Base dir: {0}".format(settings.base_dir), flush=True)
    #pp = [np.log10(0.4), np.log10(0.4), np.log10(10), 
    #    np.log10(20), 1., 500, 0.05, np.log10(1000), 232,
    #    *np.random.uniform(low=-10, high=10, size=6),
    #    *np.random.uniform(low=-1, high=10**0.5, size=1)]
    #print(loglikelihood(pp)) #test
    
    
    #print(LikelihoodModules[1].X_limits[:,2:]/1e-11 )
    
    if True:
        output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
        #output.logZ for evidence
        redshifts_str = [str(z) for z in redshifts]
        polychordnames = []
        for name,texStr in texDict_.items():
            polychordnames.append((name, texStr[1:-1]))
        for likelihood in LikelihoodModules[constraints[i]]:
            for item in likelihood.output_names.items():
                polychordnames.append(item)

        #for z in redshifts_str:#["8", "10"]:
        #    for T in ["TS", "TK", "TR"]:
        #        polychordnames.append((T+z, r"T_"+T[1]+"\,(z="+z+")"))
        #for i in range( nDerived ):#- 3*len(redshifts_str) ):
        #    polychordnames.append(("logL"+str(i), r"\log L"+str(i)))
        if False:
            if not use_MAFs:
                if "h4c" in name:            
                    if HERA:
                        for band in redshifts:
                            for field in range(2):
                                polychordnames.append(("logL_{0:.2f}_{1}".format(band,field), r"\log L_{0:.2f}_{1}".format(band,field)))
                elif "h1c" in name:
                    for i in range( 2 ):#- 3*len(redshifts_str) ):
                        polychordnames.append(("logL"+str(i), r"\log L"+str(i)))        
                if Chandra:
                    polychordnames.append(("logL_Chandra", r"\log L_Chandra"))
                    for numin,numax in X_limits[:,:2]:
                        polychordnames.append(("S_{0}_{1}".format(numin,numax), r"S({0}-{1} keV)".format(numin,numax)))
                if LWA:
                    polychordnames.append(("logL_LWA", r"\log L_LWA"))
                    for nu in nu_obs:
                        polychordnames.append(("T_Radio_z0_{0:.3f}".format(nu/1e9), r"T_Radio_{0:.3f} (z=0))".format(nu/1e9)))
                if SARAS:
                    polychordnames.append(("logL_SARAS", r"\log L_\mathrm{SARAS3}"))
            else:
                if HERA:
                    polychordnames.append(("logL_HERA", r"\log L_\mathrm{HERA,tot}"))
                if Chandra:
                    polychordnames.append(("logL_Chandra", r"\log L_\mathrm{Chandra,tot}"))
                if LWA:
                    polychordnames.append(("logL_LWA", r"\log L_\mathrm{LWA}"))
                if SARAS:
                    polychordnames.append(("logL_SARAS", r"\log L_\mathrm{SARAS3}"))

        output.make_paramnames_files(polychordnames)
        if rank==0:
            print("{0} took {1:.2f} hours ---".format( settings.base_dir.split("/")[-1], (time.time() - start_time) / 3600))

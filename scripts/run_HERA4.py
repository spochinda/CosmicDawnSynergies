path = "/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"
import numpy as np
import matplotlib.pyplot as plt #
from scipy.constants import parsec, physical_constants
from codes.emulator_poweremu import *
from codes.likelihood_hera import *
#from codes.tools import load_files
from scipy.io import loadmat
import scipy.special as ssp
import codes.itamar.radio_cutoff_calc as rad
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
import time
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
rank = 0#comm.Get_rank()
#size = comm.Get_size()
#path = "/home/sp2053/rds/hpc-work/powerspectra_analysis/"
if rank==0:
    print("Running script",flush=True)

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
             "log10fX": np.log10([1e-3, 1e3]).tolist(),
             "alpha": [-0.5, 2.5],#[-1, -1.3, -1.5],
             "nu_0": [-0.5, 16.5],#[100:100:1500, 2000, 3000],
             #"zeta": [4.3641, 3369.6118],
             #"tau": [0.02351, 0.10198],#??
             "tau": [0.054-3*0.007, 0.054+3*0.007],#??
             "log10fradio": np.log10([1e-1, 99990.]).tolist(),##############
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


# Used for initial runs and "final1"
#P = poweremu(loadfile="data/trained_emulators_poweremu/Sims_data_v03_150it_23.02.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
P = poweremu(loadfile=path+"data/trained_emulators_poweremu/Deltasq_emu_n400_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
XRB_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/XRB_emu_PL9_n500_l100100100100_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)
SFR_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/SFR_emu_PL9_n500_l100100100100_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)
TS_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_Ts_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_TK_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile=path+"data/trained_emulators_poweremu/Pk_Trad_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
#TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/Ts_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)
#TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)
#TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/Trad_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)

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
"""XRB_limits = {
    '0.5 - 2 keV': {
        'mean': 8.15*1e-12,
        'std': 0.58*1e-12,
        }, #Hickox & Markevitch (2006)
    '1 - 2 keV': {
        'mean': 1.04*1e-12,
        'std': 0.14*1e-12,
        }, #Hickox & Markevitch (2006)
    '2 - 8 keV': {
        'mean': 3.4*1e-12,
        'std': 1.7*1e-12,
        }, #Hickox & Markevitch (2006)
    '8 - 24 keV': {
        'mean': 6.773*1e-8/sr_todeg2,
        'std': 0.348*1e-8/sr_todeg2,
        }, #Harrison et al. (2016)
    '20 - 50 keV': {
        'mean': 6.205*1e-8/sr_todeg2,
        'std': 0.17*1e-8/sr_todeg2,
        }, #Harrison et al. (2016)
}"""

nu_keV = loadmat(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_nu_mat.mat")["nu_keV"][0]
nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.1)#55)
nu_keV = nu_keV[nu_mask]

X_limits = np.array([ #nu_min, nu_max, mean, std
    #[0.5, 2, 8.15*1e-12, 0.58*1e-12], #Lehmer+2012 
    [1, 2, 1.04*1e-12, 0.14*1e-12], #Hickox & Markevitch (2006)
    [2, 8, 3.4*1e-12, 1.7*1e-12], #Hickox & Markevitch (2006)
    [8, 24, 6.013*1e-8/sr_todeg2, 0.145*1e-8/sr_todeg2], #Harrison et al. (2016)
    [20, 50, 6.56*1e-8/sr_todeg2, 0.273*1e-8/sr_todeg2], #Harrison et al. (2016)
])
print("print statement: ", X_limits[2:,:])
X_limits[:,0] = [np.where(nu_keV > numin_obs_keV)[0][0] for numin_obs_keV in X_limits[:,0]]
X_limits[:,1] = [np.where(nu_keV < numax_obs_keV)[0][-1] for numax_obs_keV in X_limits[:,1]]
deltanu_obs_2 = np.array([
    nu_keV[indEmin_obs + 1 : indEmax_obs + 1] - nu_keV[indEmin_obs : indEmax_obs] for indEmin_obs,indEmax_obs in X_limits[:,0:2].astype(int)
    ])



def like_Chandra(p):
    logL = 0
    XRB_pred0 = emulatorModel1d(emu=XRB_emu,arr=np.log(nu_keV), p=p)
    sum_XRB_pred =np.array([
        np.sum(XRB_pred0[indEmin_obs:indEmax_obs] * deltanu_obs_2 * keV_toHz * cm_toMpc**2 / sr_todeg2) for (indEmin_obs,indEmax_obs),deltanu_obs_2 in zip(X_limits[:,0:2].astype(int), deltanu_obs_2)
        ])
    P = 0.5 * (1 + ssp.erf( (X_limits[:,2] - sum_XRB_pred) / np.sqrt(2) / np.sqrt(X_limits[:,3]**2+(sum_XRB_pred*0.05)**2) ))
    if 0 in P:
        logL=-np.inf
    else:    
        logL = np.log(P).sum()
    if False:
        for i,key in enumerate(XRB_limits.keys()):
            numin_obs_keV, numax_obs_keV = np.array(key[:-4].split(" - "),dtype=float) #[:-4] removes the " keV"
            indEmin_obs = np.where(nu_keV > numin_obs_keV)[0][0]
            indEmax_obs = np.where(nu_keV < numax_obs_keV)[0][-1]
            deltanu_obs = nu_keV[indEmin_obs + 1 : indEmax_obs + 1] - nu_keV[indEmin_obs : indEmax_obs]
            #print("str:",deltanu_obs, "str1:",deltanu_obs_2[i])
            XRB_pred = emulatorModel1d(emu=XRB_emu,arr=np.log(nu_keV), p=p)
            sum_XRB_pred = np.sum(XRB_pred[indEmin_obs:indEmax_obs] * deltanu_obs * keV_toHz * cm_toMpc**2 / sr_todeg2) #(erg / s / Hz / cm^2 /sr) * (keV) * (Hz/keV) / (deg^2/sr)= erg / s /cm^2 / deg^2
            
            #sum_XRB_pred_std = sum_XRB_pred*0.1 #np.sum((0.1*XRB_pred_2)**2 * deltanu_obs * keV_toHz)**0.5
            #logL += -0.5 * ((XRB_limits[key]["mean"] - sum_XRB_pred)**2 / (XRB_limits[key]["std"]**2 + sum_XRB_pred_std**2) )
            P = 0.5 * (1 + ssp.erf( (XRB_limits[key]["mean"] - sum_XRB_pred) / np.sqrt(2) / np.sqrt(XRB_limits[key]["std"]**2+(sum_XRB_pred*0.05)**2)))
            print("P2:", P)
            if P==0:
                logL += -np.inf
            else:
                logL += np.log(P)
            #print("sum_XRB_pred={1}, \nXRB_limits={2}, \nratio={3:.2f}, \nlnl={0:.2f}".format(logL,sum_XRB_pred, XRB_limits[key]["mean"],sum_XRB_pred/XRB_limits[key]["mean"]))
    return logL, sum_XRB_pred

[nu_obs, T_obs, dT_obs] = np.load(path+'codes/itamar/LWA1_with_err.npy')
def like_LWA(p, z_cutoff=7):
    fr = 10**p[7] #change to params
    z_dense = np.linspace(z_cutoff-0.01, z_cutoff+0.01,2)
    sfr_dense = 10**(np.interp(z_dense, [6,7,8], np.log10( 
        emulatorModel1d(emu=SFR_emu, arr=[6,7,8], p=p)
    ) )) 

    nu_today, T_today = rad.get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
    T_model = np.mean(T_today, axis=0) * fr
    
    T_model_interp = np.interp(nu_obs, nu_today.value, T_model)
    dT_model_interp = T_model_interp*0.05
    print(T_model_interp)
    plt.loglog(nu_obs, T_obs)
    plt.loglog(nu_obs, T_model_interp)
    plt.show()
    P = 0.5 * (1 + ssp.erf( (T_obs - T_model_interp) / np.sqrt(2) / np.sqrt(dT_obs**2+dT_model_interp**2))) 
    if 0 in P:
        logL=-np.inf
    else:    
        logL = np.log(P).sum()
    
    return logL, T_model_interp



# Polychord ingredients
_priorBounds = np.array([priorDict_HERA4[p] for p in paramNames])
prior = UniformPrior(_priorBounds.T[0], _priorBounds.T[1])

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)


IDR = [
    path+'data/idr4_screenshot_data/pspec2half_h4c_idr4_fields.npy',
    #path+'data/observations_HERA_IDR2/pspec_h1c_idr2_field{}.h5',
    #'data/idr4_screenshot_data/pspec_h6c_idr6_fields.npy'
    ]
selection_idr2 = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}
selection_idr4 = {}
idr4 = np.load(path+'data/idr4_screenshot_data/pspec2half_h4c_idr4_fields.npy',allow_pickle=True).item()
for band in idr4.keys():
    selection_idr4[band] = {}
    for field in ["0","1"]:
        min_index = np.argmin(idr4[band][field]["dsq"])
        selection_idr4[band][field] = {"kstart": idr4[band][field]["k_data"][min_index]}

from itertools import product
constraints = np.array(list(product([0,1], repeat=3))) #repeats=number of constraints
constraints[:,0] = 1 #fix hera constraint
constraints[:,1] = 1 #fix Chandra constraint
constraints[:,2] = 1 #fix LWA constraint
constraints = np.unique(constraints, axis=0)
constraints = np.array([set for set in constraints.tolist() if set!=[0,0,0]]) #remove 0 constraints

for datapath in IDR:
    
    
    for HERA,Chandra,LWA in constraints:
        def loglikelihood(p, return_individual_loglikes=True, include_HERA=HERA, include_Chandra=Chandra,  include_LWA=LWA):
            try:
                p=np.copy(p)
                for i,(key,val) in enumerate( zip(paramNames,p) ):
                    if key in discrete_params.keys():
                        j = round(val)
                        p[i] =  discrete_params[key][j]
                #HERA logL
                m = lambda z,karr,p=p: emulatorModel2d(z, karr, p) #need to change to pspec_likelihood for IDR4
                
                logL_HERA, individual_loglikes = like_hera.loglike(m, return_individual_loglikes=return_individual_loglikes) if include_HERA else (0,0) #need to change to pspec_likelihood for IDR4
                #T_emus = np.array([TS_TK_Trad_from_emulators(p,z=zi) for zi in redshifts]).flatten().tolist() #flatten nested tuples and add *extra
                
                #Chandra logL
                logL_Chandra, sum_XRB_pred = like_Chandra(p) if include_Chandra else (0.,0)

                #LWA1/ARCADE2 logL
                logL_LWA, T_model_interp = like_LWA(p) if include_LWA else (0., 0.)
                
                logL = logL_HERA + logL_Chandra + logL_LWA 
                
                nderived_params = np.array([])
                nderived_params = np.append(nderived_params, individual_loglikes) if include_HERA else nderived_params
                nderived_params = np.append(nderived_params, [logL_Chandra, *sum_XRB_pred]) if include_Chandra else nderived_params
                nderived_params = np.append(nderived_params, [logL_LWA, *T_model_interp]) if include_LWA else nderived_params
                print("nderived; ",nderived_params)
                return logL, nderived_params#, logL_LWA, logL_Chandra]#, *T_emus]#[*T8, *T10, *extra]
            except Exception as e:
                print(e, flush=True)




        selections = selection_idr2 if "idr2" in datapath else selection_idr4
        decimation_factor = 2 if "idr2" in datapath else 2 #None

        like_hera = likelihood(
            datapath=datapath,
            decimation_factor=decimation_factor,
            selections=selections
                        )
        name = like_hera.datapath.split("/")[-1].split("_")[2] if HERA else "no_idr"
        if rank==0:
            print("IDR: ", name, flush=True)
            print("Datapath: ", like_hera.datapath.split("/")[-1], flush=True)

        bandsNfields = 0
        redshifts=[]
        for band in like_hera.data.keys():
            for field in like_hera.data[band].keys():
                redshifts.append(like_hera.data[band][field]["z"])
                bandsNfields = bandsNfields+1
        redshifts = np.unique(redshifts)
        nDims = len(paramNames)


        nDerived = bandsNfields*HERA+Chandra+Chandra*len(X_limits)+LWA+LWA*len(T_obs) #+3*len(redshifts) #2*9 + 3*9 # (selections, number of bands*fields, +6 temperature outputs) # idr4=(9bands*2fields+3temps*9bands), idr2=(2bands*1fields+3temps*9redshifts(AKA bands))
        settings = PolyChordSettings(nDims, nDerived)
        settings.nlive = 5#00 #2000
        settings.base_dir = path+'scripts/non-public/{0}_{1}Chandra_{2}LWA_nlive_{3}'.format(name,Chandra,LWA,settings.nlive)
        settings.file_root = 'run_' + name
        settings.do_clustering = True
        settings.read_resume = True

        pp = [np.log10(0.4), np.log10(0.4), np.log10(10), np.log10(20), 2.3, 9.4, 0.05, np.log10(1000), 1.31,] #-0.11, 1.25] #test
        print("Constraints, nDerived:", HERA,Chandra,LWA, nDerived) #test
        print(loglikelihood(pp)) #test

        if rank==0:
            print("Starting sampling. Base dir: {0}".format(settings.base_dir), flush=True)
        if False:
            #comm.barrier()
            time.sleep(5)
            attempts = 0
            #while attempts < 3:
            #    try:
            output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
            #    except Exception as e:
            #        print("Error: ", e, flush=True)
            #        attempts += 1

            redshifts_str = [str(z) for z in redshifts]
            polychordnames = []
            for p in paramNames:
                polychordnames.append((p, texDict[p][1:-1]))
            #for z in redshifts_str:#["8", "10"]:
            #    for T in ["TS", "TK", "TR"]:
            #        polychordnames.append((T+z, r"T_"+T[1]+"\,(z="+z+")"))
            #for i in range( nDerived ):#- 3*len(redshifts_str) ):
            #    polychordnames.append(("logL"+str(i), r"\log L"+str(i)))
            X_limits = np.array([ #nu_min, nu_max, mean, std
                [0.5, 2], #Lehmer+2012 
                [1, 2], #Hickox & Markevitch (2006)
                [2, 8], #Hickox & Markevitch (2006)
                [8, 24], #Harrison et al. (2016)
                [20, 50], #Harrison et al. (2016)
            ])

            polychordnames = []
            for band in redshifts:
                for field in range(2):
                    #print("band={0:.2f},field={1}".format(band,field))
                    polychordnames.append(("logL_{0:.2f}_{1}".format(band,field), r"\log L_{0:.2f}_{1}".format(band,field)))

            polychordnames.append(("logL_Chandra", r"\log L_Chandra"))
            for numin,numax in X_limits:
                polychordnames.append(("S_{0}_{1}".format(numin,numax), r"S({0}-{1} keV)".format(numin,numax)))

            polychordnames.append(("logL_LWA", r"\log L_LWA"))
            for nu in nu_obs:
                polychordnames.append(("T_Radio_z0_{0:.3f}".format(nu/1e9), r"T_Radio_{0:.3f} (z=0))".format(nu/1e9)))
            output.make_paramnames_files(polychordnames)




"""
import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,3,figsize=(12,8),sharey="row",sharex="col")
c=["deeppink","royalblue"]
m=["o", "s"]
for i,band in enumerate(list(idr4.keys())[::-1]):
    row,col = np.unravel_index(i, shape=ax.shape)
    for j,field in enumerate(idr4[band].keys()):
        #ax[row,col].plot(idr4[band][field]["k_data"], idr4[band][field]["dsq"], marker="o",c="k", alpha=1,label="SC IDR4 Field {0}".format(field))
        ax[row,col].set_yscale("log")
    ax[row,col].text(0.95,0.95,r"$z={:.2f}$".format(idr4[band][field]["z"]), transform=ax[row,col].transAxes, ha="right", va="top", fontsize=12)
    ax[row,col].grid()
    for j in range(2):
        
        markers, caps, bars = ax[row,col].errorbar(like_hera.data[band][str(j)]["k_data"], like_hera.data[band][str(j)]["dsq"], like_hera.data[band][str(j)]["std"], 
                             c=c[j], marker=m[j], ls="None",alpha=1, capsize=2,
                             label="IDR4 Field {0}".format(j))
        [bar.set_alpha(0.5) for bar in bars]

        markers, caps, bars = ax[row,col].errorbar(idr4[band][str(j)]["k_data"], idr4[band][str(j)]["dsq"], idr4[band][str(j)]["std"], 
                             c="k", marker=m[j], ms=8, ls="None",alpha=0.4, capsize=2,
                             label="IDR4 Field {0}".format(j))
    ax[row,col].grid()
    ax[row,col].set_ylim(1e0, 1e10)
    ax[row,col].set_xlim(0, 1.5)
    if i==0:
        ax[row,col].legend()
    if row==2:
        ax[row,col].set_xlabel(r'$k\ [h\ {\rm cMpc}^{-1}]$')
    if col==0:
        ax[row,col].set_ylabel(r'$\Delta^2(k)\ [{\rm mK}^2]$')
plt.show()

print(like_hera.data)

#np.save(path+"data/idr4_screenshot_data/pspec2half_h4c_idr4_fields.npy", like_hera.data)
"""



X_limits = np.array([ #nu_min, nu_max, mean, std
    [0.5, 2], #Lehmer+2012 
    [1, 2], #Hickox & Markevitch (2006)
    [2, 8], #Hickox & Markevitch (2006)
    [8, 24], #Harrison et al. (2016)
    [20, 50], #Harrison et al. (2016)
])

polychordnames = []
for band in redshifts:
    for field in range(2):
        #print("band={0:.2f},field={1}".format(band,field))
        polychordnames.append(("logL_{0:.2f}_{1}".format(band,field), r"\log L_{0:.2f}_{1}".format(band,field)))

polychordnames.append(("logL_Chandra", r"\log L_Chandra"))
for numin,numax in X_limits:
    polychordnames.append(("S_{0}_{1}".format(numin,numax), r"S({0}-{1} keV)".format(numin,numax)))

polychordnames.append(("logL_LWA", r"\log L_LWA"))
for nu in nu_obs:
    polychordnames.append(("T_Radio_z0_{0:.3f}".format(nu/1e9), r"T_Radio_{0:.3f} (z=0))".format(nu/1e9)))


#polychordnames.append(("logL"+str(i), r"\log L"+str(i)))

print(nDerived,len(polychordnames), polychordnames)



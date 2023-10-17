import numpy as np
from codes.emulator_poweremu import *
from codes.likelihood_hera import *
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

if True: #rank==0:
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
             "feed",
             "delay"]

texDict = {"log10fstarII": r"$\log_{10} f_{\rm star, II}$",
           "log10fstarIII": r"$\log_{10} f_{\rm star, III}$",
           "log10Vc": r"$V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "alpha": r"$\alpha$",
           "nu_0": r"$\nu_{\rm 0}$",
           "tau": r"$\tau$",
           "log10fradio": r"$\log_{10} f_{\rm r}$",
           "pop": r"$\rm pop$",
           "feed": r"$\rm feed$",
           "delay": r"$\rm delay$",}

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
             "feed": [-0.5, 1.5],#[0, 1]
             "delay": [-0.5, 1.5]} #[0, 0.75]

discrete_params = {
            "alpha": [-1, -1.3, -1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            "feed": [0, 1],
            "delay": [0, 0.75]}

log_params = [
            "log10fstarII",
            "log10fstarIII",
            "log10Vc",
            "log10fX",
            #"log10tau",
            "log10fradio",
            ]

# Used for initial runs and "final1"
#P = poweremu(loadfile="data/trained_emulators_poweremu/Sims_data_v03_150it_23.02.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
P = poweremu(loadfile="data/trained_emulators_poweremu/Deltasq_emu_n400_l100100100100_t1e-06_o0.pkl", tol=1e-6, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)

like_hera = likelihood(
    #datapath='data/idr4_screenshot_data/idr4_data_dict.npy',
    #datapath='data/idr4_screenshot_data/idr4_screenshot_data_dict.npy',
    #datapath='data/idr4_screenshot_data/idr6_data_dict.npy',
    datapath='data/observations_HERA_IDR2/pspec_h1c_idr2_field{}.h5',
    decimation_factor=2,
    selections={"1": {"1": {"kstart":0.256}},
                "2": {"1": {"kstart":0.192}}}
                )
if True: #rank==0:
    print("Datapath: ", like_hera.datapath.split("/")[-1], flush=True)

bandsNfields = 0
redshifts=[]
for band in like_hera.data.keys():
    for field in like_hera.data[band].keys():
        redshifts.append(like_hera.data[band][field]["z"])
        bandsNfields = bandsNfields+1
#print("bandsNfields=", bandsNfields)
#redshifts = [25.63, 19.27, 16.65, 11.29, 9.82, 8.75, 8.14, 7.60, 7.12]
#nBands = len(like_hera.data.keys())
nDims = len(paramNames)

TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/Pk_Ts_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/Pk_TK_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/Pk_Trad_emu_test_2.pkl", preprocesss_log_x=False, offset=1e-3)
#TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/Ts_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)
#TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)
#TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/Trad_emu_n100_l100100100100_t1e-05_o0.001.pkl", preprocesss_log_x=False, offset=1e-5)

def TS_TK_Trad_from_emulators(p, z=8):
    par0 = np.array([z, *p])
    TS = TS_emu.predict(par0)
    TK = TK_emu.predict(par0)
    TR = TR_emu.predict(par0)
    return TS, TK, TR

def emulatorModel(z, karr, p):
    par0 = np.array([z, np.NaN, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return P.predict(params)

def like_LWA_Chandra(p):
    fsII_i = np.where(np.array(paramNames)=="log10fstarII")[0][0]
    fsIII_i = np.where(np.array(paramNames)=="log10fstarIII")[0][0]
    fr_i = np.where(np.array(paramNames)=="log10fradio")[0][0]
    fX_i = np.where(np.array(paramNames)=="log10fX")[0][0]
    print(fsII_i, fsIII_i, fr_i, fX_i)
    fs_fr = (10**p[fsII_i] + 10**p[fsIII_i]) * 10**p[fr_i]
    fs_fX = (10**p[fsII_i] + 10**p[fsIII_i]) * 10**p[fX_i]

    logL_LWA = 0. if fs_fr<2000. else -np.inf
    logL_Chandra = 0. if fs_fX<1*1.142 else -np.inf
    return logL_LWA, logL_Chandra

def loglikelihood(p, include_LWA=False, include_Chandra=False):
    p=np.copy(p)
    for i,(key,val) in enumerate( zip(paramNames,p) ):
        if key in discrete_params.keys():
            j = round(val)
            p[i] =  discrete_params[key][j]
    m = lambda z,karr,p=p: emulatorModel(z, karr, p) #need to change to pspec_likelihood for IDR4
    logL, individual_loglikes = like_hera.loglike(m, return_individual_loglikes=True) #need to change to pspec_likelihood for IDR4

    logL_LWA, logL_Chandra = like_LWA_Chandra(p)
    #print(logL, logL_LWA, logL_Chandra)
    logL_LWA = logL_LWA if include_LWA else 0.
    logL_Chandra = logL_Chandra if include_Chandra else 0.

    T_emus = np.array([TS_TK_Trad_from_emulators(p,z=zi) for zi in np.unique(redshifts)]).flatten().tolist() #flatten nested tuples and add *extra
    #print(logL, logL_LWA, logL_Chandra)
    return logL+logL_LWA+logL_Chandra, [*T_emus, *individual_loglikes]#[*T8, *T10, *extra]

# Polychord ingredients
_priorBounds = np.array([priorDict_HERA4[p] for p in paramNames])
prior = UniformPrior(_priorBounds.T[0], _priorBounds.T[1])

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)

nDerived = bandsNfields+3*len(np.unique(redshifts)) #2*9 + 3*9 # (selections, number of bands*fields, +6 temperature outputs) # idr4=(9bands*2fields+3temps*9bands), idr2=(2bands*1fields+3temps*9redshifts(AKA bands))
settings = PolyChordSettings(nDims, nDerived)
settings.nlive = 100 #2000
settings.base_dir = 'non-public/idr2_Sims2022_nlive_test_{0}'.format(settings.nlive)
settings.file_root = 'run_IDR2'
settings.do_clustering = True
settings.read_resume = False

pp = [np.log10(1e-0), np.log10(1e-2), np.log10(10), np.log10(20), 2.3, 9.4, 0.05, np.log10(1000), 1.31, -0.11, 1.25] #test
loglikelihood(pp,include_Chandra=True) #test

if False:
    if True: #rank==0:
        print("Starting sampling. Base dir: {0}".format(settings.base_dir), flush=True)

    if True:
        output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
        #print(e, flush=True)
        if True:
            redshifts_str = [str(z) for z in np.unique(redshifts)]
            polychordnames = []
            for p in paramNames:
                polychordnames.append((p, texDict[p][1:-1]))
            for z in redshifts_str:#["8", "10"]:
                for T in ["TS", "TK", "TR"]:
                    polychordnames.append((T+z, r"T_"+T[1]+"\,(z="+z+")"))
            for i in range( nDerived - 3*len(redshifts_str) ):
                polychordnames.append(("logL"+str(i), r"\log L"+str(i)))
            output.make_paramnames_files(polychordnames)

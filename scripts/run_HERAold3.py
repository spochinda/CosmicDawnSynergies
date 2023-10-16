import sys

# Continue, knowing user didn't forget the index
from codes.emulator_poweremu import *
from codes.likelihood_hera import *
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior


priorDict_RadLyA = {
       "log10fStar": [-3, np.log10(0.5)],
             "log10Vc": [np.log10(4.2), 2],
             "log10fX": [-4, 3],
             "tau": [0.035, 0.088],
             "log10Fr": [0, 5]}


texDict = {"Rmfp": r"$R_{\rm mfp}$",
           "log10fStar": r"$\log_{10} f_{\rm star}$",
           "log10Vc": r"$V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "powerInd": r"\alpha_X",
           "numin": r"\nu_{\rm min}",
           "tau": r"$\tau$",
           "log10Fr": r"$\log_{10} f_{\rm r}$",
           "log10Ar": r"$\log_{10} A_{\rm r}$"}


# Used for initial runs and "final1"
#P = poweremu(loadfile="data/trained_emulators_poweremu/Sims_data_v03_150it_23.02.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
P = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m_RadLyA_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)

like_hera = likelihood(
    datapath='data/observations_HERA_IDR3_final/Deltasq_Band_{1:}_Field_{0:}.h5',
    decimation_factor=2,
    selections = {"1": {
            "D": {"kstart":0.356},
            "C": {"kstart":0.356},
            "B": {"kstart":0.294},
            "E": {"kstart":0.417},
            "A": {"kstart":0.417}
        }, "2": {
            "C": {"kstart":0.337},
            "D": {"kstart":0.266},
            "B": {"kstart":0.266},
            "E": {"kstart":0.337},
            "A": {"kstart":0.478}
    }})


paramNames = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
nDerived = 2 * 5 #(selections, number of bands*fields, +6 temperature outputs)
nDims = len(paramNames)

TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TSemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TKemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TRemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)


# Polychord ingredients
_priorBounds = np.array([priorDict_RadLyA[p] for p in paramNames])
prior = UniformPrior(_priorBounds.T[0], _priorBounds.T[1])

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1])

def emulatorModel(z, karr, p):
    rsd = 1
    par0 = np.array([z, np.NaN, *p, rsd])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return P.predict(params)

def loglikelihood(p):
    m = lambda z,karr,p=p: emulatorModel(z, karr, p)
    logL, extra = like_hera.loglike(m, return_individual_loglikes=True)

    return logL, [*extra]


settings = PolyChordSettings(nDims, nDerived)
settings.base_dir = 'idr3_old_chains_final2'
settings.file_root = 'run_IDR3_old'
settings.nlive = 10000
settings.do_clustering = True
settings.read_resume = False

if True:
    output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
    polychordnames = []
    for p in paramNames:
        polychordnames.append((p, texDict[p][1:-1]))
    for i in range(nDerived):
        polychordnames.append(("logL"+str(i), r"\log L"+str(i)))
    output.make_paramnames_files(polychordnames)

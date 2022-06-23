import sys

# Continue, knowing user didn't forget the index
from codes.emulator_poweremu import *
from codes.likelihood_hera import *
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

priorDict_Sims = {
             "Rmfp": [10, 70],
             "log10fStar": [-4, np.log10(0.5)],
             "log10Vc": [np.log10(4.2), 2],
             "log10fX": [-5, 3],
             #"powerInd": [1, 1.3, 1.5], #discrete
             #"numin": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0], #discrete
             "tau": [0.02, 0.1],
             "log10Fr": [-1, 6]}

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
    datapath='data/observations_HERA_IDR2/pspec_h1c_idr2_field{}.h5',
    decimation_factor=2,
    selections={"1": {"1": {"kstart":0.256}, "2": {"kstart":0.320}, "3": {"kstart":0.256}},
                "2": {"1": {"kstart":0.192}, "2": {"kstart":0.192}, "3": {"kstart":0.256}}}
)



paramNames = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
nDerived = 2 * 3 #+ 6 #(selections, number of bands*fields, +6 temperature outputs)
nDims = len(paramNames)

#TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TSemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TKemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TRemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#
#def TS_TK_Trad_from_emulators(p, z=8):
#    emuCols = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
#    par0 = np.array([z, *p[0:4], powerInd, numin, *p[4:6]])
#    TS = TS_emu.predict(par0)
#    TK = TK_emu.predict(par0)
#    TR = TR_emu.predict(par0)
#    return TS, TK, TR


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

    #T8 = TS_TK_Trad_from_emulators(p, z=8)
    #T10 = TS_TK_Trad_from_emulators(p, z=10)

    return logL, [*extra]#[*T8, *T10, *extra]


settings = PolyChordSettings(nDims, nDerived)
settings.base_dir = 'idr2_old_chains_final2'
settings.file_root = 'run_IDR2_old'
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

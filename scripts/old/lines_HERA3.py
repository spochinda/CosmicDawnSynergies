import sys

# Continue, knowing user didn't forget the index
from codes.emulator_poweremu import *
from codes.plotlibs import *
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
P = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)

like_hera = likelihood(
    datapath='data/observations_HERA_IDR3_final/Deltasq_Band_{1:}_Field_{0:}.h5',
    decimation_factor=2,
    selections = {"2": {
            "C": {"kstart":0.337},
            "D": {"kstart":0.266},
            "B": {"kstart":0.266},
            "E": {"kstart":0.337},
            "A": {"kstart":0.478}
    }})


paramNames = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
nDerived = 2 * 5 + 6 #(selections, number of bands*fields, +6 temperature outputs)
nDims = len(paramNames)

# Polychord ingredients
_priorBounds = np.array([priorDict_Sims[p] for p in paramNames])

def emulatorModel(z, karr, p):
    par0 = np.array([z, np.NaN, *p[0:4], np.random.choice([1, 1.3, 1.5]), np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]), *p[4:6]])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return P.predict(params)


fig, ax = plt.subplots()

pp = np.random.uniform(low=_priorBounds.T[0], high=_priorBounds.T[1], size=(10000,6))
kplot = np.linspace(0.1,1,20)
for i in range(10000):
    PS = emulatorModel(8, kplot, pp[i])
    plt.plot(kplot, PS, color="grey")

pp[:, -1] = -1
for i in range(10000):
    PS = emulatorModel(8, kplot, pp[i])
    plt.plot(kplot, PS, color=ccb[0])

like_hera.plot_data(axes=[ax], color=ccb[1])
plt.xlim(0.1,1)
plt.semilogy()
plt.show()
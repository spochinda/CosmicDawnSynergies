from codes.emulator_poweremu import *
from codes.likelihood_hera import *
from codes.loader_21cmSim import *
from codes.plotlibs import *
from codes.tools import *
import pandas as pd
import anesthetic

paramNames = paramNames_Sims_poly
nDerived = 2*5
nDims = len(paramNames)

# LWA
lwa_allowed_z8 = np.load("/data/camHPC/May22/21cm_powerspectra_analysis/arcade_lwa_itamar/lwa_z8_checks_2sigma.npy", allow_pickle=True).item()
lwa_allowed_z10 = np.load("/data/camHPC/May22/21cm_powerspectra_analysis/arcade_lwa_itamar/lwa_z10_checks_2sigma.npy", allow_pickle=True).item()
PT = load_files("data/models_21cmSim/Sims2021/", name="PT", middle="_sims_", endings=["fRad"])
Pk_nodrops = load_files("data/models_21cmSim/Sims2021/", name="Pk", middle="_sims_", endings=["fRad"])
assert np.all(PT == lwa_allowed_z8["params"])
assert np.all(PT == lwa_allowed_z10["params"])

# Likelihoods
like_idr3 = likelihood(
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
            "A": {"kstart":0.478}}})

def loglike_true(data_row, debug=False, **kwargs):
    model = powerspec_of_z_k_hovercMpc(data_row, **kwargs)
    wrapper = lambda z,k: model(z,k)[:,0]
    return like_idr3.loglike(wrapper, debug=debug)
logL_true = [loglike_true(Pk_nodrops[i]) for i in range(len(Pk_nodrops))]
lwa_allowed = lwa_allowed_z8["allowed"]
logL_true, [PT, lwa_allowed] = remove_powerspectra_nans(logL_true, [PT, lwa_allowed])

import scipy.interpolate as sip
import scipy.optimize as sop
import numpy as np

def confidence_level(samples, weights=None, level=0.68):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invcdf = sip.interp1d(CDF, S)
    # Find smallest interval
    distance = lambda a, level=level: invcdf(a+level)-invcdf(a)
    res = sop.minimize(distance, (1-level)/2, bounds=[(0,1-level)], method="Nelder-Mead")
    return np.array([invcdf(res.x[0]), invcdf(res.x[0]+level)])

from codes.loader_21cmSim import *
## 21cmSim uses these redshifts for all outputs, except xHI.
z_array = np.arange(6,50.01,1)
## And these ones for xHI.
z_xHI_array = np.arange(0,30.001,0.1)
# Get the wavenumbers [1/cMpc] from the files. They
# should be all identical but double check for new data.
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])[0]
# Little h for wave number conversions, use h from simulation
h=0.6704

# Tools useful for emulator training data sampling
def powerspec_of_z_k_hovercMpc(data, z_array=z_array, k_array_over_h=k_array/h):
    # Interpolate a given power spectrum (data) at z and k within the respective bounds
    # Make sure to convert to h/cMpc and never use non-h units anywhere anymore
    f = sip.interp2d(z_array, np.log(k_array_over_h), np.log(data+1).T, kind="linear", fill_value=0, bounds_error=False)
    return lambda z,k: np.exp(f(z, np.log(k)))-1

def gen_training(n_over, params, data, fix_z=False, fix_k=False, seed=None, flag=None,
                 zlow=7, zhigh=11, klow=0.02, khigh=3):
    # Sample random z and k from the power spectra interpolations
    # Note: Use k in h/cMpc !
    # n_over = number of random (z,k) samples per model
    # params, data: Parameters and power spectra of models
    # Returns n_over*len(params) samples
    training_x = []
    training_y = []
    if seed is not None:
        np.random.seed(seed)
    for m in np.random.permutation(len(params)):
        p = params[m]
        z = [fix_z]*n_over if fix_z else np.random.uniform(low=zlow, high=zhigh, size=n_over)
        k = [fix_k]*n_over if fix_k else np.random.uniform(low=klow, high=khigh, size=n_over)
        f = powerspec_of_z_k_hovercMpc(data=data[m])
        for j in range(n_over):
            if flag is None:
                training_x.append([z[j],k[j], *p])
            else:
                training_x.append([z[j],k[j], *p, flag])
            r = f(z[j],k[j])
            training_y.append(r)
    indices = np.random.choice(len(training_y), size=len(training_y), replace=False)
    return np.array(training_x)[indices], np.array(training_y)[indices,0]

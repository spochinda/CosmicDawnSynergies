from codes.emulator_poweremu import *
from codes.tools import *
from codes.plotlibs import *
from fgivenx import plot_contours
import numpy as np
import anesthetic
from margarine.maf import MAF
import tensorflow as tf

paramNames = paramNames_RadLyA
N = int(1e4)
np.random.seed(0)
tf.random.set_seed(1)

# Prior samples
priordata = {}
for key in paramNames:
    priordata[key] = np.random.uniform(low=priorDict_Sims[key][0], high=priorDict_Sims[key][1], size=N)
prior = anesthetic.samples.MCMCSamples(priordata, columns=paramNames, label="prior", tex=texDict)

# MAF samples
hera_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/HERA.pkl')
hera_data = hera_maf.sample(N)
hera = anesthetic.samples.MCMCSamples(hera_data, columns=paramNames, label="hera MAF", tex=texDict)
hera.limits["tau"] = [hera.limits["tau"][0], 0.077]
tmp = hera.weights
tmp[hera.tau>0.077]=0
hera.weights = tmp
saras3_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/saras3.pkl')
saras3_data = saras3_maf.sample(N)
saras3 = anesthetic.samples.MCMCSamples(saras3_data, columns=paramNames, label="saras3", tex=texDict)
saras3_hera_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/saras3_hera.pkl')
saras3_hera_data = saras3_hera_maf.sample(N)
saras3_hera = anesthetic.samples.MCMCSamples(saras3_hera_data, columns=paramNames, label="saras3_hera", tex=texDict)

# Original HERA samples
orig_hera_data=np.load("/data/highz/SHdata/HERA_nov_v2/chains/Fr/emcee_flatchain.npy").T[::211]
orig_hera = anesthetic.samples.MCMCSamples(data=orig_hera_data, columns=paramNames, tex=texDict, label='HERA original samples')
tmp = orig_hera.weights
tmp[orig_hera.tau>0.077]=0
orig_hera.weights = tmp
orig_hera.limits["tau"] = [orig_hera.limits["tau"][0], 0.077]

# Plot to check MAF
fig, ax = orig_hera.plot_2d(paramNames, alpha=0.5)
hera.plot_2d(ax, alpha=0.5)
plt.legend()
plt.show()

kwargs = {"alpha":0.5, "types":{"lower": "scatter", "diagonal": "hist", "upper": "kde"}, "diagonal_kwargs":{"histtype": "step"}, "ncompress":5000}
#kwargs["upper_kwargs"] = { "hatches": ["**", "*"]}
fig, ax = saras3_hera.plot_2d(paramNames, **kwargs)
#kwargs["upper_kwargs"] = { "hatches": ["..", "."]}
saras3.plot_2d(ax, **kwargs)
#kwargs["upper_kwargs"] = { "hatches": ["oo", "o"]}
hera.plot_2d(ax, **kwargs)
plt.legend()
plt.show()

assert not np.any(np.isnan(saras3))
assert not np.any(np.isinf(saras3))
assert not np.any(np.isnan(saras3_hera))
assert not np.any(np.isinf(saras3_hera))
assert not np.any(np.isnan(hera))
assert not np.any(np.isinf(hera))

# Load emulator
P = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m4_RadLyA_adaptive.pkl",preprocesss_log_x=False)

def model_of_z(zarr, p, k=0.2, rsd=1):
    par0 = np.array([np.NaN, k, *p, rsd])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return P.predict(s)

#model_of_z(np.linspace(7,31,100), hera.iloc[0][paramNames])

def model_of_k(karr, p, z=8, rsd=1):
    par0 = np.array([z, np.NaN, *p, rsd])
    s=np.tile(par0, (len(karr), 1))
    s[:,1] = karr
    return P.predict(s)


# Plot contours

fig, ax = plt.subplots()
fig.suptitle("Constraints at k=0.2 h/Mpc\nPrior(grey), SARAS3(blue), HERA(orange), both(green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.5}
cbar_prior = plot_contours(model_of_z, np.linspace(7,30,100), np.array(prior), ax=ax, colors=plt.cm.Greys_r, cache="/tmp/fgivenx/prior", **kwargs)
cbar_post = plot_contours(model_of_z, np.linspace(8,30,100), np.array(saras3), ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx/s", **kwargs)
cbar_post = plot_contours(model_of_z, np.linspace(7,30,100), np.array(hera), ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/fgivenx/hera", **kwargs)
#cbar_post = plot_contours(model_of_z, np.linspace(7,30,100), np.array(orig_hera), weights=orig_hera.weights, ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/fgivenx/herao", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.7)
cbar_post = plot_contours(model_of_z, np.linspace(7,30,100), np.array(saras3_hera), ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/fgivenx/sh", **kwargs)
ax.set_yscale("log")
ax.set_xlabel("Redshift z")
ax.set_ylabel("Power spectrum Delta²2 [mK²]")
cbar = plt.colorbar(cbar_prior,ticks=[0,1,2,3], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
ax.set_ylim(1e2,1e6)
ax.set_xlim(7,30)
plt.show()


from codes.emulator_poweremu import *
from fgivenx import plot_contours
import numpy as np
import anesthetic

# Load combined data set from Harry
paramNames = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
SARASHERA = anesthetic.NestedSamples(root="/data/highz2/HBdata2/harry_stefan_joint_analysis/saras3_hera_nlive_500/test", columns=paramNames, label="HERA IDR2 + SARAS3")[paramNames]
assert not np.any(np.isnan(SARASHERA))
assert not np.any(np.isinf(SARASHERA))

# Load emulator
P = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_RadLyA_June06_adaptive.pkl",preprocesss_log_x=False)
# HERA optimized with z up to 12
z = 8
def model(karr, p, z=z, rsd=1):
    par0 = np.array([z, np.NaN, *p, rsd])
    s=np.tile(par0, (len(karr), 1))
    s[:,1] = karr
    return P.predict(s)

# Make fgivenx plot
fig, ax = plt.subplots()
post = plot_contours(model, np.linspace(0.02,1.5,100), np.array(SARASHERA), weights=SARASHERA.weights, ax=ax, colors=plt.cm.YlOrBr_r, cache="/tmp/fgivenx/r", fineness=1, contour_color_levels=[0,1,2,3], lines=False)

fig.suptitle("SARAS3 + HERA at z="+str(z))
ax.set_yscale("log")
ax.set_xlabel("Wavenumber k [h/Mpc]")
ax.set_ylabel("Power spectrum Delta²2 [mK²]")
cbar = plt.colorbar(post,ticks=[0,1,2,3], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
plt.ylim(1e2,1e6)
plt.xlim(0.02,3)
plt.show()

# Load HERA-only data
paramNames = ["log10fStar", "log10Vc", "log10fX", "tau", "log10"]
paramNames[-1] += "Fr"
paramTex = [r"$\log_{10} f_{\rm *}$", r"$\log_{10}V_c$", r"$\log_{10} f_{\rm X}$", r"$\tau$", r"$\log_{10} f_{\rm r}$"]
texDict = dict(zip(paramNames, paramTex))
prior_bounds = np.array([[-3,np.log10(0.5)],[np.log10(4.2),2],[-4,3],[0.035,0.088],[0, 5]])
priorDict = dict(zip(paramNames, prior_bounds))

data=np.load("/data/highz/SHdata/HERA_nov_v2/chains/Fr/emcee_flatchain.npy").T
HERA = anesthetic.samples.MCMCSamples(data=data, columns=paramNames, tex=texDict, limits=priorDict).iloc[::281]
HERA.label='HERA IDR2 Posterior'
tmp = HERA.weights
tmp[HERA.tau>0.077]=0
HERA.weights = tmp

print("Got {0:.1e} HERA".format(len(HERA)))
print("emcee MCMC HERA, all weights = 1")

# Comparison fgivenx plot
fig, ax = plt.subplots()
post = plot_contours(model, np.linspace(0.02,3,100), np.array(SARASHERA), weights=SARASHERA.weights, ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/fgivenx/r", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
post2= plot_contours(model, np.linspace(0.02,3,100), np.array(HERA), weights=HERA.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx/r2", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
fig.suptitle("SARAS3 + HERA (orange) at z="+str(z)+" vs HERA alone (blue)")
ax.set_yscale("log")
ax.set_xlabel("Wavenumber k [h/Mpc]")
ax.set_ylabel("Power spectrum Delta²2 [mK²]")
cbar = plt.colorbar(post,ticks=[0,1,2,3], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
plt.ylim(1e1,1e6)
plt.xlim(0.02,3)
plt.show()

# Comparison corner plot
fig, ax = HERA.plot_2d(paramNames, alpha=0.5, types={"lower": "kde", "diagonal": "hist", "upper": "scatter"})
fig, ax = SARASHERA.plot_2d(ax, alpha=0.5, types={"lower": "kde", "diagonal": "hist", "upper": "scatter"})
plt.legend()
plt.show()

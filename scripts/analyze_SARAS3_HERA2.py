from codes.emulator_poweremu import *
from codes.tools import *
from codes.plotlibs import *
from fgivenx import plot_contours, plot_lines
import fgivenx
import numpy as np
import anesthetic
from margarine.maf import MAF
import tensorflow as tf
# Basics
paramNames = paramNames_RadLyA
N_sample_MAF = int(1e5)
np.random.seed(0)
tf.random.set_seed(1)


################ Load samples ################

# Prior samples
priordata = {}
for key in paramNames:
    priordata[key] = np.random.uniform(low=priorDict_Sims[key][0], high=priorDict_Sims[key][1], size=N_sample_MAF)
prior = anesthetic.samples.MCMCSamples(priordata, columns=paramNames, label="prior", tex=texDict)

# MAF samples
hera_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/HERA.pkl')
hera_data = hera_maf.sample(N_sample_MAF)
hera = anesthetic.samples.MCMCSamples(hera_data, columns=paramNames, label="HERA (MAF)", tex=texDict)
hera.limits["tau"] = [hera.limits["tau"][0], 0.077]
tmp = hera.weights
tmp[hera.tau>0.077]=0
hera.weights = tmp
saras3_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/saras3.pkl')
saras3_data = saras3_maf.sample(N_sample_MAF)
saras3 = anesthetic.samples.MCMCSamples(saras3_data, columns=paramNames, label="SARAS3", tex=texDict)
saras3_hera_maf = MAF.load('/data/highz2/HBdata2/harry_stefan_joint_analysis/saras3_hera.pkl')
saras3_hera_data = saras3_hera_maf.sample(N_sample_MAF)
saras3_hera = anesthetic.samples.MCMCSamples(saras3_hera_data, columns=paramNames, label="SARAS3 + HERA", tex=texDict)

# Original HERA samples
#orig_hera_data=np.load("/data/highz/SHdata/HERA_nov_v2/chains/Fr/emcee_flatchain.npy").T[::211]
#orig_hera = anesthetic.samples.MCMCSamples(data=orig_hera_data, columns=paramNames, tex=texDict, label='HERA (original)')
#tmp = orig_hera.weights
#tmp[orig_hera.tau>0.077]=0
#orig_hera.weights = tmp
#orig_hera.limits["tau"] = [orig_hera.limits["tau"][0], 0.077]

# Plot to check MAF
#fig, ax = orig_hera.plot_2d(paramNames, alpha=0.5)
#hera.plot_2d(ax, alpha=0.5)
#plt.legend()
#plt.show()

### kwargs = {"alpha":0.5, "types":{"lower": "scatter", "diagonal": "hist", "upper": "kde"}, "diagonal_kwargs":{"histtype": "step"}, "ncompress":5000}
### #kwargs["upper_kwargs"] = { "hatches": ["**", "*"]}
### fig, ax = saras3_hera.plot_2d(paramNames, **kwargs)
### #kwargs["upper_kwargs"] = { "hatches": ["..", "."]}
### saras3.plot_2d(ax, **kwargs)
### #kwargs["upper_kwargs"] = { "hatches": ["oo", "o"]}
### hera.plot_2d(ax, **kwargs)
### plt.legend()
### plt.show()

assert not np.any(np.isnan(saras3))
assert not np.any(np.isinf(saras3))
assert not np.any(np.isnan(saras3_hera))
assert not np.any(np.isinf(saras3_hera))
assert not np.any(np.isnan(hera))
assert not np.any(np.isinf(hera))


################ Load emulators ################

P = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m4_RadLyA_adaptive.pkl",preprocesss_log_x=False)
TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TR_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TS_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
SFR_emu = poweremu(loadfile="data/trained_emulators_poweremu/SFR_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-25)

def add_vals_from_emulators(df, z=30):
    emuCols = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
    s = np.shape(df)
    arr = np.empty([s[0],len(emuCols)+1])
    arr[:,0] = z
    arr[:,1:] = df[emuCols]
    TK = TK_emu.predict(arr)
    TR = TR_emu.predict(arr)
    TS = TS_emu.predict(arr)
    SFR = SFR_emu.predict(arr)
    df["log10TK_z="+str(z)] = np.log10(TK)
    df["log10TR_z="+str(z)] = np.log10(TR)
    df["log10TS_z="+str(z)] = np.log10(TR)
    df["log10TS_over_TR_z="+str(z)] = np.log10(TS) - np.log10(TR)
    df["log10SFR_z="+str(z)] = np.log10(SFR)
    df["log10_TK_over_TR_z="+str(z)] = np.log10(TK) - np.log10(TR)

add_vals_from_emulators(saras3, z=20)
add_vals_from_emulators(saras3, z=25)
add_vals_from_emulators(saras3, z=15)

add_vals_from_emulators(hera, z=8)
add_vals_from_emulators(hera, z=10)
add_vals_from_emulators(saras3_hera, z=8)
add_vals_from_emulators(saras3_hera, z=10)


def TK_adiabatic(z):
    T30 = 15.0024 #10**np.min(prior["log10TK_z=30"])
    return T30/31**2*(1+z)**2

def Tcmb(z):
    T0 = 2.725
    return T0*(1+z)

def log10SFR_of_z(zarr, p):
    par0 = np.array([np.NaN, *p])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.log10(SFR_emu.predict(s))

def log10TR_over_TK_of_z(zarr, p):
    par0 = np.array([np.NaN, *p])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.log10(TS_emu.predict(s)) - np.log10(TK_emu.predict(s))

def log10TR_over_TK_adiabatic(zarr):
    return np.log10(Tcmb(zarr)) - np.log10(TK_adiabatic(zarr))

def log10TR_over_TS_of_z(zarr, p):
    par0 = np.array([np.NaN, *p])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.log10(TR_emu.predict(s)) - np.log10(TS_emu.predict(s))

def log10TS_over_TR_of_z(zarr, p):
    return -log10TR_over_TS_of_z(zarr, p)

def log10TK_of_z(zarr, p):
    par0 = np.array([np.NaN, *p])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.log10(TK_emu.predict(s))

def PS_of_z(zarr, p, k=0.2, rsd=1):
    par0 = np.array([np.NaN, k, *p, rsd])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.minimum(1e10, P.predict(s))

def log10PS_of_z(zarr, p, k=0.2, rsd=1):
    par0 = np.array([np.NaN, k, *p, rsd])
    s=np.tile(par0, (len(zarr), 1))
    s[:,0] = zarr
    return np.log10(P.predict(s)+1)

#######################################
################ Plots ################
#######################################

################ TS HERA-like ################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax = plt.subplots()
fig.set_size_inches((6,4))
fig.set_suptitle("Todo: 2nd x axis; 1+z?")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.8}
cachefolder = "fgivenx_TS"
zarr = np.linspace(7,30,24)
##cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, prior[paramNames], weights=prior.weights, ax=ax, colors=plt.cm.Greys_r, cache="/tmp/"+cachefolder+"/prior", **kwargs)
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], weights=hera.weights, ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/"+cachefolder+"/hera", **kwargs)
##cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], weights=saras3.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/"+cachefolder+"/saras3", **kwargs)
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], ny=100, weights=saras3_hera.weights, ax=ax, colors=plt.cm.Reds_r, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)


cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], fastCI_contours=True, weights=saras3_hera.weights, ax=ax, color=cdefault[3], level=0.95, alpha=0.5, cache="/tmp/"+cachefolder+"/saras3_hera")
cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], fastCI_contours=True, weights=saras3_hera.weights, ax=ax, color=cdefault[3], level=0.68, alpha=1, cache="/tmp/"+cachefolder+"/saras3_hera")

#4b:
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], fastCI_contours=False, weights=saras3_hera.weights, ax=ax, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)


cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color=cdefault[2], level=0.95, cache="/tmp/"+cachefolder+"/hera", lines_only=True, lw=2, ls="--")
cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color=cdefault[2], level=0.68, cache="/tmp/"+cachefolder+"/hera", lines_only=True, lw=2)

cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color=cdefault[1], level=0.95, cache="/tmp/"+cachefolder+"/saras", lines_only=True, lw=2, ls="--")
cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color=cdefault[1], level=0.68, cache="/tmp/"+cachefolder+"/saras", lines_only=True, lw=2)

#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], fastCI_contours=True, weights=saras3_hera.weights, ax=ax, color=plt.cm.Reds(255), level=0.975, alpha=0.4, cache="/tmp/"+cachefolder+"/saras3_hera", method="lower-limit")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], fastCI_contours=True, weights=saras3_hera.weights, ax=ax, color=plt.cm.Reds(255), level=0.68+0.16, alpha=0.8, cache="/tmp/"+cachefolder+"/saras3_hera", method="lower-limit")
#
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color="green", level=0.975, cache="/tmp/"+cachefolder+"/hera", lines_only=True, lw=2, ls="--", method="lower-limit")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color="darkgreen", level=0.68+0.16, cache="/tmp/"+cachefolder+"/hera", lines_only=True, lw=2, method="lower-limit")
#
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color="orange", level=0.975, cache="/tmp/"+cachefolder+"/saras", lines_only=True, lw=2, ls="--", method="lower-limit")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color="darkorange", level=0.68+0.16, cache="/tmp/"+cachefolder+"/saras", lines_only=True, lw=2, method="lower-limit")



#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color="lime", level=0.95, alpha=0.8, cache="/tmp/"+cachefolder+"/hera2")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color="olive", level=0.68, alpha=0.8, cache="/tmp/"+cachefolder+"/hera2")
#fsampls = fgivenx.drivers.compute_fsamps(log10TS_over_TR_of_z, zarr, hera[paramNames], fastCI_contours=True, weights=hera.weights, ax=ax, color="red", alpha=0.8, cache="/tmp/"+cachefolder+"/hera2")
#lower[1] = -0.7057455411808649 @ 0.95
#lower[1] = 0.02144731443188352 @ 0.68
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color="yellow", level=0.95, alpha=0.8, cache="/tmp/"+cachefolder+"/saras3")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3[paramNames], fastCI_contours=True, weights=saras3.weights, ax=ax, color="orange", alpha=0.8, cache="/tmp/"+cachefolder+"/saras3")
#cbar_post = plot_contours(log10TS_over_TR_of_z, zarr, saras3_hera[paramNames], ny=100, weights=saras3_hera.weights, ax=ax, colors=plt.cm.Reds_r, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)

#zplot = np.arange(6,31,1)
#ax.scatter(zplot, [confidence_level(hera["log10TS_over_TR_z={:}".format(z)], level=0.68)[0] for z in zplot])
#ax.scatter(zplot, [confidence_level(hera["log10TS_over_TR_z={:}".format(z)], level=0.95)[0] for z in zplot])
# HERA arrows
#ax.errorbar(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.68)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkgreen")
#ax.errorbar(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="green")
#ax.errorbar(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.68)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkgreen")
#ax.errorbar(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="green")
ax.scatter(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.68)[0], marker="^", edgecolors="k", color=cdefault[2], zorder=10, s=80)
ax.scatter(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.95)[0], marker="^", edgecolors="k", color=cdefault[2], zorder=10, s=80, alpha=0.5)
ax.scatter(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.68)[0], marker="^", edgecolors="k", color=cdefault[2], zorder=10, s=80)
ax.scatter(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.95)[0], marker="^", edgecolors="k", color=cdefault[2], zorder=10, s=80, alpha=0.5)

#plt.scatter(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.68)[0], marker="^", color="orange")
#plt.scatter(10, confidence_level(hera["log10TS_over_TR_z=10"], weights=hera.weights, level=0.95)[0], marker="^", color="darkorange")
#lt.scatter(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.68)[0], marker="^", color="orange")
#lt.scatter(8, confidence_level(hera["log10TS_over_TR_z=8"], weights=hera.weights, level=0.95)[0], marker="^", color="darkorange")

# SARAS arrows
#ax.errorbar(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.68)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="orange")
#ax.errorbar(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkorange")
#ax.errorbar(25, confidence_level(hera["log10TS_over_TR_z=25"], weights=hera.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkgreen")
ax.scatter(15, confidence_level(saras3["log10TS_over_TR_z=15"], weights=saras3.weights, level=0.95)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80, alpha=0.5)
ax.scatter(20, confidence_level(saras3["log10TS_over_TR_z=20"], weights=saras3.weights, level=0.95)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80, alpha=0.5)
ax.scatter(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.95)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80, alpha=0.5)
ax.scatter(15, confidence_level(saras3["log10TS_over_TR_z=15"], weights=saras3.weights, level=0.68)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80)
ax.scatter(20, confidence_level(saras3["log10TS_over_TR_z=20"], weights=saras3.weights, level=0.68)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80)
ax.scatter(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.68)[0], marker="^", edgecolors="k", color=cdefault[1], zorder=10, s=80)
#ax.errorbar(17, confidence_level(saras3_hera["log10TS_over_TR_z=17"], weights=saras3_hera.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkorange")
#ax.errorbar(20, confidence_level(hera["log10TS_over_TR_z=20"], weights=hera.weights, level=0.95)[0], yerr=0.1, capsize=6, markeredgewidth=3, lolims=True, color="darkgreen")
#plt.scatter(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.68)[0], marker="^", color="lightblue")
#plt.scatter(25, confidence_level(saras3["log10TS_over_TR_z=25"], weights=saras3.weights, level=0.95)[0], marker="^", color="blue")

#EDGES
#ax.scatter(17, np.log10(0.0762)) #yerr=np.array([[0.03659, 0.0447]]).T

ax.plot(zarr, -log10TR_over_TK_adiabatic(zarr), lw=3, ls="dotted", color="black", label=r"$T_{\rm Adiabatic}/T_{\rm CMB}$")
#colorbar = fig.colorbar(cbar_post, ticks=[0,1,2], label='Posterior', pad=0.1)
ax.set_xlabel("z")
ax.set_ylabel(r"$T_{\rm s}/T_{\rm r}$")
ax.set_xlim(7,30)
ax.set_ylim(-2,0)
ax.set_yticks(np.log10(np.geomspace(10**(-2),10**(0),11)))
ax.set_yticklabels([r"$10^{{{:.1f}}}$".format(a) for a in np.log10(np.geomspace(10**(-2),10**(0),11))])
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("TS_figure_4.png", dpi=600)
plt.savefig("TS_figure_4.pdf", dpi=600)
plt.show()








assert False





################ TS ################
fig, ax = plt.subplots()
fig.suptitle("TS (z) for SARAS (blue), HERA (orange), and combined (green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.5}
cachefolder = "fgivenx_TS"
cbar_post = plot_contours(log10TR_over_TS_of_z, zarr, prior[paramNames], weights=prior.weights, ax=ax, colors=plt.cm.Greys_r, cache="/tmp/"+cachefolder+"/prior", **kwargs)
cbar_post = plot_contours(log10TR_over_TS_of_z, np.linspace(7,30,10), hera[paramNames], weights=hera.weights, ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/"+cachefolder+"/hera", **kwargs)
cbar_post = plot_contours(log10TR_over_TS_of_z, np.linspace(7,30,10), saras3[paramNames], weights=saras3.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/"+cachefolder+"/saras3", **kwargs)
cbar_post = plot_contours(log10TR_over_TS_of_z, np.linspace(7,30,10), saras3_hera[paramNames], weights=saras3_hera.weights, ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)
ax.plot(zarr, log10TR_over_TK_adiabatic(zarr), lw=3, ls="dotted", color="black", label="Tcmb/Tadiabatic")
colorbar = fig.colorbar(cbar_post, ticks=[0,1,2], label='Posterior', pad=0.1)
ax.set_xlabel("z")
ax.set_ylabel(r"$\log10 (T_{\rm radio}/T_{\rm spin})$")
ax.set_xlim(7,30)
ax.set_ylim(-3,3)
plt.legend()
plt.show()

################ TK ################
fig, ax = plt.subplots()
fig.suptitle("TK (z) for SARAS (blue), HERA (orange), and combined (green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.5}
cachefolder = "fgivenx_TK"
cbar_post = plot_contours(log10TR_over_TK_of_z, zarr, prior[paramNames], weights=prior.weights, ax=ax, colors=plt.cm.Greys_r, cache="/tmp/"+cachefolder+"/prior", **kwargs)
cbar_post = plot_contours(log10TR_over_TK_of_z, zarr, hera[paramNames], weights=hera.weights, ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/"+cachefolder+"/hera", **kwargs)
cbar_post = plot_contours(log10TR_over_TK_of_z, zarr, saras3[paramNames], weights=saras3.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/"+cachefolder+"/saras3", **kwargs)
cbar_post = plot_contours(log10TR_over_TK_of_z, zarr, saras3_hera[paramNames], weights=saras3_hera.weights, ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)
ax.plot(zarr, log10TR_over_TK_adiabatic(zarr), lw=3, color="black", label="Tcmb/Tadiabatic")
colorbar = fig.colorbar(cbar_post, ticks=[0,1,2], label='Posterior', pad=0.1)
ax.set_xlabel("z")
ax.set_ylabel(r"$\log10 (T_{\rm radio}/T_{\rm gas})$")
ax.set_xlim(7,30)
ax.set_ylim(-2,3)
plt.legend()
plt.show()

################ SFR ################
fig, ax = plt.subplots()
fig.suptitle("SFR (z) for SARAS (blue), HERA (orange), and combined (green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,2], "lines": False, "alpha": 0.5}
#kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": True, "alpha": 1, "linewidths": 2, "facecolors": None}
cachefolder = "fgivenx_SFR"
cbar_post = plot_contours(log10SFR_of_z, np.linspace(7,30,10), prior[paramNames], weights=prior.weights, ax=ax, colors=plt.cm.Greys_r, cache="/tmp/"+cachefolder+"/prior", **kwargs)
cbar_post = plot_contours(log10SFR_of_z, np.linspace(7,29,10), hera[paramNames], weights=hera.weights, ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/"+cachefolder+"/hera", **kwargs)
cbar_post = plot_contours(log10SFR_of_z, np.linspace(7,28,10), saras3[paramNames], weights=saras3.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/"+cachefolder+"/saras3", **kwargs)
cbar_post = plot_contours(log10SFR_of_z, np.linspace(7,27,10), saras3_hera[paramNames], weights=saras3_hera.weights, ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/"+cachefolder+"/saras3_hera", **kwargs)
colorbar = fig.colorbar(cbar_post, ticks=kwargs["contour_color_levels"], label='Posterior', pad=0.1)
ax.set_ylim(-10,0)
ax.set_xlabel("z")
ax.set_ylabel(r"$\log10 (SFR)$")
ax.set_xlim(7,30)
plt.show()


assert False, "That's all the relevant code"
### LEGACY CODE BELOW ###



from codes.emulator_poweremu import *
from codes.loader_21cmSim import *
from codes.tools import *
import matplotlib.pyplot as plt
from copy import deepcopy
# Redshift and k ranges used in data, and load params and powerspectra
## 21cmSim uses these redshifts for all outputs, except xHI.
z_array = np.arange(6,50.01,1)
## And these ones for xHI.
z_xHI_array = np.arange(0,30.001,0.1)
## Finally get the wavenumbers [1/cMpc] from the files. They
## should be all identical but double check for new data.
#k_array = load_files('data/models_21cmSim/EmulatorPS/', name='KK', key='KK', middle="", endings=[""])[0]
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="old")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="old")
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])
k_array = k_array[0]
# Little h for wave number conversions, use h from simulation
h=0.6704


## And these with RSDs (2181-6):
PT = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="PT", endings=["fRad_RSDrand"], middle=None, key="PTout")
Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", endings=["fRad_RSDrand1"], middle=None, key="PKout1")
Pk_RSD_Itamar, [PT] = remove_powerspectra_nans(Pk, [PT])
PL_RSD_Itamar = PT9_to_PL5(PT)

greysamples = []
redsamples = []
ki = np.argmin(np.abs(k_array-0.2))
zi = np.argmin(np.abs(z_array-10))
for i in range(len(Pk_RSD_Itamar)):
    if Pk_RSD_Itamar[i,zi,ki] < 500:
        color="red"
        redsamples.append(PL_RSD_Itamar[i])
    else:
        color="grey"
        greysamples.append(PL_RSD_Itamar[i])
    plt.plot(z_array, Pk_RSD_Itamar[i,:,ki], color=color, alpha=0.5)

plt.semilogy()
plt.ylim(1e1, 1e7)
plt.show()

r = anesthetic.samples.MCMCSamples(redsamples, columns=paramNames_RadLyA)
g = anesthetic.samples.MCMCSamples(greysamples, columns=paramNames_RadLyA)
#fig, ax = g.plot_2d(paramNames_RadLyA, color="grey")
fig, ax = r.plot_2d(paramNames_RadLyA, color="red")
plt.show()

fig, ax = plt.subplots()
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2,3], "lines": False, "alpha": 0.5}
cbar_prior = plot_contours(log10model_of_z, zarr, np.array(greysamples), ax=ax, colors=plt.cm.Greys_r, cache="/tmp/grey", **kwargs)
cbar_prior = plot_contours(log10model_of_z, zarr, np.array(redsamples), ax=ax, colors=plt.cm.Reds_r, cache="/tmp/red", **kwargs)
ax.set_xlabel("z")
ax.set_ylabel("Power spectrum Delta²2 [mK²]")
ax.set_ylim(0,6)
ax.set_xlim(7,30)
plt.show()







prior.log10_TK_over_TR.hist(bins=100, histtype="step", label="prior", color="grey", lw=2, density=True)
hera.log10_TK_over_TR.hist(bins=100, histtype="step", label="HERA", color="orange", lw=2, density=True)
saras3.log10_TK_over_TR.hist(bins=100, histtype="step", label="SARAS3", color="blue", lw=2, density=True)
saras3_hera.log10_TK_over_TR.hist(bins=100, histtype="step", label="SARAS3+HERA", color="green", lw=2, density=True)
plt.xlabel("log10 ( Trad/TK )")
plt.legend()
plt.show()

def make_mask(samples):
    fr = samples.log10Fr
    fX = samples.log10fX
    fstar = samples.log10fStar
    return np.logical_and(fstar+fX<np.log10(1.141), fstar+fr<np.log10(2e3))


prior_mask = make_mask(prior)
saras3_mask = make_mask(saras3)
hera_mask = make_mask(hera)
saras3_hera_mask = make_mask(saras3_hera)

fig, ax = plt.subplots()
fig.suptitle("Constraints at k=0.2 h/Mpc\nPrior(grey), SARAS3(blue), HERA(orange), both(green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.5}
cbar_prior = plot_contours(log10model_of_z, zarr, np.array(prior), ax=ax, colors=plt.cm.Greys_r, cache="/tmp/fgivenx2y/prior", **kwargs)
cbar_post = plot_contours(log10model_of_z, np.linspace(8,30,100), np.array(saras3), ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx2y/s", **kwargs)
cbar_post = plot_contours(log10model_of_z, np.linspace(8,30,10), np.array(hera), ax=ax, colors=plt.cm.Oranges_r, cache="/tmp/fgivenx2y/hera", **kwargs)
#cbar_post = plot_contours(log10model_of_z, np.linspace(8,12,10), np.array(hera), ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx2y2/hera", **kwargs)
cbar_post = plot_contours(log10model_of_z, zarr, np.array(saras3_hera), ax=ax, colors=plt.cm.YlGn_r, cache="/tmp/fgivenx2y/sh", **kwargs)
ax.set_xlabel("Redshift z")
ax.set_ylabel("log10 Power spectrum Delta²2 [mK²]")
#cbar = plt.colorbar(cbar_prior,ticks=[0,1,2], label="Posterior")
#cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$'])
ax.set_ylim(0,7)
ax.set_xlim(7,30)


fig, ax = plt.subplots()
fig.suptitle("Constraints at k=0.2 h/Mpc\nPrior(grey), SARAS3(blue), HERA(orange), both(green)")
kwargs = {"fineness": 1, "contour_color_levels": [0,1,2], "lines": False, "alpha": 0.5}
cbar_prior = plot_contours(log10model_of_z, zarr, np.array(prior), ax=ax, weights=prior_mask, colors=plt.cm.Greys_r, cache="/tmp/mfgivenx2y/prior", **kwargs)
cbar_post = plot_contours(log10model_of_z, np.linspace(8,30,100), np.array(saras3), ax=ax, weights=saras3_mask, colors=plt.cm.Blues_r, cache="/tmp/mfgivenx2y/s", **kwargs)
cbar_post = plot_contours(log10model_of_z, np.linspace(8,30,10), np.array(hera), ax=ax, weights=hera_mask, colors=plt.cm.Oranges_r, cache="/tmp/mfgivenx2y/hera", **kwargs)
cbar_post = plot_contours(log10model_of_z, zarr, np.array(saras3_hera), ax=ax, weights=saras3_hera_mask, colors=plt.cm.YlGn_r, cache="/tmp/mfgivenx2y/sh", **kwargs)
ax.set_xlabel("Redshift z")
ax.set_ylabel("Power spectrum Delta²2 [mK²]")
#cbar = plt.colorbar(cbar_prior,ticks=[0,1,2], label="Posterior")
#cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$'])
ax.set_ylim(0,7)
ax.set_xlim(7,30)
plt.show()


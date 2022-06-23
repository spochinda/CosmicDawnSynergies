from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mpc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
ccb = sns.color_palette("colorblind")
# Matplotlib settings
params = {'legend.fontsize':  12,
          'figure.figsize': (6, 5),
         'axes.labelsize':  14,
         'axes.titlesize': 14,
         'xtick.labelsize': 12,
         'ytick.labelsize': 12}
plt.rcParams.update(params)

import pandas as pd
import anesthetic
import numpy as np
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.integrate as sin
from copy import deepcopy

def trapezoidal_bump(a,b,c,d, peak=1):
    return sip.interp1d([a,b,c,d], [0,peak,peak,0], fill_value=(0,0), bounds_error=False)

def sum_pdf_1d(alpha, xmin, xmax, ymin, ymax):
    # PDF for alpha which is the sum of two uniformly distributed
    # random variables x and y, alpha = x + y
    a = xmin+ymin
    b = ymin+xmax#np.minimum(xmin+ymax, ymin+xmax)
    c = xmin+ymax#np.maximum(xmin+ymax, ymin+xmax)
    d = xmax+ymax
    f = trapezoidal_bump(a,b,c,d)
    norm = sin.quad(f,a,d)[0]
    return f(alpha)/norm

def sum_pdf_2d(alpha, beta, xmin, xmax, ymin, ymax, zmin, zmax, debug=False):
    # 2D PDF in (alpha, beta) where the alpha = x + z and beta = y + z
    # where x, y and z are uniformly distributed random variables.
    # Approach to calculate this: Compute the beta-1d-pdf for every alpha:
    # There are (up to) 5 distinct regimes. Use empirical formulas. Proof: Todo.
    alphamin = xmin+zmin 
    betamin = ymin+zmin
    alphamax = xmax+zmax
    betamax = ymax+zmax
    alphalow = np.minimum(xmax+zmin, xmin+zmax)
    alphaup = np.maximum(xmax+zmin, xmin+zmax)
    betalow = np.minimum(ymax+zmin, ymin+zmax)
    betaup = np.maximum(ymax+zmin, ymin+zmax)
    p1_prelim = xmax+zmin
    p2_prelim = xmin+zmax
    if p1_prelim>p2_prelim:
        yellow = "rect"
    else:
        yellow = "diag"
    if debug:
        print("yellow =", yellow)
    p1 = np.minimum(p1_prelim, p2_prelim)
    p2 = np.maximum(p1_prelim, p2_prelim)
    if debug:
        print("p1", p1)
        print("p2", p2)
        print("alphamin", alphamin)
    overallnorm = trapezoidal_bump(alphamin, p1, p2, alphamax)(alpha)
    if yellow=="rect":
        a = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin, betamin, betalow], fill_value=0, bounds_error=False)(alpha)
        b = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betalow, betalow, betalow], fill_value=0, bounds_error=False)(alpha)
        c = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betaup, betaup, betaup, betamax], fill_value=0, bounds_error=False)(alpha)
        d = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betaup, betamax, betamax, betamax], fill_value=0, bounds_error=False)(alpha)
        if debug:
            print("a,b,c,d", a,b,c,d)
        overallnorm = trapezoidal_bump(alphamin, alphalow, alphaup, alphamax)(alpha)
        return trapezoidal_bump(a,b,c,d,peak=overallnorm)(beta)
    elif yellow=="diag":
        assert betalow+(alphaup-alphamin) == betamax
        a = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin, betamin+(alphaup-alphalow), betamin+(alphamax-alphalow)], fill_value=0, bounds_error=False)(alpha)
        b = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin+(alphalow-alphamin), betamin+(alphaup-alphamin), betamin+(alphaup-alphamin)], fill_value=0, bounds_error=False)(alpha)
        c = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betalow, betalow, betalow+(alphaup-alphalow), betalow+(alphamax-alphalow)], fill_value=0, bounds_error=False)(alpha)
        d = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betalow, betalow+(alphalow-alphamin), betalow+(alphaup-alphamin), betalow+(alphaup-alphamin)], fill_value=0, bounds_error=False)(alpha)
        overallnorm = trapezoidal_bump(alphamin, alphalow, alphaup, alphamax)(alpha)
        return trapezoidal_bump(a,b,c,d,peak=overallnorm)(beta)
    else:
        assert False, yellow

def powerInd_and_numin_from_index(index):
    powerInds = [1, 1.3, 1.5]
    numins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]
    powerInd = powerInds[int(index/len(numins))]
    numin = numins[index % len(numins)]
    return powerInd, numin

paramNames_Sims_poly = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
paramNames_Sims_full = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
priorDict_Sims = {
             "Rmfp": [10, 70],
             "log10fStar": [-4, np.log10(0.5)],
             "log10Vc": [np.log10(4.2), 2],
             "log10fX": [-5, 3],
             "powerInd": [1, 1.3, 1.5], #discrete
             "numin": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0], #discrete
             "tau": [0.02, 0.1],
             "log10Fr": [-1, 6]}

paramNames = paramNames_Sims_poly
nDerived = 2*5
nDims = len(paramNames)

print("=== Loading chains ===")
tmp = pd.read_feather("non-public/idr3.feather")
idr3 = anesthetic.samples.MCMCSamples(tmp, weights=tmp.weights)
tmp = pd.read_feather("non-public/prior.feather")
prior = anesthetic.samples.MCMCSamples(tmp, weights=tmp.weights)

#print("=== Save TS and TR data for Punchline plot ===")
#save_npy_dict = {}
#for key in ["log10TS_z8", "log10TR_z8", "log10TS_z10", "log10TR_z10", "weights"]:
#    save_npy_dict[key] = np.array(idr3.weights) if key=="weights" else np.array(idr3[key])
#np.save("non-public/punchline_TS_TR_idr3.npy", save_npy_dict)
#
#save_npy_dict = {}
#for key in ["log10TS_z8", "log10TR_z8", "log10TS_z10", "log10TR_z10", "weights"]:
#    save_npy_dict[key] = np.array(prior.weights) if key=="weights" else np.array(prior[key])
#np.save("non-public/punchline_TS_TR_prior.npy", save_npy_dict)

## Demo code for Jordan
#import numpy as np
#import matplotlib.pyplot as plt
#s = np.load("non-public/punchline_TS_TR_idr3.npy", allow_pickle=True).item()
#p = np.load("non-public/punchline_TS_TR_prior.npy", allow_pickle=True).item()
#plt.hist(p["log10TS_z10"]-p["log10TR_z10"], weights=p["weights"], bins=100, alpha=0.5, density=True)
#plt.hist(s["log10TS_z10"]-s["log10TR_z10"], weights=s["weights"], bins=100, alpha=0.5, density=True)
#plt.hist(p["log10TS_z8"]-p["log10TR_z8"], weights=p["weights"], bins=100, alpha=0.5, density=True)
#plt.hist(s["log10TS_z8"]-s["log10TR_z8"], weights=s["weights"], bins=100, alpha=0.5, density=True)
#plt.show()




print("=== 'Venn diagram style' figure ===")

prior["log10fsfX"]=prior["log10fStar"]+prior["log10fX"]
prior["log10fsfR"]=prior["log10fStar"]+prior["log10Fr"]
idr3["log10fsfX"]=idr3["log10fStar"]+idr3["log10fX"]
idr3["log10fsfR"]=idr3["log10fStar"]+idr3["log10Fr"]

print("Correcting weights to account for non-flat prior, might take a while (~ 10mins)")
venn_corrected_idr3 = anesthetic.samples.MCMCSamples(idr3.copy())
priorpdf = [sum_pdf_2d(venn_corrected_idr3.iloc[i]["log10fsfX"], venn_corrected_idr3.iloc[i]["log10fsfR"], -5,3, -1,6, -5,-0.3) for i in range(len(venn_corrected_idr3))]
print("Done.")
# Make sure to not divide by 0
assert not np.any(np.array(priorpdf)<1e-10)
venn_corrected_idr3.importance_sample(-np.log(priorpdf), inplace=True)
# Formatting
venn_corrected_idr3.limits = {}
prior.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
prior.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"
venn_corrected_idr3.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
venn_corrected_idr3.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"

# Code to make the info-plot
#idr3.label="Old Posterior"
#prior.label="Old prior"
#venn_corrected_idr3.label="New Posterior"
#fig, ax = prior.plot_2d(["log10fsfX", "log10fsfR"], alpha=0.2, types={"lower": "kde"})
#idr3.plot_2d(ax, alpha=1, color="red", facecolor=None)
#venn_corrected_idr3.plot_2d(ax, alpha=0.6)
#fig.legend(*ax.log10fsfX.log10fsfR.get_legend_handles_labels())
#plt.savefig("non-public/info_HERA_IDR3_LWA_Chandra.pdf")
#plt.close()

# Code to make paper-plot
venn_corrected_idr3.label=None
# Make 2d contour plot
fig, ax = venn_corrected_idr3.plot_2d(["log10fsfX", "log10fsfR"], alpha=1, types={"lower": "kde"}, facecolor=None, lw=2, color="k")
fig.set_size_inches(8,6)
# Extract contour lines
lineA = ((((fig.axes[0].collections[0].get_paths()[0]).vertices.T)))
lineB = ((((fig.axes[0].collections[1].get_paths()[0]).vertices.T)))
# Draw the exclusion plots correspondingly, here "region that lies outside of the HERA 1-sigma contour"
ax.log10fsfX.log10fsfR.fill_between(lineB[0], y1=lineB[1], y2=10*np.ones(len(lineB[0])), where=lineB[1]>-1, color=ccb[-1], alpha=1, label="Excluded by HERA (this work)\nat >68% confidence")
# fill_between didn't do the left hand side automatically so do this manually:
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-5,10,100), x1=-10, x2=-3, alpha=1, color=ccb[-1])
# And new the region that lies outside of HERA 2-sigma contour
ax.log10fsfX.log10fsfR.fill_between(lineA[0], y1=lineA[1], y2=10*np.ones(len(lineA[0])), where=lineA[1]>-1, color=ccb[0], alpha=1, label="at >95% confidence")
# Formatting and axis labels
ax.log10fsfX.log10fsfR.set_xlim(-6,2)
ax.log10fsfX.log10fsfR.set_ylim(0,5)
ax.log10fsfX.log10fsfR.set_xticks(np.linspace(-6,2,9), [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^1$", r"$10^2$"])
ax.log10fsfX.log10fsfR.set_yticks(np.linspace(0,5,6), [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
ax.log10fsfX.log10fsfR.set_xlabel(r"$f_{\rm star} \cdot f_X$")
ax.log10fsfX.log10fsfR.set_xlabel(r"$f_{\rm star} \cdot f_{\rm r}}$")
# LWA, with numbers read off from plot -- compare to old HERA paper but slightly more conservative taking the 2-sigma levels from LWA and Chandra
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, hatch="/", color=ccb[1], fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, alpha=0.5, color=ccb[1], label="Exceeds LWA extra-galactic\n radio background today", lw=2)
# Original limit (~1) computed with total 0.5–2 keV CXRB flux 8.15 erg cm−2 s−1, errorbar is 8.15 ± 0.58, scale up since fX proportional to X-Ray Background
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), np.log10(1*1.142), 5, hatch="\\", color=ccb[4], fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), np.log10(1*1.142), 5, alpha=0.5, color=ccb[4], label="Exceeds Chandra unresolved extra-\ngalactic X-ray background today", lw=2)
fig.legend(*ax.log10fsfX.log10fsfR.get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(0,0.1,1,1))
plt.tight_layout()
plt.savefig("non-public/HERA_IDR3_LWA_Chandra.pdf")
plt.show()




print("=== Confidence levels ===")

## Test plot to look at impact of numin (as expected) or powerInd (no impact)
#fig, ax = idr3.plot_2d(paramNames, alpha=0, color=ccb[0])
#for i in range(51):
#    print("plot sample", i)
#    #c = ["red", "blue", "green"][int(i/17)] #for powerInd
#    c = plt.get_cmap("viridis")((i%17)/17.)
#    samples[i].plot_2d(ax, alpha=0.5, types={"diagonal": "kde"}, color=c)
#    # blue = low i = low numin = warmer = weaker constraints
#idr3.plot_2d(ax, alpha=1, color=ccb[0])
#plt.savefig("non-public/info_HERA_IDR3_triangle.pdf")
#plt.show()


# Formatting
idr3.label = None
prior.label = None

# Compute confidence level from samples (see anesthetic GitHub PR)
def fastCL(samples, weights=None, level=0.68, method="iso-probability"):
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples[order])
    weights = np.array(weights[order]/np.sum(weights))
    # Compute inverse cumulative distribution function
    c = np.cumsum(weights)
    i = np.insert(c, 0, 0)
    CDF = np.append(i, 1)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    invcdf = sip.interp1d(CDF, S)
    if method=="iso-probability":
        # Find smallest interval
        distance = lambda a, level=level: invcdf(a+level)-invcdf(a)
        a = sop.minimize_scalar(distance, bounds=(0,1-level), method="Bounded")#, method="SLSQP"
        #print(a, distance(0.16), distance(0.2))
        a =a.x
        interval = np.array([invcdf(a), invcdf(a+level)])
    elif method=="lower-limit":
        # Get value from which we reach the desired level
        interval = invcdf(1-level)
    elif method=="upper-limit":
        # Get value to which we reach the desired level
        interval = invcdf(level)
    else:
        assert False, method
    return interval

# Double checked, should be identical and they are.
yellow_line_Xray = fastCL(idr3.log10fX, weights=idr3.weights)[0]
yellow_line_Xray2 = fastCL(idr3.log10fX, weights=idr3.weights, method="lower-limit")
yellow_line_radio = fastCL(idr3["log10Fr"], weights=idr3.weights)[1]
yellow_line_radio2 = fastCL(idr3["log10Fr"], weights=idr3.weights, method="upper-limit")

print("Exclude at 68% CL:")
print("  F_r > {0:.0f}".format(10**yellow_line_radio))
#print("  F_r > {0:.0f}".format(10**yellow_line_radio2))
print("    Corresponds to L_R = {0:.3e}".format(1e22*(150/150)**-0.7*10**yellow_line_radio))
print("  f_X < {0:.2f}".format(10**yellow_line_Xray))
#print("  f_X < {0:.2f}".format(10**yellow_line_Xray2))
print("    Corresponds to L_X = {0:.3e}".format(3e40*10**yellow_line_Xray))




print("=== Corner plots -- check if look different from before ===")

# Here are 3 blocks for the 3 sizes of the figure:

fig, ax = idr3.plot_2d(["log10fX", "log10Fr"], types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0], "ncompress": 3000})
idr3.plot_2d(ax.loc[['log10fX'],['log10fX']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
idr3.plot_2d(ax.loc[["log10Fr"],["log10Fr"]], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
# Yellow band
y1 = yellow_line_Xray #0
y2 = yellow_line_radio #3
ax['log10fX']["log10Fr"].plot([priorDict_Sims["log10fX"][0],y1],[y2,y2], color=ccb[1], lw=4, alpha=0.4, linestyle='solid')
ax['log10fX']["log10Fr"].plot([y1,y1],[y2,priorDict_Sims["log10Fr"][1]], color=ccb[1], lw=4, alpha=0.4, linestyle='solid', label='Excluded region')
ax['log10fX']['log10fX'].axvline(y1, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax["log10Fr"]["log10Fr"].axvline(y2, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax['log10fX']["log10Fr"].set_xticks([-2,0,2])
# Pandas fun -- make a dataframe of the lower part of corner plot for anesthetic
lower_ax = {}
for key in ax.keys():
    lower_ax[key] = {}
    axk = ax[key]
    for key2 in axk.keys():
        if key != key2:
            lower_ax[key][key2] = ax[key][key2]
        else:
            lower_ax[key][key2] = None
lower_ax = pd.DataFrame(lower_ax)
# KDE lines for lower part
idr3.plot_2d(lower_ax, types={'lower':'fastkde'}, lower_kwargs={"levels":[0.95, 0.68], "linestyles":['dashed', "dotted"], "color":'black', "facecolor":None}, ncompress=3000)
# Decorations
handles, labels = ax["log10fX"]["log10Fr"].get_legend_handles_labels()
fig.set_size_inches(5,4)
ax["log10fX"]["log10fX"].get_yaxis().set_visible(False)
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.7, 0.84))
plt.figure()
img = plt.imshow(np.array([[0,1]]), cmap=LinearSegmentedColormap.from_list("CCB", ['#ffffff', ccb[0]]))
img.set_visible(False)
cax = fig.add_axes([0.6, 0.6, 0.2, 0.02])
plt.colorbar(img, cax=cax, orientation='horizontal', ticks=[0,1])
cax.set_title(r"Posterior PDF value", fontsize=10)
cax.set_xticklabels(["0", "Max"], fontsize=8)

fig.savefig("non-public/HERA_small_triangle.pdf")
plt.show()







fig, ax = idr3.plot_2d(["log10fStar", "log10fX", "log10Fr"], types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0], "ncompress": 3000})
idr3.plot_2d(ax.loc[['log10fX'],['log10fX']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
idr3.plot_2d(ax.loc[['log10fStar'],['log10fStar']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
idr3.plot_2d(ax.loc[["log10Fr"],["log10Fr"]], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
# Yellow band
y1 = yellow_line_Xray #0
y2 = yellow_line_radio #3
ax['log10fX']["log10Fr"].plot([priorDict_Sims["log10fX"][0],y1],[y2,y2], color=ccb[1], lw=4, alpha=0.4, linestyle='solid')
ax['log10fX']["log10Fr"].plot([y1,y1],[y2,priorDict_Sims["log10Fr"][1]], color=ccb[1], lw=4, alpha=0.4, linestyle='solid', label='Excluded region')
ax['log10fX']['log10fX'].axvline(y1, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax["log10Fr"]["log10Fr"].axvline(y2, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax['log10fX']["log10Fr"].set_xticks([-2,0,2])
# Pandas fun -- make a dataframe of the lower part of corner plot for anesthetic
lower_ax = {}
for key in ax.keys():
    lower_ax[key] = {}
    axk = ax[key]
    for key2 in axk.keys():
        if key != key2:
            lower_ax[key][key2] = ax[key][key2]
        else:
            lower_ax[key][key2] = None
lower_ax = pd.DataFrame(lower_ax)
# KDE lines for lower part
idr3.plot_2d(lower_ax, types={'lower':'fastkde'}, lower_kwargs={"levels":[0.95, 0.68], "linestyles":['dashed', "dotted"], "color":'black', "facecolor":None}, ncompress=3000)
# Decorations
handles, labels = ax["log10fX"]["log10Fr"].get_legend_handles_labels()
fig.set_size_inches(5,4)
ax["log10fX"]["log10fX"].get_yaxis().set_visible(False)
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.7, 0.84))
plt.figure()
img = plt.imshow(np.array([[0,1]]), cmap=LinearSegmentedColormap.from_list("CCB", ['#ffffff', ccb[0]]))
img.set_visible(False)
cax = fig.add_axes([0.6, 0.7, 0.2, 0.02])
plt.colorbar(img, cax=cax, orientation='horizontal', ticks=[0,1])
cax.set_title(r"Posterior PDF value", fontsize=10)
cax.set_xticklabels(["0", "Max"], fontsize=8)

fig.savefig("non-public/HERA_medium_triangle.pdf")
plt.show()



fig, ax = idr3.plot_2d(paramNames, types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0], "ncompress": 3000})
idr3.plot_2d(ax.loc[['log10fX'],['log10fX']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
idr3.plot_2d(ax.loc[['log10fStar'],['log10fStar']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
idr3.plot_2d(ax.loc[["log10Fr"],["log10Fr"]], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
# Yellow band
y1 = yellow_line_Xray #0
y2 = yellow_line_radio #3
ax['log10fX']["log10Fr"].plot([priorDict_Sims["log10fX"][0],y1],[y2,y2], color=ccb[1], lw=4, alpha=0.4, linestyle='solid')
ax['log10fX']["log10Fr"].plot([y1,y1],[y2,priorDict_Sims["log10Fr"][1]], color=ccb[1], lw=4, alpha=0.4, linestyle='solid', label='Excluded region')
ax['log10fX']['log10fX'].axvline(y1, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax["log10Fr"]["log10Fr"].axvline(y2, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax['log10fX']["log10Fr"].set_xticks([-2,0,2])
# Pandas fun -- make a dataframe of the lower part of corner plot for anesthetic
lower_ax = {}
for key in ax.keys():
    lower_ax[key] = {}
    axk = ax[key]
    for key2 in axk.keys():
        if key != key2:
            lower_ax[key][key2] = ax[key][key2]
        else:
            lower_ax[key][key2] = None
lower_ax = pd.DataFrame(lower_ax)
# KDE lines for lower part
idr3.plot_2d(lower_ax, types={'lower':'kde'}, lower_kwargs={"levels":[0.95, 0.68], "linestyles":['dashed', "dotted"], "color":'black', "facecolor":None}, ncompress=3000)
# Decorations
handles, labels = ax["log10fX"]["log10Fr"].get_legend_handles_labels()
fig.set_size_inches(10,10)
ax["log10fX"]["log10fX"].get_yaxis().set_visible(False)
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.7, 0.84))
plt.figure()
img = plt.imshow(np.array([[0,1]]), cmap=LinearSegmentedColormap.from_list("CCB", ['#ffffff', ccb[0]]))
img.set_visible(False)
cax = fig.add_axes([0.6, 0.7, 0.2, 0.02])
plt.colorbar(img, cax=cax, orientation='horizontal', ticks=[0,1])
cax.set_title(r"Posterior PDF value", fontsize=10)
cax.set_xticklabels(["0", "Max"], fontsize=8)

fig.savefig("non-public/HERA_large_triangle.pdf")
plt.show()








print("=== Temperature plots ===")

# Confidences
posterior_limit_Trad_over_TK = -fastCL(idr3["log10TK_z8"]-idr3["log10TR_z8"], weights=idr3.weights, method="lower-limit", level=0.95)
prior_limit_Trad_over_TK = -fastCL(prior["log10TK_z8"]-prior["log10TR_z8"], weights=prior.weights, method="lower-limit", level=0.95)
posterior_limit_Trad_over_TS = -fastCL(idr3["log10TS_z8"]-idr3["log10TR_z8"], weights=idr3.weights, method="lower-limit", level=0.95)
prior_limit_Trad_over_TS = -fastCL(prior["log10TS_z8"]-prior["log10TR_z8"], weights=prior.weights, method="lower-limit", level=0.95)
print("Constrain at 95% CL:")
print("  Posterior log10(Trad/TS) < {0:.1f}".format(posterior_limit_Trad_over_TS))
print("  Prior log10(Trad/TS) < {0:.1f}".format(prior_limit_Trad_over_TS))

# TS
import getdist
from getdist import plots
settings = plots.GetDistPlotSettings()
settings.legend_fontsize=12
settings.axes_fontsize=14
settings.axes_labelsize=14
sett = {"smooth_scale_2D": -5}
g = plots.get_single_plotter(width_inch=4, ratio=1, settings=settings)
s = getdist.mcsamples.MCSamples(samples=np.array([prior["log10TS_z8"],prior["log10TR_z8"]]).T, weights=np.array(prior.weights), names=["log10TS_z8","log10TR_z8"], labels=[r'\log_{10} \overline{T}_S', r'\log_{10} \overline{T}_{\rm rad}'], settings=sett)
s2 = getdist.mcsamples.MCSamples(samples=np.array([idr3["log10TS_z8"],idr3["log10TR_z8"]]).T, weights=np.array(idr3.weights), names=["log10TS_z8","log10TR_z8"], labels=[r'\log_{10} \overline{T}_S', r'\log_{10} \overline{T}_{\rm rad}'], settings=sett)
g.plot_2d([s, s2], 'log10TS_z8', 'log10TR_z8', filled=True, colors=["#505050", "#984ea3"], alphas=[1,0.5])
g.add_legend(['Prior', 'Posterior'], legend_loc='lower right')
a2 = g.get_axes()
a2.set_xlim(np.min(prior['log10TS_z8']),np.max(prior['log10TS_z8']))
a2.set_ylim(np.min(prior['log10TR_z8']),np.max(prior['log10TR_z8']))

Tplot = np.linspace(0,6,1000)
a2.plot(Tplot, posterior_limit_Trad_over_TS+Tplot, ls='dashed', color="#984ea3")
plt.savefig("non-public/HERA_TS_TRad.pdf")
plt.show()


# TK
settings = plots.GetDistPlotSettings()
settings.legend_fontsize=11
settings.axes_fontsize=11
settings.axes_labelsize=9
sett = {"smooth_scale_2D": -5}
g = plots.get_single_plotter(width_inch=4, ratio=1, settings=settings)
s = getdist.mcsamples.MCSamples(samples=np.array([prior["log10TK_z8"],prior["log10TR_z8"]]).T, weights=np.array(prior.weights), names=["log10TK_z8","log10TR_z8"], labels=[r'\log_{10} \overline{T}_K', r'\log_{10} \overline{T}_{\rm rad}'], settings=sett)
s2 = getdist.mcsamples.MCSamples(samples=np.array([idr3["log10TK_z8"],idr3["log10TR_z8"]]).T, weights=np.array(idr3.weights), names=["log10TK_z8","log10TR_z8"], labels=[r'\log_{10} \overline{T}_K', r'\log_{10} \overline{T}_{\rm rad}'], settings=sett)
g.plot_2d([s, s2], 'log10TK_z8', 'log10TR_z8', filled=True, colors=["#505050", "#984ea3"], alphas=[1,0.5])
g.add_legend(['Prior', 'Posterior'], legend_loc='lower right')
a2 = g.get_axes()
a2.set_xlim(np.min(prior['log10TK_z8']),np.max(prior['log10TK_z8']))
a2.set_ylim(np.min(prior['log10TR_z8']),np.max(prior['log10TR_z8']))

Tplot = np.linspace(0,6,1000)
a2.plot(Tplot, posterior_limit_Trad_over_TK+Tplot, ls='dashed', color=ccb[0])
a2.plot(Tplot, prior_limit_Trad_over_TK+Tplot, ls='dashed', color=ccb[1])
plt.savefig("non-public/HERA_TK_TRad.pdf")
plt.show()
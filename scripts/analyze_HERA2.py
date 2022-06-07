import numpy as np
from codes.tools import *
import anesthetic
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
ccb = sns.color_palette("colorblind")
from matplotlib.colors import LinearSegmentedColormap
# Plot settings
params = {'legend.fontsize':  11,
          'figure.figsize': (15, 5),
         'axes.labelsize':  11,
         'axes.titlesize': 11,
         'xtick.labelsize': 9,
         'ytick.labelsize': 9}
plt.rcParams.update(params)


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

paramNames = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
nDerived = 2
nDims = len(paramNames)

samples = anesthetic.NestedSamples(root="/tmp/testchains/run_IDR2")
assert np.all(samples.logL0+samples.logL1-samples.logL < 1e-12)






def fastCL(samples, weights=None, level=0.68, method="iso-probability"):
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = samples[order]
    weights = weights[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
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


yellow_line_Xray = fastCL(samples.log10fX, weights=samples.weights, method="lower-limit")
yellow_line_radio = fastCL(samples[paramNames[-1]], weights=samples.weights, method="upper-limit")
#posterior_limit_Trad_over_TK = -fastCL(emcee_results.TK_over_Trad, method="lower-limit", level=0.95)
#prior_limit_Trad_over_TK = -fastCL(prior_distrib.TK_over_Trad, method="lower-limit", level=0.95)

print("Exclude at 68% CL:")
print("  F_r > {0:.0f}".format(10**yellow_line_radio))
print("    Corresponds to L_R = {0:.3e}".format(1e22*(150/150)**-0.7*10**yellow_line_radio))
print("  f_X < {0:.2f}".format(10**yellow_line_Xray))
print("    Corresponds to L_X = {0:.3e}".format(3e40*10**yellow_line_Xray))
#print("Constrain at 95% CL:")
#print("  Posterior log10(Trad/TK) < {0:.1f}".format(posterior_limit_Trad_over_TK))
#print("  Prior log10(Trad/TK) < {0:.1f}".format(prior_limit_Trad_over_TK))












fig, ax = samples.plot_2d(paramNames, types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0]})
samples.plot_2d(ax.loc[['log10fX'],['log10fX']], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])
samples.plot_2d(ax.loc[[paramNames[-1]],[paramNames[-1]]], types={'diagonal':'kde'}, diagonal_kwargs={"edgecolor": ccb[0], "facecolor": 'grey'}, color=ccb[0], levels=[0.68])

# Yellow band
y1 = yellow_line_Xray #0
y2 = yellow_line_radio #3
ax['log10fX'][paramNames[-1]].plot([priorDict_RadLyA["log10fX"][0],y1],[y2,y2], color=ccb[1], lw=4, alpha=0.4, linestyle='solid')
ax['log10fX'][paramNames[-1]].plot([y1,y1],[y2,priorDict_RadLyA["log10Fr"][1]], color=ccb[1], lw=4, alpha=0.4, linestyle='solid', label='Excluded region')
ax['log10fX']['log10fX'].axvline(y1, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax[paramNames[-1]][paramNames[-1]].axvline(y2, ls='solid', color=ccb[1], lw=4, alpha=0.4)
ax['log10fX'][paramNames[-1]].set_xticks([-2,0,2])

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

samples.plot_2d(lower_ax, types={'lower':'fastkde'}, lower_kwargs={"levels":[0.95, 0.68], "linestyles":['dashed', "dotted"], "color":'black', "facecolor":None})

handles, labels = ax['tau'][paramNames[-1]].get_legend_handles_labels()
fig.set_size_inches(10,10)

ax["log10fStar"]["log10fStar"].get_yaxis().set_visible(False)
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.7, 0.84))

plt.figure()
img = plt.imshow(np.array([[0,1]]), cmap=LinearSegmentedColormap.from_list("CCB", ['#ffffff', ccb[0]]))
img.set_visible(False)
cax = fig.add_axes([0.6, 0.75, 0.2, 0.02])
plt.colorbar(img, cax=cax, orientation='horizontal', ticks=[0,1])
cax.set_title(r"Posterior PDF value", fontsize=10)
cax.set_xticklabels(["0", "Max"], fontsize=8)

#plt.tight_layout()
#fig.savefig(figure_path+"triangle_posteriors_"+model_type+".pdf")


plt.show()

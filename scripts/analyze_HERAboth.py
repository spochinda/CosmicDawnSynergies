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

TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TSemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TKemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TRemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)

def TS_TK_Trad_from_emulators(df, z=8):
    emuCols = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
    s = np.shape(df)
    arr = np.empty([s[0],len(emuCols)+1])
    arr[:,0] = z
    arr[:,1:] = df[emuCols]
    TS = TS_emu.predict(arr)
    TK = TK_emu.predict(arr)
    TR = TR_emu.predict(arr)
    return TS, TK, TR

def mergeAnesthetic(prior_samples, weights=None):
    merge_prior = anesthetic.samples.merge_samples_weighted(prior_samples, weights=weights)
    merge_prior = merge_prior.reset_index().set_index("weights", append=True).drop(columns="#")
    merge_prior.index.set_names(["#", "weights"], inplace=True)
    return merge_prior

print("=== Loading chains ===")
# Old samples need to pass `columns` manually, should be automatic in new run
#columns = [*paramNames, *["ll"+str(i) for i in range(10)]]

samples = []; logModelWeights= []
for i in range(51):
    root = "data/chains/idr3_chains_final2/run_IDR3_{:02}".format(i)
    #root = "/data/camHPC/May22/21cm_powerspectra_analysis/chains2/run_{:02}".format(i)
    tmp = anesthetic.anesthetic.samples.NestedSamples(root=root)
    #, columns=columns
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    samples.append(tmp)
    logModelWeights.append(tmp.logZ()+np.log(numinPrior(i)*powerIndPrior))
    print("LogZ(",i,") =", tmp.logZ())
idr3 = mergeAnesthetic(samples, weights=np.exp(logModelWeights))
idr3["log10TS_z8"], idr3["log10TK_z8"], idr3["log10TR_z8"] = np.log10(TS_TK_Trad_from_emulators(idr3, z=8))
idr3["log10TS_z10"], idr3["log10TK_z10"], idr3["log10TR_z10"] = np.log10(TS_TK_Trad_from_emulators(idr3, z=10))

samples2 = []; logModelWeights2= []
for i in range(51):
    root = "data/chains/idr2_chains_final2/run_IDR2_{:02}".format(i)
    tmp = anesthetic.anesthetic.samples.NestedSamples(root=root)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    samples2.append(tmp)
    logModelWeights2.append(tmp.logZ()+np.log(numinPrior(i)*powerIndPrior))
    print("LogZ(",i,") =", tmp.logZ())
idr2 = mergeAnesthetic(samples2, weights=np.exp(logModelWeights2))
idr2["log10TS_z8"], idr2["log10TK_z8"], idr2["log10TR_z8"] = np.log10(TS_TK_Trad_from_emulators(idr2, z=8))
idr2["log10TS_z10"], idr2["log10TK_z10"], idr2["log10TR_z10"] = np.log10(TS_TK_Trad_from_emulators(idr2, z=10))


root = "data/chains/idr2_old_chains_final2/run_IDR2_old"
idr2old = anesthetic.anesthetic.samples.NestedSamples(root=root, columns=[*paramNames_RadLyA, *["ll"+str(i) for i in range(6)]])
idr2old.tex = texDict

root = "data/chains/idr2_orig_chains_final2/run_IDR2_orig"
idr2orig = anesthetic.anesthetic.samples.NestedSamples(root=root)
idr2orig.tex = texDict

root = "data/chains/idr3_old_chains_final2/run_IDR3_old"
idr3old = anesthetic.anesthetic.samples.NestedSamples(root=root, columns=[*paramNames_RadLyA, *["ll"+str(i) for i in range(10)]])
idr3old.tex = texDict
#idr2["log10TS_z8"], idr2["log10TK_z8"], idr2["log10TR_z8"] = np.log10(TS_TK_Trad_from_emulators(idr2, z=8))
#idr2["log10TS_z10"], idr2["log10TK_z10"], idr2["log10TR_z10"] = np.log10(TS_TK_Trad_from_emulators(idr2, z=10))


print("=== Making priors ===")
psamples = []; plogModelWeights= []
for i in range(51):
    print(i)
    priordata = {}
    for key in paramNames_Sims_poly:
        priordata[key] = np.random.uniform(low=priorDict_Sims[key][0], high=priorDict_Sims[key][1], size=10000)
    tmp = anesthetic.samples.MCMCSamples(priordata)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    psamples.append(tmp)
    plogModelWeights.append(np.log(numinPrior(i)*powerIndPrior))
prior = mergeAnesthetic(psamples, weights=np.exp(plogModelWeights))
prior["log10TS_z8"], prior["log10TK_z8"], prior["log10TR_z8"] = np.log10(TS_TK_Trad_from_emulators(prior, z=8))
prior["log10TS_z10"], prior["log10TK_z10"], prior["log10TR_z10"] = np.log10(TS_TK_Trad_from_emulators(prior, z=10))

# Save to files
idr3.reset_index().to_feather("non-public/idr3.feather")
prior.reset_index().to_feather("non-public/prior.feather")


print("=== Save TS and TR data for Punchline plot ===")
save_npy_dict = {}
for key in ["log10TS_z8", "log10TR_z8", "log10TS_z10", "log10TR_z10", "weights"]:
    save_npy_dict[key] = np.array(idr3.weights) if key=="weights" else np.array(idr3[key])
np.save("non-public/punchline_TS_TR_idr3.npy", save_npy_dict)

save_npy_dict = {}
for key in ["log10TS_z8", "log10TR_z8", "log10TS_z10", "log10TR_z10", "weights"]:
    save_npy_dict[key] = np.array(idr2.weights) if key=="weights" else np.array(idr2[key])
np.save("non-public/punchline_TS_TR_idr2.npy", save_npy_dict)

save_npy_dict = {}
for key in ["log10TS_z8", "log10TR_z8", "log10TS_z10", "log10TR_z10", "weights"]:
    save_npy_dict[key] = np.array(prior.weights) if key=="weights" else np.array(prior[key])
np.save("non-public/punchline_TS_TR_prior.npy", save_npy_dict)

## Demo code for Jordan
#import numpy as np
#import matplotlib.pyplot as plt
#s = np.load("non-public/punchline_TS_TR_idr3.npy", allow_pickle=True).item()
#s2 = np.load("non-public/punchline_TS_TR_idr2.npy", allow_pickle=True).item()
#p = np.load("non-public/punchline_TS_TR_prior.npy", allow_pickle=True).item()
#plt.hist(p["log10TS_z10"]-p["log10TR_z10"], weights=p["weights"], bins=100, alpha=0.5, density=True, histtype="step")
#plt.hist(s["log10TS_z10"]-s["log10TR_z10"], weights=s["weights"], bins=100, alpha=0.5, density=True, histtype="step")
#plt.hist(s2["log10TS_z10"]-s2["log10TR_z10"], weights=s2["weights"], bins=100, alpha=0.5, density=True, histtype="step")
#plt.show()




print("=== 'Venn diagram style' figure ===")

prior["log10fsfX"]=prior["log10fStar"]+prior["log10fX"]
prior["log10fsfR"]=prior["log10fStar"]+prior["log10Fr"]
idr3["log10fsfX"]=idr3["log10fStar"]+idr3["log10fX"]
idr3["log10fsfR"]=idr3["log10fStar"]+idr3["log10Fr"]
idr2["log10fsfX"]=idr2["log10fStar"]+idr2["log10fX"]
idr2["log10fsfR"]=idr2["log10fStar"]+idr2["log10Fr"]

print("Correcting weights to account for non-flat prior, might take a while (~ 10mins)")
venn_corrected_idr3 = anesthetic.samples.MCMCSamples(idr3.copy())
venn_corrected_idr2 = anesthetic.samples.MCMCSamples(idr2.copy())
priorpdf = [sum_pdf_2d(venn_corrected_idr3.iloc[i]["log10fsfX"], venn_corrected_idr3.iloc[i]["log10fsfR"], -5,3, -1,6, -5,-0.3) for i in range(len(venn_corrected_idr3))]
priorpdf2 = [sum_pdf_2d(venn_corrected_idr2.iloc[i]["log10fsfX"], venn_corrected_idr2.iloc[i]["log10fsfR"], -5,3, -1,6, -5,-0.3) for i in range(len(venn_corrected_idr2))]
print("Done.")
# Make sure to not divide by 0
assert not np.any(np.array(priorpdf)<1e-10)
assert not np.any(np.array(priorpdf2)<1e-10)
venn_corrected_idr3.importance_sample(-np.log(priorpdf), inplace=True)
venn_corrected_idr2.importance_sample(-np.log(priorpdf2), inplace=True)
# Formatting
prior.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
prior.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"
venn_corrected_idr3.limits = {}
venn_corrected_idr2.limits = {}
venn_corrected_idr3.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
venn_corrected_idr3.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"
venn_corrected_idr2.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
venn_corrected_idr2.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"

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
venn_corrected_idr2.label=None
# Make 2d contour plot
fig, ax = venn_corrected_idr3.plot_2d(["log10fsfX", "log10fsfR"], alpha=1, types={"lower": "kde"}, facecolor=None, lw=2, color="k")
fig.set_size_inches(8,6)
# Extract contour lines
lineA = ((((fig.axes[0].collections[0].get_paths()[0]).vertices.T)))
lineB = ((((fig.axes[0].collections[1].get_paths()[0]).vertices.T)))
# Draw the exclusion plots correspondingly, here "region that lies outside of the HERA 1-sigma contour"
ax.log10fsfX.log10fsfR.fill_between(lineB[0], y1=lineB[1], y2=10*np.ones(len(lineB[0])), where=lineB[1]>-1, color="pink", alpha=1, label="Excluded by HERA (this work)\nat >68% confidence")
# fill_between didn't do the left hand side automatically so do this manually:
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-5,10,100), x1=-10, x2=-3, alpha=1, color="pink")
# And new the region that lies outside of HERA 2-sigma contour
ax.log10fsfX.log10fsfR.fill_between(lineA[0], y1=lineA[1], y2=10*np.ones(len(lineA[0])), where=lineA[1]>-1, color="purple", alpha=1, label="at >95% confidence")
# IDR2
venn_corrected_idr2.plot_2d(ax, alpha=1, types={"lower": "kde"}, facecolor="none", lw=2, color="red", label="IDR2")
# Formatting and axis labels
ax.log10fsfX.log10fsfR.set_xlim(-6,2)
ax.log10fsfX.log10fsfR.set_ylim(0,5)
ax.log10fsfX.log10fsfR.set_xticks(np.linspace(-6,2,9), [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^1$", r"$10^2$"])
ax.log10fsfX.log10fsfR.set_yticks(np.linspace(0,5,6), [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
ax.log10fsfX.log10fsfR.set_xlabel(r"$f_{\rm star} \cdot f_X$")
ax.log10fsfX.log10fsfR.set_xlabel(r"$f_{\rm star} \cdot f_{\rm r}}$")
# LWA, with numbers read off from plot -- compare to old HERA paper but slightly more conservative taking the 2-sigma levels from LWA and Chandra
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, hatch="/", color="orange", fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, alpha=0.5, color="orange", label="Exceeds LWA extra-galactic\n radio background today", lw=2)
# Original limit (~1) computed with total 0.5–2 keV CXRB flux 8.15 erg cm−2 s−1, errorbar is 8.15 ± 0.58, scale up since fX proportional to X-Ray Background
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), np.log10(1*1.142), 5, hatch="\\", color="blue", fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), np.log10(1*1.142), 5, alpha=0.5, color="blue", label="Exceeds Chandra unresolved extra-\ngalactic X-ray background today", lw=2)
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

# Double checked, should be identical and they are.
yellow_line_Xray = fastCL(idr3.log10fX, weights=idr3.weights)[0]
yellow_line_Xray2 = fastCL(idr3.log10fX, weights=idr3.weights, method="lower-limit")
yellow_line_radio = fastCL(idr3["log10Fr"], weights=idr3.weights)[1]
yellow_line_radio2 = fastCL(idr3["log10Fr"], weights=idr3.weights, method="upper-limit")

print("f_X >")
print("      {0:.2f}".format(10**fastCL(idr2.log10fX, weights=idr2.weights)[0]), "(full idr2 new models)")
print("      {0:.2f}".format(10**fastCL(idr2old.log10fX, weights=idr2old.weights)[0]), "(full idr2 old models)")
print("      {0:.2f}".format(10**fastCL(idr2orig.log10fX, weights=idr2orig.weights)[0]), "(orig idr2 old models)")
print("      {0:.2f}".format(10**fastCL(idr3.log10fX, weights=idr3.weights)[0]), "(full idr3 new models)")
print("      {0:.2f}".format(10**fastCL(idr3old.log10fX, weights=idr3old.weights)[0]), "(full idr3 old models)")

print("f_r <")
print("      {0:.2f}".format(10**fastCL(idr2.log10Fr, weights=idr2.weights)[1]), "(full idr2 new models)")
print("      {0:.2f}".format(10**fastCL(idr2old.log10Fr, weights=idr2old.weights)[1]), "(full idr2 old models)")
print("      {0:.2f}".format(10**fastCL(idr2orig.log10Fr, weights=idr2orig.weights)[1]), "(orig idr2 old models)")
print("      {0:.2f}".format(10**fastCL(idr3.log10Fr, weights=idr3.weights)[1]), "(full idr3 new models)")
print("      {0:.2f}".format(10**fastCL(idr3old.log10Fr, weights=idr3old.weights)[1]), "(full idr3 old models)")

print()
print(10**fastCL(idr3.log10fX, weights=idr3.weights)[0])

print(10**fastCL(idr2.log10Fr, weights=idr2.weights)[1])
print(10**fastCL(idr3.log10Fr, weights=idr3.weights)[1])

print("Exclude at 68% CL:")
print("  F_r > {0:.0f}".format(10**yellow_line_radio))
#print("  F_r > {0:.0f}".format(10**yellow_line_radio2))
print("    Corresponds to L_R = {0:.3e}".format(1e22*(150/150)**-0.7*10**yellow_line_radio))
print("  f_X < {0:.2f}".format(10**yellow_line_Xray))
#print("  f_X < {0:.2f}".format(10**yellow_line_Xray2))
print("    Corresponds to L_X = {0:.3e}".format(3e40*10**yellow_line_Xray))




print("=== Corner plots -- check if look different from before ===")

# Here are 3 blocks for the 3 sizes of the figure:

fig, ax = idr3.plot_2d(["log10fX", "log10Fr"], types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0]})
idr2.plot_2d(ax, types={'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': "red"})
idr3old.plot_2d(ax, types={'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': "cyan"})
idr2old.plot_2d(ax, types={'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': "orange"})
idr2orig.plot_2d(ax, types={'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': "yellow"})
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
idr2.plot_2d(lower_ax, types={'lower':'fastkde'}, lower_kwargs={"levels":[0.95, 0.68], "linestyles":['dashed', "dotted"], "color":'red', "facecolor":None}, ncompress=3000)
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







fig, ax = idr3.plot_2d(["log10fStar", "log10fX", "log10Fr"], types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0]})
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



fig, ax = idr3.plot_2d(paramNames, types={'lower':'hist', 'diagonal':'kde'}, lower_kwargs={"bins": 20, 'color': ccb[0], "vmin":0, "zorder":-10, "rasterized":True}, diagonal_kwargs={'edgecolor': ccb[0]})
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
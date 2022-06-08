from codes.emulator_poweremu import *
from codes.likelihood_hera import *
from codes.loader_21cmSim import *
from codes.plotlibs import *
from codes.tools import *
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

#def loglike_true(data_row, debug=False, **kwargs):
#    model = powerspec_of_z_k_hovercMpc(data_row, **kwargs)
#    wrapper = lambda z,k: model(z,k)[:,0]
#    return like_idr3.loglike(wrapper, debug=debug)
#logL_true = [loglike_true(Pk_nodrops[i]) for i in range(len(Pk_nodrops))]
#lwa_allowed = lwa_allowed_z8["allowed"]
#logL_true, [PT, lwa_allowed] = remove_powerspectra_nans(logL_true, [PT, lwa_allowed])


TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TS_emu_Sims_prelim_v2_31.05.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_Sims_prelim_v2_31.05.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False, offset=1e-3)
TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TR_emu_Sims_prelim_v2_31.05.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False, offset=1e-3)
def TS_TK_Trad_from_emus(df):
    emuCols = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
    TS = TS_emu.predict(df[emuCols])
    TK = TK_emu.predict(df[emuCols])
    TR = TR_emu.predict(df[emuCols])
    return TS, TK, TR

print("=== Loading chains ===")
# Old samples need to pass cols manually
cols=[*paramNames, *["ll"+str(i) for i in range(10)]]

samples = []; logModelWeights= []
for i in range(51):
    tmp = anesthetic.anesthetic.samples.NestedSamples(root="/data/camHPC/May22/21cm_powerspectra_analysis/chains2/run_{:02}".format(i), columns=cols)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    tmp["log10TS"], tmp["log10TK"], tmp["log10TR"] = np.nan_to_num(np.log10(TS_TK_Trad_from_emus(tmp)), nan=-3)
    samples.append(tmp)
    logModelWeights.append(tmp.logZ()+np.log(numinPrior(i)*powerIndPrior))
    print("LogZ(",i,") =", tmp.logZ())
idr3 = anesthetic.samples.merge_samples_weighted(samples, weights=np.exp(logModelWeights))

print("=== 'venn' figure ===")
priordata = {}
for key in ["log10fStar", "log10fX", "log10Fr"]:
    priordata[key] = np.random.uniform(low=priorDict_Sims[key][0], high=priorDict_Sims[key][1], size=10000)
prior = anesthetic.samples.MCMCSamples(priordata)
prior["log10fsfX"]=prior["log10fStar"]+prior["log10fX"]
prior["log10fsfR"]=prior["log10fStar"]+prior["log10Fr"]

idr3["log10fsfX"]=idr3["log10fStar"]+idr3["log10fX"]
idr3["log10fsfR"]=idr3["log10fStar"]+idr3["log10Fr"]

venn_corrected_idr3 = anesthetic.samples.MCMCSamples(idr3.copy())
print("Computing prior PDF values -- might take a while ~ 10mins")
priorpdf = [sum_pdf_2d(venn_corrected_idr3.iloc[i]["log10fsfX"], venn_corrected_idr3.iloc[i]["log10fsfR"], -5,3, -1,6, -5,-0.3) for i in range(len(venn_corrected_idr3))]
assert not np.any(np.array(priorpdf)<1e-10)
venn_corrected_idr3.importance_sample(-np.log(priorpdf), inplace=True)
venn_corrected_idr3.limits = {}
idr3.label="Old Posterior"
prior.label="Old prior"
prior.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
prior.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"
venn_corrected_idr3.tex["log10fsfX"] = r"$\log_{10} f_{\rm star}\cdot f_X$"
venn_corrected_idr3.tex["log10fsfR"] = r"$\log_{10} f_{\rm star}\cdot f_r$"

venn_corrected_idr3.label="New Posterior"
fig, ax = prior.plot_2d(["log10fsfX", "log10fsfR"], alpha=0.2, types={"lower": "kde"})
idr3.plot_2d(ax, alpha=1, color="red", facecolor=None)
venn_corrected_idr3.plot_2d(ax, alpha=0.6)
fig.legend(*ax.log10fsfX.log10fsfR.get_legend_handles_labels())
plt.savefig("non-public/info_HERA_IDR3_LWA_Chandra.pdf")
plt.show()

venn_corrected_idr3.label=None
fig, ax = venn_corrected_idr3.plot_2d(["log10fsfX", "log10fsfR"], alpha=1, types={"lower": "kde"}, facecolor=None, lw=2, color="k")
fig.set_size_inches(8,6)
sigma1 = ((((fig.axes[0].collections[0].get_paths()[0]).vertices.T)))
sigma2 = ((((fig.axes[0].collections[1].get_paths()[0]).vertices.T)))
ax.log10fsfX.log10fsfR.fill_between(sigma2[0], y1=sigma2[1], y2=10*np.ones(len(sigma2[0])), where=sigma2[1]>-1, color=ccb[-1], alpha=1, label="Outside of HERA 1sigma contours\n aka 68% excluded")
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-5,10,100), x1=-10, x2=-3, alpha=1, color=ccb[-1])
ax.log10fsfX.log10fsfR.fill_between(sigma1[0], y1=sigma1[1], y2=10*np.ones(len(sigma1[0])), where=sigma1[1]>-1, color=ccb[0], alpha=1, label="Outside of HERA 2sigma contours\n aka 95% excluded")
ax.log10fsfX.log10fsfR.set_xlim(-6,2)
ax.log10fsfX.log10fsfR.set_ylim(0,5)
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, hatch="/", color=ccb[1], fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_between(np.linspace(-10,10, 10), np.log10(2e3), 7, alpha=0.5, color=ccb[1], label="LWA excluded 2 sigma", lw=2)
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), 0, 5, hatch="\\", color=ccb[4], fc=(1,1,1,0), lw=2)
ax.log10fsfX.log10fsfR.fill_betweenx(np.linspace(-10,10, 10), 0, 5, alpha=0.5, color=ccb[4], label="Chandra excluded", lw=2)
fig.legend(*ax.log10fsfX.log10fsfR.get_legend_handles_labels(), loc="center right")
plt.savefig("non-public/HERA_IDR3_LWA_Chandra.pdf")
plt.show()


print("=== 'triangle' figure ===")

# A few tests:
fig, ax = idr3.plot_2d(paramNames, alpha=0, color=ccb[0])
for i in range(51):
    print("plot sample", i)
    #c = ["red", "blue", "green"][int(i/17)]
    c = plt.get_cmap("viridis")((i%17)/17.)
    samples[i].plot_2d(ax, alpha=0.5, types={"diagonal": "kde"}, color=c)
    # blue = low i = low numin = warmer = weaker constraints
idr3.plot_2d(ax, alpha=1, color=ccb[0])
plt.savefig("non-public/info_HERA_IDR3_triangle.pdf")
plt.show()
















assert False
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




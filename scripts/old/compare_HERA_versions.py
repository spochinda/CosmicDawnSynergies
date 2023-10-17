from emulator_poweremu.poweremu import *
from likelihood_hera.hera import *



# Real model
#P = poweremu(loadfile="/home/stefan/powerspectra_analysis/data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
#PRadLyA = poweremu(loadfile="/home/stefan/powerspectra_analysis/data/trained_emulators_poweremu/Pk_emu_m_RadLyA_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
paramNames = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
paramTex = [r"$R_{\rm mfp}$", r"$\log_{10} f_{\rm star}$", r"$V_c$", r"$\log_{10} f_{\rm X}$", r"$\tau$", r"$\log_{10} f_{\rm r}$"]
texDict = dict(zip(paramNames, paramTex))
cols=[*paramNames, *["loglike_band"+str(int(1+i/5))+"_field"+str(i%5) for i in range(10)]]
cols_idr2a=[*paramNames, *["loglike_band"+str(int(1+i/1))+"_field"+str(i%1) for i in range(2)]]
cols_idr2b=[*paramNames, *["loglike_band"+str(int(1+i/3))+"_field"+str(i%3) for i in range(6)]]
def powerInd_and_numin_from_index(index):
    powerInds = [1, 1.3, 1.5]
    numins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]
    powerInd = powerInds[int(index/len(numins))]
    numin = numins[index % len(numins)]
    return powerInd, numin
samples = []; logZs= []
for i in range(51):
    tmp = anesthetic.NestedSamples(root="chains2/run_{:02}".format(i), columns=cols)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    samples.append(tmp)
    logZs.append(tmp.logZ())
    print("LogZ(",i,") =", tmp.logZ())
merge = anesthetic.samples.merge_samples_weighted(samples, weights=np.exp(logZs)).reset_index()
merge.weights = merge["weights"]


samples_IDR2 = []; logZs_IDR2= []
for i in range(51):
    tmp = anesthetic.NestedSamples(root="chains_idr2/idr2_run_{:02}".format(i), columns=cols_idr2a)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    #tmp["log10TS"], tmp["log10TK"], tmp["log10TR"] = np.nan_to_num(np.log10(TS_TK_Trad_from_emus(tmp)), nan=-3)
    samples_IDR2.append(tmp)
    logZs_IDR2.append(tmp.logZ())
    print("LogZ(",i,") =", tmp.logZ())
merge_IDR2 = anesthetic.samples.merge_samples_weighted(samples_IDR2, weights=np.exp(logZs_IDR2)).reset_index()
merge_IDR2.weights = merge_IDR2["weights"]
real_params_IDR2 = merge_IDR2[model_params]

samples_IDR2b = []; logZs_IDR2b= []
for i in range(51):
    tmp = anesthetic.NestedSamples(root="chains_idr2/idr2b_run_{:02}".format(i), columns=cols_idr2b)
    tmp.tex = texDict
    tmp["powerInd"], tmp["numin"] = powerInd_and_numin_from_index(i)
    #tmp["log10TS"], tmp["log10TK"], tmp["log10TR"] = np.nan_to_num(np.log10(TS_TK_Trad_from_emus(tmp)), nan=-3)
    samples_IDR2b.append(tmp)
    logZs_IDR2b.append(tmp.logZ())
    print("LogZ(",i,") =", tmp.logZ())
merge_IDR2b = anesthetic.samples.merge_samples_weighted(samples_IDR2b, weights=np.exp(logZs_IDR2b)).reset_index()
merge_IDR2b.weights = merge_IDR2b["weights"]
real_params_IDR2b = merge_IDR2b[model_params]


# Original HERA samples
orig_hera_data=np.load("/data/highz/SHdata/HERA_nov_v2/chains/Fr/emcee_flatchain.npy").T[::211]
orig_hera = anesthetic.samples.MCMCSamples(data=orig_hera_data, columns=['log10fStar', 'log10Vc', 'log10fX', 'tau', 'log10Fr'], tex=texDict, label='HERA original samples')
tmp = orig_hera.weights
tmp[orig_hera.tau>0.077]=0
orig_hera.weights = tmp
orig_hera.limits["tau"] = [orig_hera.limits["tau"][0], 0.077]

def orig_reproduce_model(karr, p, z=8):
    rsd = 1
    par0 = np.array([z, np.NaN, *p, rsd])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return PRadLyA.predict(params)

model_params = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
def real_model(karr, p, z=8):
    par0 = np.array([z, np.NaN, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return P.predict(params)
def real_model_RadLyA(karr, p, z=8):
    rsd = 1
    par0 = np.array([z, np.NaN, *p[np.array([1,2,3,-2,-1])], rsd])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return PRadLyA.predict(params)
real_params = merge[model_params]

def loglikelihood(A):
    m = lambda z,karr,A=A: toy_model(karr, A)
    return like_idr3.loglike(m)


Aprior = np.geomspace(1e0, 1e7, 500000)
logL = [loglikelihood(a) for a in Aprior]

from fgivenx import plot_contours, samples_from_getdist_chains


fig, ax = plt.subplots()
ax.set_yscale("log")
prior = plot_contours(toy_model, np.linspace(0.1,3,3), np.array([Aprior]).T, ax=ax, colors=plt.cm.YlOrBr_r, cache="/tmp/fgivenx/a", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
post = plot_contours(toy_model, np.linspace(0.1,3,3), np.array([Aprior]).T, weights=np.exp(logL), ax=ax, colors=plt.cm.Purples_r, cache="/tmp/fgivenx/b", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
cbar = plt.colorbar(post,ticks=[0,1,2,3,4,5], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
cbar = plt.colorbar(prior,ticks=[0,1,2,3,4,5], label="Prior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
like_idr3.plot_data(axes=[ax], color="black")
plt.show()

fig, ax = plt.subplots()
ax.set_yscale("log")
post = plot_contours(toy_model, np.linspace(0.1,3,3), np.array([Aprior]).T, weights=np.exp(logL), ax=ax, colors=plt.cm.Purples_r, cache="/tmp/fgivenx/b", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
cbar = plt.colorbar(post,ticks=[0,1,2,3,4,5], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
like_idr3.plot_data_violin(axes=[ax])
plt.show()




P = poweremu(loadfile="/home/stefan/powerspectra_analysis/data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
#PRadLyA = poweremu(loadfile="/home/stefan/powerspectra_analysis/data/trained_emulators_poweremu/Pk_emu_m_RadLyA_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
fig, ax = plt.subplots()
ax.set_yscale("log")
#post = plot_contours(real_model, np.linspace(0.1,3,10), np.array(real_params), weights=real_params.weights, ax=ax, colors=plt.cm.YlOrBr_r, cache="/tmp/fgivenx/r", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
#post = plot_contours(real_model_RadLyA, np.linspace(0.1,3,10), np.array(real_params), weights=real_params.weights, ax=ax, colors=plt.cm.YlOrBr_r, cache="/tmp/fgivenx/rr", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
post = plot_contours(real_model, np.linspace(0.1,3,10), np.array(real_params_IDR2), weights=real_params_IDR2.weights, ax=ax, colors=plt.cm.Reds_r, cache="/tmp/fgivenx/rrr", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
post = plot_contours(real_model, np.linspace(0.1,3,10), np.array(real_params_IDR2b), weights=real_params_IDR2b.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx/rrrr", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
 #= plot_contours(toy_model, np.linspace(0.1,3,3), np.array([Aprior]).T, weights=np.exp(logL), ax=ax, colors=plt.cm.Purples_r, cache="/tmp/fgivenx/b", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
cbar = plt.colorbar(post,ticks=[0,1,2,3,4,5], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
#cbar = plt.colorbar(prior,ticks=[0,1,2,3,4,5], label="Prior")
#cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
#like_idr3.plot_data(axes=[ax], color="black")
plt.ylim(1,1e4)
plt.xlim(0.1,1.5)
plt.show()

P = poweremu(loadfile="emulator_poweremu/trained_emulators/Sims_data_v03_150it_23.02.2022.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
PRadLyA = poweremu(loadfile="/home/stefan/powerspectra_analysis/data/trained_emulators_poweremu/Pk_emu_m_RadLyA_adaptive.pkl", tol=0, n_iter_no_change=99999, preprocesss_log_x=False)
# Old emu
fig, ax = plt.subplots()
ax.set_yscale("log")
post = plot_contours(real_model, np.linspace(0.1,3,10), np.array(real_params), weights=real_params.weights, ax=ax, colors=plt.cm.YlOrBr_r, cache="/tmp/fgivenx/r", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
post = plot_contours(orig_reproduce_model, np.linspace(0.1,1.25,7), np.array(orig_hera), weights=orig_hera.weights, ax=ax, colors=plt.cm.Blues_r, cache="/tmp/fgivenx2/rr5", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
post = plot_contours(real_model, np.linspace(0.1,1.2,7), np.array(real_params_IDR2), weights=real_params_IDR2.weights, ax=ax, colors=plt.cm.Reds_r, cache="/tmp/fgivenx2/rrr", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
#post = plot_contours(real_model, np.linspace(0.1,1.2,7), np.array(real_params_IDR2b), weights=real_params_IDR2b.weights, ax=ax, colors=plt.cm.Reds_r, cache="/tmp/fgivenx2/rrrr", fineness=1, contour_color_levels=[0,1,2,3], lines=False, alpha=0.5)
 #= plot_contours(toy_model, np.linspace(0.1,3,3), np.array([Aprior]).T, weights=np.exp(logL), ax=ax, colors=plt.cm.Purples_r, cache="/tmp/fgivenx/b", fineness=1, contour_color_levels=[0,1,2,3,4,5], lines=False)
cbar = plt.colorbar(post,ticks=[0,1,2,3,4,5], label="Posterior")
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$'])
plt.ylim(1,1e4)
plt.xlim(0.1,1.3)
plt.show()

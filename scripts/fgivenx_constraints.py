import os
import numpy as np
import matplotlib.pyplot as plt
import codes.itamar.radio_cutoff_calc as rad
from codes.emulator_poweremu import *
from codes.loader_21cmSim import *
from codes.likelihood_hera import *
import anesthetic
from tensorflow import keras
from globalemu.eval import evaluate
from fgivenx import plot_contours
from scipy.constants import parsec, physical_constants
import seaborn as sns
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
})


ccb = sns.color_palette("colorblind",as_cmap=True)

#ccb = [ccb[0],ccb[1], ccb[2], ccb[-2], ccb[4]]


files = [#"scripts/non-public/1HERA_1Chandra_1LWA_1SARAS_globalemufinal_nlive_1000/run_h1c_idr2",
         #"scripts/non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2",
         #"scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemufinal_nlive_10000/run_no_idr",
         #"scripts/non-public/0HERA_0Chandra_1LWA_0SARAS_globalemuz6_nlive_10000/run_no_idr",
         #"scripts/non-public/0HERA_0Chandra_0LWA_1SARAS_globalemu_nlive_1000/run_no_idr",
         
         "scripts/non-public/1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14test2idr3_nlive_1000/run",#"scripts/non-public/1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14_nlive_1000/run_h1c_idr2",
         "scripts/non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu315emu14testidr3_nlive_10000/run_idr3",#"scripts/non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_h1c_idr2",
         "scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
         "scripts/non-public/0HERA_0Chandra_1LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
         "scripts/non-public/0HERA_0Chandra_0LWA_1SARAS_globalemu315emu14_nlive_1000/run_no_idr",
        ]

paramNames = [
    "log10fstarII",
    "log10fstarIII",
    "log10Vc",
    "log10fX",
    "alpha", "nu_0",#
    "tau",
    "log10fradio",
    "pop",#
    #"a0", "a1", "a2", "a3", "a4", "a5", "a6", "std21",
]
"""
######## DATA ########

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
axes=[axes,axes,axes]
fontsize=24


h=0.6704
#H1C IDR2

#selections = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}
#data = {}
#for band, selection in selections.items():
#    data[band] = {}
#    for field, sel in selection.items():
#        data[band][field] = extract_data(
#            datapath='data/observations_H1C_IDR2/pspec_h1c_idr2_field{}.h5',
#            band=int(band),
#            field=field,
#            kstart=sel["kstart"],
#            kstart_modulo=0.192,
#            decimation_factor=2,
#            set_negative_to_zero=True
#            )

selection = [{"1": {"D": {"kstart":0.36}}}, {"2": {"C": {"kstart":0.34}}}]
dpath = [
    'data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
    'data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5']
like_hera = likelihood(
    datapath=dpath,
    decimation_factor=2,
    selections=selection
)
data = like_hera.data

c = ["deeppink", "blue"]
fields = ["D", "C"]
for i,band in enumerate(data.keys()):    
    k_data_mask = (data[band][fields[i]]["k_data"] >= 8.5e-2/h) & (data[band][fields[i]]["k_data"] <= 1/h)
    axes[0].errorbar(x=data[band][fields[i]]["k_data"][k_data_mask], y=data[band][fields[i]]["dsq"][k_data_mask], yerr=data[band][fields[i]]["std"][k_data_mask],
                     elinewidth=4, capsize=5, capthick=3,
                    #alpha=1,
                    ls="None",
                    color=c[i],
                    label="Band {0} ($z{1}{2:.2f}$) Field {3}".format(band, r"\approx", data[band][fields[i]]["z"],fields[i])
                    )
axes[0].set_yscale("log")
axes[0].set_ylim([1e0,1e8])
axes[0].set_xlabel(r"Wavevector $k\ [h\ \mathrm{Mpc^{-1}}]$",fontsize=fontsize)
axes[0].set_ylabel(r"Power Spectrum $\Delta_{21}^2 [\mathrm{mK^2}]$",fontsize=fontsize)
axes[0].legend(loc="upper left", fontsize=fontsize)

axes[0].tick_params(axis='both', which='major', labelsize=fontsize-6)
#plt.show()
plt.savefig("images/data_hera.pdf",bbox_inches="tight")


#LWA
[nu_obs, T_obs, dT_obs] = np.load('codes/itamar/LWA1_with_err.npy')
yerr = np.array([np.log10(T_obs+dT_obs) - np.log10(T_obs), np.log10(T_obs) - np.log10(T_obs-dT_obs)])
axes[1].axhline(np.log10(2.73),ls="dashed",label=r"$T_\mathrm{CMB}$",color="k")
axes[1].errorbar(x=np.log10(nu_obs/1e9), y=np.log10(T_obs), yerr=yerr,
                 lw=2, elinewidth=3, capsize=4, capthick=3,
                 color="k",
                #alpha=1,
                ls="solid",
                label="Present-day radio\nbackground temperature"#"LWA1/ARCADE2"
                )
axes[1].set_ylabel(r"$\log_{10} T_\mathrm{Radio}(z=0) \ [K]$",fontsize=fontsize)
axes[1].set_xlabel(r"$\log_{10} \nu [\mathrm{GHz}]$",fontsize=fontsize)
axes[1].legend(fontsize=fontsize)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize-6)

#plt.show()
plt.savefig("images/data_lwa.pdf",bbox_inches="tight")


#SARAS
freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt("data/SARAS3/SARAS_3_averaged_spectrum.txt").T
axes[2].plot((1420/freq - 1)[::-1], (T_SARAS-fg_fit)*1000, color="k", label="SARAS 3 Residuals") 
axes[2].set_xlabel(r"Redshift $z$",fontsize=fontsize)
axes[2].set_ylabel(r"Global Signal $T_{21}\ [\mathrm{mK}]$",fontsize=fontsize)
axes[2].legend(fontsize=fontsize)

axes[2].tick_params(axis='both', which='major', labelsize=fontsize-6)
#[ax.tick_params(axis='both', which='major', labelsize=fontsize) for ax in axes]
#fig.set_size_inches(8, 6)
#plt.tight_layout() #subplots_adjust(wspace=0.3)

#plt.savefig("images/data0.png", bbox_inches="tight")
#plt.savefig("images/data_saras3.pdf", bbox_inches="tight")
#plt.show()

#Chandra
from scipy.constants import parsec, physical_constants

eV_toHz = physical_constants['electron volt-hertz relationship'][0]
keV_toHz = eV_toHz*1e3
sr_todeg2 = (180/np.pi)**2
Mpc_tom = 1e6 * parsec
Mpc_tocm = Mpc_tom * 1e2
cm_toMpc = 1/Mpc_tocm

X_limits = np.array([ #nu_min, nu_max, mean, std
    #[0.5, 2, 8.15*1e-12, 0.58*1e-12], #Lehmer+2012 
    [1, 2, 1.04*1e-12, 0.14*1e-12], #Hickox & Markevitch (2006)
    [2, 8, 3.4*1e-12, 1.7*1e-12], #Hickox & Markevitch (2006)
    [8, 24, 6.773*1e-8/sr_todeg2, 0.348*1e-8/sr_todeg2], #Harrison et al. (2016)
    [20, 50, 6.205*1e-8/sr_todeg2, 0.17*1e-8/sr_todeg2], #Harrison et al. (2016)
])

table = [
    [r"$(1.04\pm 0.14)\times 10^{-12} \mathrm{ergs\ cm^{-2}s^{-1}deg^{-2}}$"],
    [r"$(3.4\pm 1.7)\times 10^{-12} \mathrm{ergs\ cm^{-2}s^{-1}deg^{-2}}$"],
    [r"$(1.832\pm 0.042)\times 10^{-11} \mathrm{ergs\ cm^{-2}s^{-1}deg^{-2}}$"],
    [r"$(2.00\pm 0.08)\times 10^{-11} \mathrm{ergs\ cm^{-2}s^{-1}deg^{-2}}$"],
    ]
rownames = [
    r"$S(1-2\ \mathrm{keV})$",
    r"$S(2-8\ \mathrm{keV})$",
    r"$S(8-24\ \mathrm{keV})$",
    r"$S(20-50\ \mathrm{keV})$"
    ]
#axes[3].axis("off")
#axes[3].table(cellText=table, rowLabels=rownames, cellLoc='left',loc="center",)
#plt.savefig("images/data.png", bbox_inches="tight")
#plt.show()


######## GLOBAL SIGNAL ########

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()




freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt("data/SARAS3/SARAS_3_averaged_spectrum.txt").T
z = np.linspace(6,28,100)#(1420/freq-1)[::-1][::10]
model = keras.models.load_model('data/globalemu/emulator14/results/model.h5', compile=False)
predictor = evaluate(base_dir='data/globalemu/emulator14/results/', model=model, z=z, gc=False, logs=[]) #0,1,2,3,7], )

def signal_emu(z, p):
    signal = predictor(p[:9])[0] #to K
    return signal


fig, axes = plt.subplots(nrows=1, ncols=len(files), sharey="row", figsize=(24,4))

#axes[0].set_title("Prior")
axes[0].set_ylabel(r"$T_{21}\ [\mathrm{mK}]$")
#axes[0].set_xlabel(r"$z$")

axes[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2, 
            ls="dotted",#if "1SARAS" in fn else 0,
            )

#axes[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", alpha=0.2,) 

for i,(ax,file) in enumerate(zip(axes, files)):
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])

    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()
    

    cbar1 = plot_contours(f=signal_emu, x=z, samples=p_prior, ax=ax, weights=weights_prior,
                    lines=False,
                    colors=plt.cm.Greys_r, 
                    #**{"contour_line_levels": [2,]}
                    cache="data/fgivenx/prior_T21_final",
                    alpha=0.8
                    )

    sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    cbar2 = plot_contours(f=signal_emu, x=z, samples=p_sample, ax=ax, weights=weights_sample,
                        colors=c, lines=True,
                        cache="data/fgivenx/"+fn+"_T21_final"
                        #**{"contour_line_levels": [2,]}
                        )
    
    ax.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2 if "SARAS" in fn else 0.2,
            ls="solid" if "SARAS" in fn else "dotted",
            ) 
    ax.set_xlabel(r"$z$")
    ax.set_title(fn)



plt.subplots_adjust(wspace=0, hspace=0)
#plt.savefig("images/global_signal_fgivenx_final.pdf", bbox_inches="tight")
#plt.savefig("images/global_signal_fgivenx_final.png", bbox_inches="tight")

plt.show()


######## POWER SPECTRUM SECTION ########

h=0.6704
selections = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}
data = {}
for band, selection in selections.items():
    data[band] = {}
    for field, sel in selection.items():
        data[band][field] = extract_data(
            datapath='data/observations_H1C_IDR2/pspec_h1c_idr2_field{}.h5',
            band=int(band),
            field=field,
            kstart=sel["kstart"],
            kstart_modulo=0.192,
            decimation_factor=2,
            set_negative_to_zero=True
            )




fig, axes = plt.subplots(nrows=2, ncols=len(files), sharey="row", figsize=(24,8))



emu_dsq = poweremu(loadfile="data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
k_array = load_files('data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_k_", name="hera", key='ks', endings=["mat"])[0]
kmask = np.array(k_array >= 8.5e-2) & (k_array <= 1)
k_array = k_array[kmask]/h
z_array = np.array([data["1"]["1"]["z"], data["2"]["1"]["z"]])






sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()

for band,z in enumerate(z_array):

    def emulatorModel2d(x,p, emu=emu_dsq, z=z):
        par0 = np.array([z, np.NaN, *p])
        params=np.tile(par0, (len(x), 1))
        params[:,1] = x
        #print(par0)
        return emu.predict(params)
    
    k_data_mask = (data[str(band+1)]["1"]["k_data"] >= 8.5e-2/h) & (data[str(band+1)]["1"]["k_data"] <= 1/h)

    if False:
        cbar1 = plot_contours(f=emulatorModel2d, x=k_array, samples=p_prior, ax=axes[band,0], weights=weights_prior,
                            colors=plt.cm.Blues_r, 
                            #lines=False,
                            #**{"contour_line_levels": [2,]}
                            cache="data/fgivenx/prior_dsq_z{:.2f}".format(z)
                            )
        
        axes[band,0].errorbar(x=data[str(band+1)]["1"]["k_data"][k_data_mask], y=data[str(band+1)]["1"]["dsq"][k_data_mask], yerr=data[str(band+1)]["1"]["std"][k_data_mask], capsize=3, color="k",
                    alpha=0.4,
                    ls="None",
                    ) 
        axes[band,0].set_xlim([k_array.min(), k_array.max()])
    axes[band,0].text(0.01,.99,"z={:.2f}".format(z), transform=axes[band,0].transAxes, ha="left", va="top", fontsize=12)
    axes[band,0].set_title("Prior")
    axes[band,0].set_ylabel(r"$\Delta_{21}^{2}\ [\mathrm{mK^2}]$")
    #axes[band,0].set_xlabel(r"$k\ [\mathrm{h\ cMpc^{-1}}]$")
    

    for j,(ax,file) in enumerate(zip(axes[band,:], files)):
        fn = file.split("/")[-2].split("_globalemu")[0].split("_")
        fn = "+".join([element[1:] for element in fn if "1" in element])
        sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
        p_sample = sample[paramNames].values
        weights_sample = sample.get_weights()
        c = anesthetic.plot.basic_cmap(ccb[j]).reversed()

        cbar1 = plot_contours(f=emulatorModel2d, x=k_array, samples=p_prior, ax=ax, weights=weights_prior,
                            colors=plt.cm.Greys_r, 
                            lines=True,
                            #**{"contour_line_levels": [2,]}
                            cache="data/fgivenx/prior_dsq_z{:.2f}_final".format(z),
                            alpha=0.4
                            )
        
        cbar2 = plot_contours(f=emulatorModel2d, x=k_array, samples=p_sample, ax=ax, weights=weights_sample,
                            colors=c, 
                            lines=True,
                            cache="data/fgivenx/"+fn+"_dsq_final"
                            # #**{"contour_line_levels": [2,]}
                            )
        ax.errorbar(x=data[str(band+1)]["1"]["k_data"][k_data_mask], y=data[str(band+1)]["1"]["dsq"][k_data_mask], yerr=data[str(band+1)]["1"]["std"][k_data_mask], capsize=3, color="k",
                alpha=1 if "HERA" in fn else 0.4,
                ls="None"
                )
        ax.set_xlabel(r"$k\ [\mathrm{h\ cMpc^{-1}}]$")
        ax.set_xlim([k_array.min(), k_array.max()])
        ax.set_yscale("log")
        ax.set_title(fn, fontsize=10)

        









plt.subplots_adjust(wspace=0, hspace=0.4)

#########LWA#########

import numpy as np
import matplotlib.pyplot as plt
import codes.itamar.radio_cutoff_calc as rad
from codes.emulator_poweremu import *
import anesthetic
from fgivenx import plot_contours

fig, axes = plt.subplots(nrows=1, ncols=len(files), sharey="row", figsize=(24,4))

SFR_emu = poweremu(loadfile="data/trained_emulators_poweremu/SFR1_emu_n400_l80808080_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)

def emulatorModel1d(p, emu=SFR_emu, arr=[6,7,8]):
    par0 = np.array([np.NaN, *p])
    params=np.tile(par0, (len(arr), 1))
    params[:,0] = arr
    return emu.predict(params)


[nu_obs, T_obs, dT_obs] = np.load('codes/itamar/LWA1_with_err.npy')
def Tradio_emu(x,p, z_cutoff=6.01):
    fr = 10**p[7] #change to params
    z_dense = np.linspace(z_cutoff-0.01, z_cutoff+0.01,2)
    sfr_dense = 10**(np.interp(z_dense, [6,7,8], np.log10( 
        emulatorModel1d(emu=SFR_emu, arr=[6,7,8], p=p)
    ) )) 

    nu_today, T_today = rad.get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
    T_model = np.mean(T_today, axis=0) * fr
    T_model_interp = np.interp((10**x)*1e9, nu_today.value, T_model)
    return np.log10(T_model_interp)

yerr = np.array([np.log10(T_obs+dT_obs) - np.log10(T_obs), np.log10(T_obs) - np.log10(T_obs-dT_obs)])

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").compress().prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()
x_prior=np.log10(np.geomspace((nu_obs/1e9).min(), (nu_obs/1e9).max(), 300))

#cbar1 = plot_contours(f=Tradio_emu, x=x_prior, 
#                    samples=p_prior, ax=axes[0], weights=weights_prior,
#                    colors=plt.cm.Blues_r, 
#                    #lines=False,
#                    #**{"contour_line_levels": [2,]}
#                    cache="data/fgivenx/prior_Tradio_res%s"%len(x_prior)
#                    )
#
#cbar2 = plot_contours(f=Tradio_emu, x=x, 
#                    samples=p_prior, ax=axes[1], weights=weights_prior,
#                    colors=plt.cm.Greys_r, 
#                    #lines=False,
#                    #**{"contour_line_levels": [2,]}
#                    cache="data/fgivenx/prior_Tradio_res%s"%len(x),
#                    #zorder=-1
#                    )
#
#axes[0].set_title("Prior")
axes[0].set_ylabel(r"$\log_{10} T_\mathrm{Radio}(z=0) \ [K]$")
#axes[0].set_xlabel(r"$\log_{10} \nu [\mathrm{GHz}]$")

#axes[0].errorbar(x=np.log10(nu_obs/1e9), y=np.log10(T_obs), yerr=yerr, capsize=4, lw=2, color="k",
#                alpha=0.5,
#                ls="dotted"
#                )

for i,(ax,file) in enumerate(zip(axes[:], files)):
    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])
    #fn = file.split("/")[-2].split("_globalemu")[0].replace("_", " ")
    sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    sample = sample if ("1Chandra"==fn) or ("1LWA"==fn) else sample.compress()
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    num = 400 if ("1Chandra"==fn) or ("1LWA"==fn) else 300
    x=np.log10(np.geomspace((nu_obs/1e9).min(), (nu_obs/1e9).max(), num))

    cbar1 = plot_contours(f=Tradio_emu, x=x_prior, 
                    samples=p_prior, ax=ax, weights=weights_prior,
                    colors=plt.cm.Greys_r, 
                    lines=False,
                    #**{"contour_line_levels": [2,]}
                    cache="data/fgivenx/prior_Tradio_res%s_final"%len(x_prior),
                    #zorder=-1
                    )
    
    cbar2 = plot_contours(f=Tradio_emu, x=x, 
                          samples=p_sample, ax=ax, weights=weights_sample,
                          ny=200 if ("1Chandra"==fn) or ("1LWA"==fn) else 100,
                          colors=c,#plt.cm.Blues_r, 
                          lines=True,
                          cache="data/fgivenx/"+fn+"_Tradio_res%s_final"%num,
                          #**{"contour_line_levels": [2,]}
                          )
    ax.set_xlabel(r"$\log_{10} \nu [\mathrm{GHz}]$")
    ax.set_title(fn, fontsize=10)

    ax.errorbar(x=np.log10(nu_obs/1e9), y=np.log10(T_obs), yerr=yerr, capsize=4, lw=2, color="k",
                alpha=1 if "LWA" in fn else 0.5,
                ls="solid" if "LWA" in fn else "dotted"
                #label="LWA1/ARCADE2"
                )


plt.subplots_adjust(wspace=0, hspace=0)

#print(np.logspace(np.log10(nu_obs/1e9).min(), np.log10(nu_obs/1e9).max(), 500))
plt.savefig("images/Tradio_fgivenx_final.pdf", bbox_inches="tight")
plt.savefig("images/Tradio_fgivenx_final.png", bbox_inches="tight")

#plt.show()

######## CXB fgivenx ########

fig, axes = plt.subplots(nrows=1, ncols=len(files), sharey="row", figsize=(24,4))

XRB_emu = poweremu(loadfile="data/trained_emulators_poweremu/CXBlog10_emu_n400_l50505050_t1e-05_o0.pkl", preprocesss_log_x=False, tol=1e-5, offset=0)

nu_keV = load_files("data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_nu_", name="hera", key="nu_keV", endings=["mat"])[0]
nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.1)#55)
nu_keV = nu_keV[nu_mask]

def emulatorModel1d(x, p, emu=XRB_emu):
    par0 = np.array([np.NaN, *p])
    params=np.tile(par0, (len(x), 1))
    params[:,0] = x
    return np.log10(emu.predict(params))

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_0Chandra_1LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").compress().prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()


#axes[0].set_title("Prior")
axes[0].set_ylabel(r"$\log \mathrm{CXB}$")
#axes[0].set_xlabel(r"$\log\nu$")

x_prior=np.log10(np.geomspace(start=nu_keV.min(), stop=nu_keV.max(), num=400))

#axes[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
#            alpha=0.2, 
#            ls="dotted",#if "1SARAS" in fn else 0,
#            )

#axes[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", alpha=0.2,) 

for i,(ax,file) in enumerate(zip(axes[:], files)):
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])
    sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    sample = sample if (fn=="LWA") or (fn=="SARAS") else sample.compress()
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    x = np.log10(nu_keV) if (fn=="LWA") or (fn=="SARAS") else np.log10(np.geomspace(start=nu_keV.min(), stop=nu_keV.max(), num=400))
    cbar1 = plot_contours(f=emulatorModel1d, x=x_prior,#np.log(nu_keV), 
                      samples=p_prior, ax=ax, weights=weights_prior,
                      ny=1000,
                    colors=plt.cm.Greys_r, 
                    lines=False,
                    #**{"contour_line_levels": [2,]}
                    cache="data/fgivenx/prior_XRB_res%s_final"%len(x_prior)
                    )
    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()
    cbar2 = plot_contours(f=emulatorModel1d, x=x,#np.log(nu_keV), 
                          samples=p_sample, ax=ax, weights=weights_sample,
                        colors=c, lines=False,
                        cache="data/fgivenx/"+fn+"_XRB_res%s_final"%len(x)
                        #**{"contour_line_levels": [2,]}
                        )
    #ax.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
    #        alpha=0.2 if "1SARAS" in fn else 0.2,
    #        ls="solid" if "1SARAS" in fn else "dotted",
    #        ) 
    ax.set_xlabel(r"$\log\nu$")
    ax.set_title(fn)



plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("images/CXB_fgivenx_final.pdf", bbox_inches="tight")
plt.savefig("images/CXB_fgivenx_final.png", bbox_inches="tight")

plt.show()

#print(np.logspace(start=np.log(nu_keV).min(), stop=np.log(nu_keV).max(), num=500, base=np.exp(1)))
#print(np.log(np.geomspace(start=nu_keV.min(), stop=nu_keV.max(), num=500)))
#print(np.log(nu_keV))



#########GLOBAL AND POWER#########
import matplotlib.gridspec as gridspec
labels = ["Joint",
          "HERA",
          "X-ray Background",
          "Radio Background",
          "SARAS3",
         ]


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(6, 5, height_ratios=[1, 1, 0.4, 1, 0.4, 0.5,], width_ratios=[1, 1, 1, 1, 1],wspace=0, hspace=0)
#plt.rcParams["font.size"]=8

axes_ps = np.empty(shape=(2,5),dtype=object)
for row in range(1,-1,-1):
    for col in range(5):
        axes_ps[row,col] = plt.subplot(gs[row, col],
                                       sharex = axes_ps[1,col] if row==0 else None,
                                       sharey = axes_ps[row,0] if col!=0 else None,
                                        )
        axes_ps[row, col].tick_params(axis='y', labelleft=False if col !=0 else True)
        axes_ps[row, col].tick_params(axis='x', labelbottom=False if row == 0 else True)
        #axes_ps[row,col] = plt.subplot(gs[row, col])

axes_gs = np.empty(shape=(5),dtype=object)
for col in range(5):
    axes_gs[col] = plt.subplot(gs[3, col], sharey=axes_gs[0] if col!=0 else None)
    axes_gs[col].tick_params(axis='y', labelleft=False if col!=0 else True)

cbar_grid = gridspec.GridSpecFromSubplotSpec(6, 4, subplot_spec=gs[5, 4:], wspace=0., hspace=0.)
axes_cbar = np.empty(shape=(6),dtype=object)
for row in range(6):
    axes_cbar[row] = plt.subplot(cbar_grid[row, 1:])

fineness=1

h=0.6704

#like_hera.loglike(m, return_individual_loglikes=return_individual_loglikes)

#selection = {"1": {"D": {"kstart":0.36}}}#, "C": {"2": {"kstart":0.34}}}
#selection = {"1": {"1": {"kstart":0.256}}, "2": {"1": {"kstart":0.192}}}

selection = [{"1": {"D": {"kstart":0.36}}}, {"2": {"C": {"kstart":0.34}}}]

dpath = [
    'data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
    'data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5']

like_hera = likelihood(
    datapath=dpath,
    decimation_factor=2,
    selections=selection
)

data = like_hera.data

emu_dsq = poweremu(loadfile="data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
k_array = load_files('data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_k_", name="hera", key='ks', endings=["mat"])[0]
kmask = np.array(k_array >= 8.5e-2) & (k_array <= 1)
k_array = k_array[kmask]/h
z_array = np.array([data["1"]["D"]["z"], data["2"]["C"]["z"]])

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()
fields = ["D","C"]
for band,z in enumerate(z_array):

    def emulatorModel2d(x,p, emu=emu_dsq, z=z):
        par0 = np.array([z, np.NaN, *p])
        params=np.tile(par0, (len(x), 1))
        params[:,1] = x
        #print(par0)
        return emu.predict(params)
    
    k_data_mask = (data[str(band+1)][fields[band]]["k_data"] >= 8.5e-2/h) & (data[str(band+1)][fields[band]]["k_data"] <= 1/h)

    axes_ps[band,0].text(0.01,.99,"z={:.2f}".format(z), transform=axes_ps[band,0].transAxes, ha="left", va="top", fontsize=12)
    axes_ps[band,0].set_ylabel(r"$\Delta_{21}^{2}\ [\mathrm{mK^2}]$")
    

    for j,(ax,file) in enumerate(zip(axes_ps[band,:], files)):
        fn = file.split("/")[-2].split("_globalemu")[0].split("_")
        fn = "+".join([element[1:] for element in fn if "1" in element])
        sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
        p_sample = sample[paramNames].values
        weights_sample = sample.get_weights()
        c = anesthetic.plot.basic_cmap(ccb[j]).reversed()

        cbar1 = plot_contours(f=emulatorModel2d, x=k_array, samples=p_prior, ax=ax, weights=weights_prior,
                            colors=plt.cm.Greys_r, 
                            lines=False,
                            #**{"contour_line_levels": [2,]}
                            cache="data/fgivenx/prior_dsq_z{:.2f}_final".format(z),
                            alpha=0.4,
                            fineness=fineness
                            )
        
        cbar2 = plot_contours(f=emulatorModel2d, x=k_array, samples=p_sample, ax=ax, weights=weights_sample,
                            colors=c, 
                            lines=True,
                            cache="data/fgivenx/"+fn+"_dsq_final",
                            fineness=fineness
                            # #**{"contour_line_levels": [2,]}
                            )
        ax.errorbar(x=data[str(band+1)][fields[band]]["k_data"][k_data_mask], y=data[str(band+1)][fields[band]]["dsq"][k_data_mask], yerr=data[str(band+1)][fields[band]]["std"][k_data_mask], capsize=3, color="k",
                alpha=1 if "HERA" in fn else 0.4,
                ls="None",label="HERA Upper Limits"
                )
        
        ax.set_xlim([k_array.min(), k_array.max()])
        ax.set_yscale("log")
        if band==0:
            ax.set_title(labels[j], fontsize=10)
        if band!=0:
            ax.set_xlabel(r"$k\ [\mathrm{h\ cMpc^{-1}}]$")







sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()

freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt("data/SARAS3/SARAS_3_averaged_spectrum.txt").T
z = np.linspace(6,28,100)#(1420/freq-1)[::-1][::10]
model = keras.models.load_model('data/globalemu/emulator14/results/model.h5', compile=False)
predictor = evaluate(base_dir='data/globalemu/emulator14/results/', model=model, z=z, gc=False, logs=[]) #0,1,2,3,7], )

def signal_emu(z, p):
    signal = predictor(p[:9])[0] #to K
    return signal




axes_gs[0].set_ylabel(r"$T_{21}\ [\mathrm{mK}]$")
#axes_gs[0].set_xlabel(r"$z$")

axes_gs[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2, 
            ls="solid"# if "1SARAS" in fn else 0,
            )

#axes_gs[0].plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", alpha=0.2,) 
insets = []
for i,(ax,file) in enumerate(zip(axes_gs, files)):
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])

    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()
    

    cbar1 = plot_contours(f=signal_emu, x=z, samples=p_prior, ax=ax, weights=weights_prior,
                    lines=False,
                    colors=plt.cm.Greys_r, 
                    #**{"contour_line_levels": [2,]}
                    cache="data/fgivenx/prior_T21_final",
                    alpha=0.8,
                    fineness=fineness
                    )

    sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    cbar2 = plot_contours(f=signal_emu, x=z, samples=p_sample, ax=ax, weights=weights_sample,
                        colors=c, lines=True,
                        cache="data/fgivenx/"+fn+"_T21_final",
                        fineness=fineness,
                        #**{"contour_line_levels": [2,]}
                        )
    
    ax.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2 if "SARAS" in fn else 0.2,
            ls="solid" if "SARAS" in fn else "dotted",
            ) 
    ax.set_xlabel(r"$z$")

    cb = fig.colorbar(cbar2, cax=axes_cbar[i], orientation='horizontal')
    #axes_cbar[i].set_ylabel("pik og loort")
    #cb.set_label(label=fn,position="bottom")
    axes_cbar[i].tick_params(axis='both', labelbottom=False,labelleft=False, labelright=False)

    #print(
    #     [element for element in dir(axes_cbar[i]) if 'colorbar' in element]
    #    )# colorbar.set_label(label=fn, position="left")
    
    #axes_cbar[i].set_yticks([0.5])
    #axes_cbar[i].set_yticklabels([fn])
    #axes_cbar[i].set_yticklabels(["label%s"%i])
    ax.set_ylim([-5500, 600])
    #print(4)
    if (i==0) or (i==1):
        #prior colorbar
        fig.colorbar(cbar1, cax=axes_cbar[-1], orientation='horizontal')
        axes_cbar[-1].set_xlabel(r"$\sigma$")
        #axes_cbar[-1].set_ylabel("Prior")
        #cb.set_label("Prior",position="center")
        #inset
        axes_ins = axes_gs[i].inset_axes([0.1,0.15-1.3,0.7,0.8])
        insets.append(axes_ins)
        saras_plot = axes_ins.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2 if "SARAS" in fn else 0.2,
            ls="solid" if "SARAS" in fn else "dotted",
            label="SARAS3 Global Signal"
            ) 
        contour=plot_contours(f=signal_emu, x=z, samples=p_sample, ax=axes_ins, weights=weights_sample,
                        colors=c, lines=True,
                        cache="data/fgivenx/"+fn+"_T21_final",
                        fineness=fineness
                        )
        #axes_ins.axhline(-230)
        axes_ins.set_ylim([-250,100])
        if i==0: axes_ins.set_ylabel(r"$T_{21}\ [\mathrm{mK}]$")

        ax.indicate_inset_zoom(axes_ins, edgecolor="black", lw=2)


handle_saras, label_saras = insets[0].get_legend_handles_labels()

handle_hera, label_hera = axes_ps[0,2].get_legend_handles_labels()

ax_leg = fig.add_subplot(gs[5, 2])
ax_leg.axis("off")
import matplotlib.lines as mlines
ax_leg.legend(
    handles=[handle_saras[0],handle_hera[0], mlines.Line2D([], [], lw=4, linestyle="solid", color="grey", label="Prior")] + 
    [mlines.Line2D([], [], lw=4, linestyle="solid", color=ccb[i], label=label) for i,label in enumerate(labels)], 
    loc='upper left', 
    ncol=2,
    #bbox_to_anchor=(0,1.) ,#fontsize=16,frameon=False, 
    #title="smoothened {0} contour level".format(lvl)
)


#plt.figure()
#for i,coll in enumerate(contour_collection):
#    print(i)
#    contour_path = coll.get_paths()[0]
#    # Get the contour curve coordinates
#    contour_curve = contour_path.vertices

#    plt.plot(contour_curve[:, 0], contour_curve[:, 1])



plt.savefig("images/dsq_gs_fgivenx_final.pdf", bbox_inches="tight")
#plt.savefig("images/dsq_gs_fgivenx_final.svg", bbox_inches="tight")

#plt.show()


"""
"""
#########GLOBAL AND POWER v2#########
import matplotlib.gridspec as gridspec
files = [
         "scripts/non-public/1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14test2idr3_nlive_1000/run",#"scripts/non-public/1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14_nlive_1000/run_h1c_idr2",
         "scripts/non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu315emu14testidr3_nlive_10000/run_idr3",#"scripts/non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_h1c_idr2",
         "scripts/non-public/0HERA_0Chandra_0LWA_1SARAS_globalemu315emu14_nlive_1000/run_no_idr",
         "scripts/non-public/0HERA_0Chandra_1LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
         "scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
        ]

labels = ["Joint",
          "HERA",
          "SARAS3",
          "Radio Background",
          "X-ray Background",
         ]


#fig = plt.figure(figsize=(12, 8))
#gs = gridspec.GridSpec(4, 5, height_ratios=[1, 1, 0.4, 0.5,], width_ratios=[1, 1, 1, 1, 1],wspace=0, hspace=0)



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure and GridSpec
plt.rcParams["font.size"]=13
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 0.4, 0.5], wspace=0, hspace=0)

# Create subplots using add_subplot
axes = np.empty(shape=(2,5),dtype=object)# [[None] * 5 for _ in range(4)]

for row in range(1,-1,-1):
    for col in range(5):
        axes[row,col] = fig.add_subplot(gs[row, col], sharex=axes[1,col] if row!=1 else None, sharey=axes[row,0] if col!=0 else None)
        axes[row,col].tick_params(axis='both', labelbottom=False if row!=1 else True, labelleft=False if col!=0 else True)

cbar_grid = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[3, 3:4], wspace=0., hspace=0.)
axes_cbar = np.empty(shape=(6),dtype=object)
for row in range(6):
    axes_cbar[row] = plt.subplot(cbar_grid[row, :-1])

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()

emu_dsq = poweremu(loadfile="data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)

fineness=1
h=0.6704

selection = [{"1": {"D": {"kstart":0.36}}}, {"2": {"C": {"kstart":0.34}}}]

dpath = [
    'data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
    'data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5']

like_hera = likelihood(
    datapath=dpath,
    decimation_factor=2,
    selections=selection
)

data = like_hera.data

k = data["2"]["C"]["k_data"][0]




#axes[0,0].text(0.01,.99,r"$k={:.2f}\ \mathrm{{h/Mpc}}$".format(k), transform=axes[0,0].transAxes, ha="left", va="top", fontsize=12)
axes[0,0].set_ylabel(r"$\Delta_{{21}}^{{2}}\ \left[\mathrm{{mK^2}}\right]$".format(k))

def emulatorModel2d(x,p, emu=emu_dsq):
    par0 = np.array([np.NaN, k, *p])
    params=np.tile(par0, (len(x), 1))
    params[:,0] = x
    return np.log10(emu.predict(params))

z = np.linspace(7,26,400)#(1420/freq-1)[::-1][::10]
for i,(ax,file) in enumerate(zip(axes[0,:],files)):
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])

    sample = anesthetic.read_chains(root=files[i]) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    
    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()

    cbar_prior = plot_contours(f=emulatorModel2d, x=z, samples=p_prior, ax=ax, weights=weights_prior,
                    colors=plt.cm.Greys_r, 
                    lines=False,
                    #ny=200,
                    cache="data/fgivenx/prior_dsqkvz_final",
                    fineness=fineness,
                    alpha=0.4,
                    log_transform=True,
                    )
    
    cbar_posterior = plot_contours(f=emulatorModel2d, x=z, samples=p_sample, ax=ax, weights=weights_sample,
                                   colors=c, 
                                   lines=True,
                                   #ny=200,
                                   cache="data/fgivenx/"+fn+"_dsqkvz_final",
                                   fineness=fineness,
                                   log_transform=True,
                                   )
    
    ax.scatter(data["2"]["C"]["z"], data["2"]["C"]["dsq"][0]+2*data["2"]["C"]["std"][0], color="k", 
               alpha=1 if "HERA" in fn else 0.4, marker=7, s=60,
               label="HERA Upper Limits")
    ax.set_yscale("log")
    ax.set_title(labels[i])#, fontsize=14)





freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt("data/SARAS3/SARAS_3_averaged_spectrum.txt").T
model = keras.models.load_model('data/globalemu/emulator14/results/model.h5', compile=False)
predictor = evaluate(base_dir='data/globalemu/emulator14/results/', model=model, z=z, gc=False, logs=[]) #0,1,2,3,7], )

def signal_emu(z, p):
    signal = predictor(p[:9])[0] #to K
    return signal

axes[1,0].set_ylabel(r"$T_{21}\ [\mathrm{mK}]$")

for i,(ax,ax_cbar,file) in enumerate(zip(axes[1,:],axes_cbar,files)):
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "+".join([element[1:] for element in fn if "1" in element])

    sample = anesthetic.read_chains(root=file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()

    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()

    cbar_prior = plot_contours(f=signal_emu, x=z, samples=p_prior, ax=ax, weights=weights_prior,
                    lines=False,
                    colors=plt.cm.Greys_r, 
                    cache="data/fgivenx/prior_T21_final",
                    alpha=0.8,
                    fineness=fineness
                    )

    cbar_posterior = plot_contours(f=signal_emu, x=z, samples=p_sample, ax=ax, weights=weights_sample,
                        colors=c, lines=True,
                        cache="data/fgivenx/"+fn+"_T21_final",
                        fineness=fineness,
                        )
    
    cbar = fig.colorbar(cbar_posterior, cax=ax_cbar, orientation='horizontal')
    if i==0: cbar2 = fig.colorbar(cbar_prior, cax=axes_cbar[5], orientation='horizontal')
    
    ax.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2 if "SARAS" in fn else 0.2,
            ls="solid" if "SARAS" in fn else "dotted",
            ) 

    if (i<3):
        #inset
        axes_ins = ax.inset_axes([0.1,0.15-1.3,0.7,0.8])
        saras_plot = axes_ins.plot((1420/freq-1)[::-1], (T_SARAS-fg_fit)*1000, color="k", 
            alpha=0.2 if "SARAS" in fn else 0.2,
            ls="solid" if "SARAS" in fn else "dotted",
            label="SARAS3 Residuals"
            ) 
        
        contour=plot_contours(f=signal_emu, x=z, samples=p_sample, ax=axes_ins, weights=weights_sample,
                        colors=c, lines=True,
                        cache="data/fgivenx/"+fn+"_T21_final",
                        fineness=fineness
                        )
        #axes_ins.axhline(-230)
        axes_ins.set_ylim([-250,100])
        axes_ins.set_xlabel(r"Redshift $z$")
        if i==0: axes_ins.set_ylabel(r"$T_{21}\ [\mathrm{mK}]$")

        ax.indicate_inset_zoom(axes_ins, edgecolor="black", lw=0)

        if i==0:
            ax_test = axes_ins

    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylim([-5500, 600])
    ax_cbar.tick_params(axis='both', labelbottom=False,labelleft=False, labelright=False)
    



handle_saras, label_saras = axes_ins.get_legend_handles_labels()
handle_hera, label_hera = axes[0,-1].get_legend_handles_labels()

ax_leg = fig.add_subplot(gs[3, 4])
ax_leg.axis("off")
import matplotlib.lines as mlines
ax_leg.legend(
    handles=[handle_saras[0],
             handle_hera[0], 
             mlines.Line2D([], [], lw=4, linestyle="solid", color="grey", label="Prior")] + 
             [mlines.Line2D([], [], lw=4, linestyle="solid", color=ccb[i], label=label) for i,label in enumerate(labels)], 
    loc=(0,-0.8)#'upper left', 
    #ncol=2,
    #bbox_to_anchor=(0,1.) ,#fontsize=16,frameon=False, 
    #title="smoothened {0} contour level".format(lvl)
)


#fig,axes = plt.subplots(1,2)
#like_hera.plot_data(axes=axes)

#print("1D k:", data["1"]["D"]["k_data"],"\ndsq:",data["1"]["D"]["dsq"],"\nstd:",data["1"]["D"]["std"])
#print("2C k:", data["2"]["C"]["k_data"],"\ndsq:",data["2"]["C"]["dsq"],"\nstd:",data["2"]["C"]["std"])


axes_cbar[5].set_xlabel(r"$\sigma$")
fig.align_ylabels(axes[:,0])
#plt.savefig("images/dsq_gs_fgivenx_final.pdf", bbox_inches="tight")
#plt.show() 
"""
"""

#print(dir(axes[0,0].collections))#.get_paths()[0].vertices)
plt.figure()
for i,collection in enumerate(ax_test.collections):
    if i>4:
        contour_paths = collection.get_paths()
        for j,path in enumerate(contour_paths):
            x, y = path.vertices[:, 0], path.vertices[:, 1]
            plt.plot(x, y, linestyle='-', marker=None, label="i:{0}, j:{1}".format(i,j))
plt.legend()

plt.figure()
cont = ax_test.collections[5].get_paths()
for j,path in enumerate(cont):
    if j==0:
        x, y = path.vertices[:, 0], path.vertices[:, 1]
        print(x.shape,x[np.argmin(y)], y.min())
        plt.plot(x, y, marker=".", label="i:{0}, j:{1}".format(7,j)) 
plt.legend()

#plt.show()

"""

######## xHI ########

sample_prior = anesthetic.read_chains(root="scripts/non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr").prior()
p_prior = sample_prior[paramNames].values
weights_prior = sample_prior.get_weights()

path="/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"
data_dir = path+"data/globalemu/xHI_emulator1/" #'downloaded_data/'
base_dir = data_dir+"results/"#'downloaded_data/'
model = keras.models.load_model(base_dir+'model.h5', compile=False)
predictor = evaluate(base_dir=base_dir, model=model, gc=False, logs=[])#0,1,2,3,7

def xHI_emu(z, p):
    signal = predictor(p[:9])[0] #to K
    return signal


fig, ax = plt.subplots(nrows=1, ncols=1)#, sharey="row", figsize=(24,4))

ax.set_ylabel(r"$x_{\mathrm{HI}}$")



    

cbar1 = plot_contours(f=xHI_emu, x=np.arange(6,20.1,0.1).round(1), samples=p_prior, ax=ax, weights=weights_prior,
                lines=False,
                colors=plt.cm.Greys_r, 
                #**{"contour_line_levels": [2,]}
                cache="data/fgivenx/prior_xHI",
                alpha=0.8,
                fineness=1,
                )


for i,file in enumerate([#"1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14test2idr3_nlive_1000/run",
                         "0HERA_0Chandra_0LWA_0SARAS_1xHI_globalemu315emu14testpipe_nlive_10000/run", 
                         ]):
    c = anesthetic.plot.basic_cmap(ccb[i]).reversed()
    sample = anesthetic.read_chains(root="scripts/non-public/"+file) #h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2
    p_sample = sample[paramNames].values
    weights_sample = sample.get_weights()
    cbar2 = plot_contours(f=xHI_emu, x=np.arange(6,20.1,0.1).round(1), samples=p_sample, ax=ax, weights=weights_sample,
                            colors=c, lines=True,
                            cache="data/fgivenx/sample_1_xHI",
                            #**{"contour_line_levels": [2,]}
                            fineness=1
                            )


z_approx=[6.0, 7.0, 7.6]
xHI=np.array([0.09, 0.57, 0.855]) #[approx McGreer+15, Mason+18, Hoag+19]0.06
xHI_sigma=np.array([0.08, 0.13, 0.075])#0.05
ax.errorbar(z_approx, xHI, xHI_sigma, ls="none", c="k", fmt="o", capsize=4, label="Approx McGreer+2014, \nMason+2018, Hoag+2019")
ax.legend()
ax.set_xlabel(r"Redshift $z$")



#plt.savefig("images/global_signal_fgivenx_final.pdf", bbox_inches="tight")
#plt.savefig("images/global_signal_fgivenx_final.png", bbox_inches="tight")
plt.savefig("images/xHI.png", dpi=300, bbox_inches="tight")
#plt.show()

import matplotlib
import matplotlib.pyplot as plt
from codes.emulator_poweremu import *
import anesthetic
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter

ccb = sns.color_palette("colorblind")

#data_idr2 = anesthetic.read.samplereader.PolyChordReader("non-public/idr2_Sims2022_nlive_10000_reduced_tau/run_IDR2")
data_idr2 = anesthetic.read.samplereader.PolyChordReader("non-public/idr6_Sims2022_nlive_10000_reduced_tau/run_IDR6")

sample = np.copy(data_idr2.samples()[0])
logL = np.copy(data_idr2.samples()[1])
logL_birth = np.copy(data_idr2.samples()[2])
params, tex = data_idr2.paramnames()

####Remove discrete params####

discrete_params = {
            "alpha": [-1, -1.3, -1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            "feed": [0, 1],
            "delay": [0, 0.75]}

i = np.array([ np.where( np.array(params) == key )[0] for key in discrete_params.keys()]).flatten()

sample = np.delete(sample, i, axis=1)
params = np.delete(params, i, axis=0)
for key in discrete_params.keys():
    del tex[key]

data_idr2 = anesthetic.NestedSamples(data = sample, columns = params, logL = logL, tex = tex, logL_birth = logL_birth)




##################################################################################################################################





#data_idr4 = anesthetic.read.samplereader.PolyChordReader("scripts/non-public/idr4_Sims2022_nlive_10000/run_IDR4")
data_idr4 = anesthetic.read.samplereader.PolyChordReader("non-public/idr4_Sims2022_nlive_10000_reduced_tau_sc/run_IDR4")

sample = np.copy(data_idr4.samples()[0])
logL = np.copy(data_idr4.samples()[1])
logL_birth = np.copy(data_idr4.samples()[2])
params, tex = data_idr4.paramnames()

####Remove discrete params####

discrete_params = {
            "alpha": [-1, -1.3, -1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            "feed": [0, 1],
            "delay": [0, 0.75]}

i = np.array([ np.where( np.array(params) == key )[0] for key in discrete_params.keys()]).flatten()

sample = np.delete(sample, i, axis=1)
params = np.delete(params, i, axis=0)
for key in discrete_params.keys():
    del tex[key]

data_idr4 = anesthetic.NestedSamples(data = sample, columns = params, logL = logL, tex = tex, logL_birth = logL_birth)



##################################################################################################################################

samples_idr2 = data_idr2#anesthetic.NestedSamples(root="scripts/non-public/idr_chains4/run_IDR4")
samples = data_idr4#anesthetic.NestedSamples(root="scripts/non-public/idr_chains4_real_2/run_IDR4")

paramNames = list(samples.columns[:6])#

texDict = {"log10fstarII": r"$\log_{10} f_{\rm star, II}$",
           "log10fstarIII": r"$\log_{10} f_{\rm star, III}$",
           "log10Vc": r"$\log_{10} V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "alpha": r"$\alpha$",
           "nu_0": r"$\nu_{\rm 0}$",
           "tau": r"$\tau$",
           "log10fradio": r"$\log_{10} f_{\rm r}$",
           "pop": r"$\rm pop$",
           "feed": r"$\rm feed$",
           "delay": r"$\rm delay$",}

discrete_params = {
            "alpha": [-1, -1.3, -1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
            "feed": [0, 1],
            "delay": [0, 0.75]}

discrete_params_bins = {
            "alpha": np.arange(-0.5,4,1), #[-1, -1.3, -1.5],
            "nu_0": np.arange(-0.5,17,1),#[*range(100,1500,100), 1500, 2000, 3000],
            "pop": np.arange(-0.5,4,1),#[231, 232, 233],
            "feed": np.arange(-0.5,3,1),#[0, 1],
            "delay": np.arange(-0.5,3,1),}#[0, 0.75]}

types={'lower':'hist', 'diagonal':'hist'}#, 'upper':'scatter'}


fig = plt.figure(figsize=(15,15))
fig,axes = anesthetic.plot.make_2d_axes(params=paramNames, fig=fig, tex=texDict, lower=types["lower"], diagonal=types["diagonal"],)
bins=100
local_kwargs = {'diagonal': {'histtype': 'step', 'linewidth': 2,'edgecolor': ccb[0], 'bins': bins}, 
           'lower': {'bins': bins, 'color': ccb[0], 'vmin': 0, 'zorder': -10, 'rasterized': True, "range": None, 'levels': None},#[0.95,0.68]}, 
           'upper': {'alpha': 0.3}}
lvl = 4.550
for y, row in axes.iterrows():
    for x, ax in row.iteritems():
        
        #j = np.unravel_index(i, axes.shape)
        
        if ax is not None:
            pos = ax.position
            ax_ = ax.twin if x == y else ax
            plot_type = types.get(pos, None)
            local_kwargs["lower"]["range"] = [[samples[x].min(), samples[x].max()], [samples[y].min(), samples[y].max()]]
            if x in discrete_params or y in discrete_params:
                n_bin_x = bins
                n_bin_y = bins
                if x in discrete_params:
                    n_bin_x = discrete_params_bins[x] #len(discrete_params[x])
                    local_kwargs["lower"]["range"][0] = [discrete_params_bins[x].min(), discrete_params_bins[x].max()]
                    local_kwargs["diagonal"]["bins"] = discrete_params_bins[x] 
                if y in discrete_params:
                    n_bin_y = discrete_params_bins[y]#len(discrete_params[y])
                    local_kwargs["lower"]["range"][1] = [discrete_params_bins[y].min(), discrete_params_bins[y].max()]
                    
                local_kwargs["lower"]["bins"] = [n_bin_x,n_bin_y]
            else:
                local_kwargs["lower"]["bins"] = bins
                local_kwargs["lower"]["range"] = None
                local_kwargs["diagonal"]["bins"] = bins
            lkwargs = local_kwargs.get(pos, {})
            

            
            local_kwargs["lower"]["color"] = ccb[0]
            local_kwargs["diagonal"]["edgecolor"] = ccb[0]
            #anesthetic.NestedSamples.plot(self=samples, ax=ax_, paramname_x=x, paramname_y=y, plot_type=plot_type, **lkwargs)
            if pos=="lower": 

                color = matplotlib.colors.rgb2hex(local_kwargs[pos]["color"], keep_alpha=True)
                cmap = anesthetic.plot.basic_cmap( color )
                
                
                
                pdf, xx, yy = np.histogram2d(samples[x], samples[y], weights=samples.weights, 
                                           bins=local_kwargs["lower"]["bins"], range=local_kwargs[pos]["range"], density=False)
                image = ax.pcolormesh(xx, yy, pdf.T, cmap=cmap, vmin=local_kwargs[pos]["vmin"],)
                              
                #pdf, xx, yy, image = ax.hist2d(samples[x], samples[y], weights=samples.weights, cmap=cmap, 
                #                               range=local_kwargs[pos]["range"], vmin=local_kwargs[pos]["vmin"],
                #                               bins=local_kwargs["lower"]["bins"],
                #                               )
                smooth_data = gaussian_filter(pdf, 1.5)
                cs = ax.contour(smooth_data.T,extent=[xx.min(),xx.max(),yy.min(),yy.max()],linewidths=3,alpha=0.7,levels=[lvl], colors=color)
                
                samples_idr2
                pdf, xx, yy = np.histogram2d(samples_idr2[x], samples_idr2[y], weights=samples_idr2.weights, 
                                           bins=local_kwargs["lower"]["bins"], range=local_kwargs[pos]["range"], density=False)
                smooth_data = gaussian_filter(pdf, 2) #sigma=1.5
                cs = ax.contour(smooth_data.T,extent=[xx.min(),xx.max(),yy.min(),yy.max()],linewidths=3,alpha=0.7,levels=[lvl],colors=matplotlib.colors.rgb2hex(ccb[1], keep_alpha=True))
                
            if pos=="diagonal":
                local_kwargs["diagonal"]["edgecolor"] = ccb[0]
                anesthetic.NestedSamples.plot(self=samples, ax=ax_, paramname_x=x, paramname_y=y, plot_type=plot_type, **lkwargs)
                local_kwargs["diagonal"]["edgecolor"] = ccb[1]
                anesthetic.NestedSamples.plot(self=samples_idr2, ax=ax_, paramname_x=x, paramname_y=y, plot_type=plot_type, **lkwargs)
                
                if x=="tau":
                    #ax.axvspan(0.0569-0.0066, 0.0569+0.0073, alpha=0.2, color='grey', label="Planck+2018")
                    ax.axvspan(0.054-1*0.007, 0.054+1*0.007, alpha=0.2, color='grey', label="Planck+2018")
                    #ax.legend()
            
            #local_kwargs["lower"]["alpha"] = 0.3
            #local_kwargs["lower"]["color"] = ccb[1]
            #local_kwargs["diagonal"]["edgecolor"] = ccb[1]
            #image = anesthetic.NestedSamples.plot(self=samples_idr2, ax=ax_, paramname_x=x, paramname_y=y, plot_type=plot_type, **lkwargs)

            
            
            #    counts,ybins,xbins = np.histogram2d(samples[x].values,samples[y].values,range=local_kwargs["lower"]["range"], bins=local_kwargs["lower"]["bins"])
            #    ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)


            if plot_type is None:
                ax.set_axis_off()
            

            ax.tick_params(axis='both', labelrotation = 45)
            if x in discrete_params and x!="nu_0" and ax!=None:
                ax.set_xticks(range( len( discrete_params[x] )  ))
                ax.set_xticklabels( discrete_params[x] )
            if y in discrete_params and y!="nu_0" and ax!=None:
                ax.set_yticks(range( len( discrete_params[y] )  ))
                ax.set_yticklabels( discrete_params[y] )
                
            if x=="nu_0" and ax is not None:
                ax.set_xticks(range( len( discrete_params["nu_0"] )  )[::5])
                ax.set_xticklabels( discrete_params["nu_0"][::5] )
            if y=="nu_0" and ax is not None:
                ax.set_yticks(range( len( discrete_params["nu_0"] )  )[::5])
                ax.set_yticklabels( discrete_params["nu_0"][::5] )

            #if local_kwargs["lower"]["range"] == None:
            #    strr= "None"
            #else:
            #    strr= str( np.round( local_kwargs["lower"]["range"], 2 ) )
            #    #print(x,y,ax_.get_xlim(), ax_.get_ylim())
            #ax_.text(0.1,0.7,strr, transform=ax_.transAxes,fontsize=8)
            
IDR4_leg  = matplotlib.lines.Line2D([], [], lw=4, linestyle="solid", color=ccb[0], label="IDR4")
IDR2_leg = matplotlib.lines.Line2D([], [], lw=4, linestyle="solid", color=ccb[1], label="IDR2")
fig.axes[-5].legend(handles=[IDR4_leg, IDR2_leg], loc='upper left',fontsize=16,frameon=False, title="smoothened {0} contour level".format(lvl))
plt.show()
plt.savefig("analyze_plots/emulator_test_idr4_sc_idr6.png")



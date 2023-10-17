import matplotlib
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import scipy.interpolate as sip
import numpy as np
import anesthetic
import seaborn as sns
ccb = sns.color_palette("colorblind")

#################################settings to change#################################################################
bins=20
sigma_smooth = 1.25 #gaussian sigma to smoothen bins for smooth contour lines
drop_discrete = True #remove discrete parameters
files = ["/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/scripts/non-public/idr4_nlive_10000/run_idr4", #path to Polychord files
         #"idr2_nlive_10000/run_idr2",
        ]

#################################define params and load samples#################################################################
texDict = {"log10fstarII": r"$\log_{10} f_{\rm star, II}$",
           "log10fstarIII": r"$\log_{10} f_{\rm star, III}$",
           "log10Vc": r"$\log_{10} V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "alpha": r"$\alpha$",
           "nu_0": r"$\nu_{\rm 0}$",
           "tau": r"$\tau$",
           "log10fradio": r"$\log_{10} f_{\rm r}$",
           "pop": r"$\rm pop$",
          }
discrete_params = {
            "alpha": [1, 1.3, 1.5],
            "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
            "pop": [231, 232, 233],
}
discrete_params_bins = {
            "alpha": np.arange(-0.5,len(discrete_params["alpha"]),1), #[-1, -1.3, -1.5],
            "nu_0": np.arange(-0.5,len(discrete_params["nu_0"]),1),#[*range(100,1500,100), 1500, 2000, 3000],
            "pop": np.arange(-0.5,len(discrete_params["pop"]),1),#[231, 232, 233],
}

#Load samples and keep/remove discrete parameters
if drop_discrete:
    samples_list = [anesthetic.NestedSamples(root=file).drop(columns=list(discrete_params.keys())) for file in files]
    nparams = len(texDict.keys()) - len(discrete_params.keys())
else:
    samples_list = [anesthetic.NestedSamples(root=file) for file in files]
    for sample in samples_list:
        for name in discrete_params.keys():
            i = np.round(sample[name].values).astype("int")
            sample[name] = i #convert Polychord continuous parameter to discrete parameter
    nparams = len(texDict.keys())

#################################create grid for plots#################################################################

fig = plt.figure(figsize=(18,18))
grid = GS(nparams, nparams, hspace=0, wspace=0)

axes = np.full((nparams, nparams), False, dtype=object)
axes_info = np.empty(shape=(nparams, nparams), dtype=object) #[]

shared = {}

for row in range(nparams-1,-1,-1):
    for col in range(nparams):
        if row>col: #lower triangle
            if row==nparams-1:
                shared["sharex"]=False
                shared["sharey"]=False
                if col!=0:
                    shared["sharey"]=axes[nparams-1,0]
            if col==0:
                shared["sharex"]=False
                shared["sharey"]=False
                if row!=nparams-1:
                    shared["sharex"]=axes[nparams-1,0]
            if (row!=nparams-1) & (col!=0):
                shared["sharey"]=axes[row,0]
                shared["sharex"]=axes[nparams-1,col]
            ax = fig.add_subplot(grid[row, col], **{k:v for k, v in shared.items() if v is not False})
            #labels
            if (row==nparams-1) & (col!=0):
                ax.tick_params('y', left=False, labelleft=False, labelrotation=45)
            elif (col==0) & (row!=nparams-1):
                ax.tick_params('x', bottom=False, labelbottom=False, labelrotation=45)
            elif (row!=0) and (col!=0):
                ax.tick_params('both', bottom=False, labelbottom=False, left=False, labelleft=False, labelrotation=45)
            axes[row,col] = ax

        elif row==col: #diagonal
            if row==nparams-1:
                ax = fig.add_subplot(grid[row, col])
            else:
                ax = fig.add_subplot(grid[row, col],sharex=axes[nparams-1,col])
            #labels
            if row==nparams-1:
                ax.tick_params('both', bottom=True, labelbottom=True, left=False, labelleft=False, right=True, labelright=True)#, labelrotation=45)
            elif row==0:
                ax.tick_params('both', bottom=False, labelbottom=False, left=True, labelleft=True)#, labelrotation=45)
            else:
                ax.tick_params('both', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)#, labelrotation=45)
            axes[row,col] = ax
        elif row<col:
            ax = None

#################################plotting section#################################################################

xlims = np.empty(shape=nparams-1, dtype=object)
for row in range(nparams-1,-1,-1):
    for col in range(nparams):
        for i,sample in enumerate(samples_list):
            paramNames = list(sample.columns[:nparams])#
            y = paramNames[row]
            x = paramNames[col]
            ax = axes[row,col]
            if row>col:
                if (ax!=None) & (x!=y):
                    ymin, ymax = sample.limits.get(y, (None, None))
                    xmin, xmax = sample.limits.get(x, (None, None))
                    nbins_x = bins if x not in discrete_params.keys() else discrete_params_bins[x]
                    nbins_y = bins if y not in discrete_params.keys() else discrete_params_bins[y]
                    pdf, xx, yy = np.histogram2d(sample[x], sample[y], weights=sample.weights, #np.histogram2d
                                                   bins=[nbins_x,nbins_y],
                                                 range=[[xmin, xmax], [ymin, ymax]],
                                                 #density=False
                                                )
                    xbins = xx[:-1] + (xx[1] - xx[0]) / 2 #middle of bins
                    ybins = yy[:-1] + (yy[1] - yy[0]) / 2 #middle of bins
                    #levels for contours
                    smooth_data = gaussian_filter(pdf, sigma_smooth) if sigma_smooth!=0 else pdf
                    S = np.array([smooth_data.max(), *np.sort(smooth_data.flatten())[::-1], smooth_data.min()])
                    invCDF = np.cumsum(S)/S.sum()
                    invcdf = sip.interp1d(invCDF, S)
                    levels=[.95][::-1]
                    levels_values = invcdf(levels)
                    if x and y not in discrete_params.keys():
                        color = matplotlib.colors.rgb2hex(ccb[i], keep_alpha=True)
                        cs = ax.contour(*np.meshgrid(xbins, ybins),smooth_data.T,
                                        extent=[xx.min(),xx.max(),yy.min(),yy.max()],
                                        linewidths=2, linestyles="dashed", alpha=0.7,levels=levels_values, colors=color)
                    if (i==0):
                        cmap = anesthetic.plot.basic_cmap( color )
                        image = ax.pcolormesh(xx, yy, pdf.T, cmap=cmap, vmin=0)
                        if col==0:
                            ax.set_ylabel(texDict[y],fontsize=16)
                        if row==nparams-1:
                            ax.set_xlabel(texDict[x],fontsize=16)
            if row==col:
                nbins_x = bins if x not in discrete_params.keys() else discrete_params_bins[x]
                h, edges, bars = ax.hist(sample[x], #color=color,
                                         #range=(xmin, xmax),
                                         density=True,
                                         bins=nbins_x,
                                         histtype="step", weights=sample.weights)
                if row!=0:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
                if x not in discrete_params.keys():
                    hist_x = np.array(edges + (np.roll(edges,-1) - edges)/2)[:-1]
                    hist_y = gaussian_filter1d(h, sigma_smooth) if sigma_smooth!=0 else h
                    ax.plot(hist_x,hist_y, c = ccb[i], lw=2)
                if row==nparams-1:
                    ax.set_xlabel(texDict[x], fontsize=16)


ax = fig.add_subplot(grid[0, 1])
ax.axis("off")
l = ["","",""]
ax.legend(
    handles=[mlines.Line2D([], [], lw=4, linestyle="solid", color=ccb[i], label=name.split("_")[-1]+l[i]) for i,name in enumerate(files)],
    loc='upper left',fontsize=16,frameon=False,
)
plt.show()

#plt.savefig("IDR4_triangle.png", bbox_inches="tight")

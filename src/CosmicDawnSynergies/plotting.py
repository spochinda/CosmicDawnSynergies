import matplotlib
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import scipy.interpolate as sip
import numpy as np
import anesthetic
import seaborn as sns

def create_triangle_axes(nparams, figsize=(18,18)):
    fig = plt.figure(figsize=figsize)

    grid = GS(nparams, nparams, hspace=0, wspace=0)

    axes = np.full((nparams, nparams), False, dtype=object)#np.empty(shape=(nparams, nparams), dtype=object) #[]

    shared = {}
    for row in range(nparams-1,-1,-1):
        for col in range(nparams):
            if row>col:
                shared["sharex"]=False
                shared["sharey"]=False

                if col > 0:
                    if row==nparams-1: #bottom row
                        shared["sharey"]=axes[nparams-1,0]
                    elif row<nparams-1:
                        shared["sharey"]=axes[row,0]
                        shared["sharex"]=axes[nparams-1,col]
                if col==0:
                    if row<nparams-1:
                        shared["sharex"]=axes[nparams-1,0]
                
                if shared["sharex"]==False:
                    shared.pop("sharex")
                if shared["sharey"]==False:
                    shared.pop("sharey")

                axes[row,col] = fig.add_subplot(grid[row, col], **shared)
            elif row==col:
                if row==nparams-1:
                    axes[row,col] = fig.add_subplot(grid[row, col])
                else:
                    axes[row,col] = fig.add_subplot(grid[row, col],sharex=axes[nparams-1,col])
            
            if axes[row,col]!=False:
                #labels
                axes[row,col].tick_params('both',
                            left=True if (col==0) and (row!=0) else False, 
                            labelleft=True if (col==0) and (row!=0) else False, 
                            bottom = True, #if row==nparams-1 else False,
                            labelbottom = True if row==nparams-1 else False,
                            right=True if row==col else False, 
                            labelright=True if row==col else False,
                            #labelrotation=0 if (col==row) and (row!=nparams-1) else 45,
                            )
                #axes[row,col].text(0.5, 0.5, f'{row},{col}', transform=axes[row,col].transAxes,
                #        fontsize=30, color='red', alpha=0.5,
                #        ha='center', va='center')
    return fig, axes, grid

def axes_triangle_plot(files : list,
                       fig : matplotlib.figure.Figure,
                       axes : np.ndarray,
                       grid : GS,
                       paramNames : list,
                       plot_path : str = "",
                       **kwargs):
    labels = [file.split("/")[-2].replace("_", " ") for file in files]
    labels = kwargs.get("labels", labels)
    plot_2d_idx = kwargs.get("plot_2d_idx", 0)
    contour_idx = kwargs.get("contour_idx", [0,])
    levels = kwargs.get("levels", [0.95,])
    bins = kwargs.get("bins", 20)
    sigma_smooth = kwargs.get("sigma_smooth", 1.25)
    
    assert len(files)==len(labels), "Number of files and labels must match"

    nparams = len(paramNames)
    
    samples_list = [anesthetic.read_chains(root=file)[paramNames] for file in files]
    
    paramNames = list(samples_list[0].columns)
    
    ############# hardcoded discrete params #############
    discrete_params = {"alpha": [-1, -1.3, -1.5],
                    "nu_0": [*range(100,1500,100), 1500, 2000, 3000],
                    "pop": [231, 232, 233]
                    }
    discrete_params_bins = {
                "alpha": np.arange(-0.5,len(discrete_params["alpha"]),1), #[-1, -1.3, -1.5],
                "nu_0": np.arange(-0.5,len(discrete_params["nu_0"]),1),#[*range(100,1500,100), 1500, 2000, 3000],
                "pop": np.arange(-0.5,len(discrete_params["pop"]),1),#[231, 232, 233],
                }
    #####################################################
    
    ccb = np.roll(sns.color_palette("colorblind")[:len(files)], shift=0, axis=0)

    samples_min = np.array([sample.values[:,:nparams].min(axis=0) for sample in samples_list]).min(axis=0)
    samples_max = np.array([sample.values[:,:nparams].max(axis=0) for sample in samples_list]).max(axis=0)
    bin_edges = np.array([np.linspace(min_,max_,bins+1) for min_,max_ in zip(samples_min, samples_max)])
    bin_centers = np.array([np.array(edges[:-1] + (edges[1] - edges[0]) / 2) for edges in bin_edges])

    for i,sample in enumerate(samples_list):
        for row, ax_row in zip(range(nparams-1,-1,-1), axes[::-1]):
            for col, ax in zip(range(nparams), ax_row):
                y = paramNames[row]
                x = paramNames[col]
                nbins_x = bin_edges[col] if x not in discrete_params.keys() else discrete_params_bins[x]
                nbins_y = bin_edges[row] if y not in discrete_params.keys() else discrete_params_bins[y]
                xmin, xmax = nbins_x.min(), nbins_x.max()
                ymin, ymax = nbins_y.min(), nbins_y.max()
                
                #2d marginals
                if row>col:
                    pdf, _, _ = np.histogram2d(x=sample[x].values, y=sample[y].values, 
                                                    weights=sample.get_weights(), 
                                                    bins=[nbins_x,nbins_y], 
                                                )                
                    
                    #colormesh
                    color = matplotlib.colors.rgb2hex(ccb[i], keep_alpha=True)
                    if i==plot_2d_idx:
                        cmap = anesthetic.plot.basic_cmap( color )
                        image = ax.pcolormesh(nbins_x, nbins_y, pdf.T, cmap=cmap, vmin=0)
                    
                    #2d contours
                    if (i in contour_idx) and (x not in discrete_params.keys()) and (y not in discrete_params.keys()):
                        smooth_data = gaussian_filter(pdf, sigma_smooth) if sigma_smooth!=0 else pdf
                        S = np.array([smooth_data.max(), *np.sort(smooth_data.flatten())[::-1], smooth_data.min()])
                        invCDF = np.cumsum(S)/S.sum()
                        invcdf = sip.interp1d(invCDF, S)
                        levels_values = invcdf(levels)
                        #interpolate contours at bin centers to bin edges
                        dx = bin_centers[col][1]-bin_centers[col][0]
                        dy = bin_centers[row][1]-bin_centers[row][0]
                        smooth_data = np.pad(smooth_data, pad_width=1, mode="reflect")
                        x_centers = np.array([bin_centers[col][0]-dx, *bin_centers[col], bin_centers[col][-1]+dx])
                        y_centers = np.array([bin_centers[row][0]-dy, *bin_centers[row], bin_centers[row][-1]+dy])
                        interpolator = sip.RegularGridInterpolator((x_centers, y_centers), smooth_data, bounds_error=False, fill_value=None)
                        x_edges, y_edges = np.meshgrid(bin_edges[col], bin_edges[row])
                        smooth_data = interpolator((x_edges, y_edges))
                        cs = ax.contour(bin_edges[col], bin_edges[row], smooth_data, linewidths=3, linestyles="dashed", alpha=0.7,levels=levels_values, colors=color) 
                    
                    if col==0:
                        ax.set_ylabel(y[1])
                    
                    if row==nparams-1:
                        ax.set_xlabel(x[1])

                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                
                #1d marginals
                if row==col:
                    h, _, _ = ax.hist(sample[x], color=ccb[i], density=True, bins=nbins_x, histtype="step", weights=sample.get_weights(), alpha=0.4)

                    #1d contours
                    if x not in discrete_params.keys(): 
                        h = gaussian_filter1d(h, sigma_smooth) if sigma_smooth!=0 else h
                        h = np.interp(bin_edges[col], bin_centers[col], h)
                        ax.plot(bin_edges[col], h, color=ccb[i], lw=3, zorder=-i)
                        
                    ax.set_ylabel("PDF")#, labelrotation=45)
                    ax.yaxis.set_label_position("right")    
                    if row==nparams-1:
                        ax.set_xlabel(x[1])#, fontsize=16)
                    
                    ax.set_xlim(xmin, xmax)



    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")

    ax.legend(
        handles=[mlines.Line2D([], [], lw=4, linestyle="solid", color=ccb[i], label=label) for i,label in enumerate(labels)],# + [mlines.Line2D([], [], lw=4, linestyle="dashed", color="k",alpha=0.5, label="Prior 68\\%")], 
        loc='upper left', bbox_to_anchor=(0.4,1), #fontsize=16,
        frameon=False, 
    )

    plt.savefig(plot_path, bbox_inches="tight")

    plt.close()

    return fig, axes, grid

def triangle_plot(files, paramNames, plot_path="", **kwargs):
    figsize = kwargs.get("figsize", (18,18))
    fig, axes, grid = create_triangle_axes(nparams=len(paramNames), figsize=figsize)
    fig, axes, grid = axes_triangle_plot(files, fig, axes, grid, paramNames, plot_path, **kwargs)
    return fig, axes, grid


if __name__ == "__main__":
    plt.rcParams["font.size"]=22 #30

    files = ["/Users/simonpochinda/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies/scripts/non-public/LikelihoodXRB_LikelihoodRadioBackground_LikelihoodHERA/run",]

    paramNames = ["log10fstarII", "log10fstarIII", "log10Vc", "log10fX", "tau", "log10fradio"]

    triangle_plot(files, paramNames)
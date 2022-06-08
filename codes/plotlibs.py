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
params = {'legend.fontsize':  11,
          'figure.figsize': (6, 5),
         'axes.labelsize':  11,
         'axes.titlesize': 11,
         'xtick.labelsize': 9,
         'ytick.labelsize': 9}
plt.rcParams.update(params)

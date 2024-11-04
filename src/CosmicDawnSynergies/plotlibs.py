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
cdefault = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Matplotlib settings
params = {'legend.fontsize':  12,
          'figure.figsize': (6, 5),
         'axes.labelsize':  14,
         'axes.titlesize': 14,
         'xtick.labelsize': 12,
         'ytick.labelsize': 12}
plt.rcParams.update(params)

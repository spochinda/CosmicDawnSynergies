from copy import deepcopy
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi, log, sqrt
from os.path import isfile
from os import makedirs
from scipy import special
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import anesthetic
import hera_pspec as hp # HERA-Stack
import joblib
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.interpolate as sip
import scipy.stats as sst
import matplotlib.colors as mpc


def benchmark(PS_of_params, test_PS, test_params):
    # Main benchmark z = 8 at k = 0.192 h/Mpc
    # Other benchmark: "all 78 numbers from old emulator",
    # or all HERA-k at all HERA-z, or something likelihood
    # related such as L_true vs L_emu.
    # Also consider TS emulator etc.
    raise NotImplementedError

# The main power spectrum emulator, also for RSD ratios
class poweremu():
    def __init__(self, loadfile=None, hidden_layer_sizes=None, preprocesss_log_x=True, offset=1, max_iter=10, **kwargs):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (100,100,100,100)
        self.mlp = make_pipeline(StandardScaler(), MLPRegressor(
            # Changeable non-defaults
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
            # Mandatory non-defaults
            verbose=True, validation_fraction=0, warm_start=True,
            # Defaults
            #activation='relu', early_stopping=False, 
            #alpha=1e-4, solver='adam', learning_rate='constant',
            **kwargs))
        if preprocesss_log_x:
            self.preprocess_x = lambda x: np.log(x)
            self.inv_preprocess_x = lambda x: np.exp(x)
        else:
            self.preprocess_x = lambda x: x
            self.inv_preprocess_x = lambda x: x
        self.preprocess_y = lambda y: np.log(y+offset)
        self.inv_preprocess_y = lambda y: np.exp(y)-offset
        if loadfile is None:
            print("Not loading from file.")
        elif isfile(loadfile):
            self.load(loadfile)
        else:
            assert False, ("loadfile", loadfile, "not found.")
    def load(self, loadfile):
        self.mlp = joblib.load(loadfile)
        print("Loaded from", loadfile)
    def save(self, loadfile):
        joblib.dump(self.mlp, loadfile)
        print("Saved to", loadfile)
    def train(self, input_0, output_0):
        # Step 0: Take the ln of x
        input_1 = self.preprocess_x(input_0)
        # Step 1: Take the ln of y+1
        output_1 = self.preprocess_y(output_0)
        # Step 2: Train emulator incl. scaler
        return self.mlp.fit(input_1, output_1)
    def predict(self, x):
        single_point = True if len(np.shape(x))==1 else False
        x = [x] if single_point else x
        x1 = self.preprocess_x(x)
        y1 = self.mlp.predict(x1)
        y0 = self.inv_preprocess_y(y1)
        y0 = y0[0] if single_point else y0
        return y0

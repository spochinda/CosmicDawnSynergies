import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import copy
from collections import OrderedDict

import astropy.constants as c
import astropy.units as u
from scipy.interpolate import interp1d

from CosmicDawnSynergies.itamar.radio_cutoff_calc import H0


class SimpleDataset(Dataset):
    """Simple PyTorch Dataset that returns dict with 'params' and 'target' keys."""

    def __init__(self, params, targets):
        """
        Args:
            params: pandas DataFrame or numpy array of parameters
            targets: numpy array of targets
        """
        self.params = torch.FloatTensor(params.values if hasattr(params, 'values') else params)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {
            'params': self.params[idx],
            'target': self.targets[idx]
        }


def load_fn(file_path, key=None):
    """Load data from a given file path."""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.txt'):
        data = np.loadtxt(file_path)
    elif file_path.endswith('.mat'):
        data = loadmat(file_path)[key]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return data

class BaseDataset(Dataset):
    """Custom Dataset for Cosmic Dawn Synergies."""

    def __init__(self, opt: dict):
        """
        Args:
            opt (dict): Dictionary containing dataset options.
        """
        self.val_size = 0.2
        self.train_size = 0.8
        self.random_state = 42
        self.opt = opt
        self.params_opt = opt.get("params_opt", {})
        self.targets_opt = opt.get("targets_opt", {})
        self.data_dims = opt.get("data_dims", {})

        self.params = load_fn(self.params_opt.get("file"), self.params_opt.get("key", None))
        self.targets = load_fn(self.targets_opt.get("file"), self.targets_opt.get("key", None))

        self.discard_nan()
        self.params = pd.DataFrame(self.params, columns=self.params_opt.get("names"))
        if self.targets_opt.get("transform", False):
            transform_fn = globals().get(self.targets_opt["transform"])
            self.targets = transform_fn(self.targets, self.params, **self.targets_opt.get("transform_kwargs", {}))
        
        self.prepare_data()

        self.params_train, self.params_val, self.targets_train, self.targets_val = train_test_split(self.params, self.targets, test_size=self.val_size, train_size=self.train_size, random_state=self.random_state)
        
        dim_shapes = self.targets.shape[1:]
        for i,(key,shape) in enumerate(zip(self.data_dims.keys(), dim_shapes)):
            if self.data_dims[key].get("file") is not None:
                self.data_dims[key]["values"] = load_fn(self.data_dims[key]["file"], self.data_dims[key].get("key", None)).flatten().astype(np.float64)
            else:
                print(f"File for {key} not provided. Generating {shape} geomspace values from between {self.data_dims[key]['lims_nsample'][0]} and {self.data_dims[key]['lims_nsample'][1]}.")
                self.data_dims[key]["values"] = np.geomspace(*self.data_dims[key]["lims_nsample"][:2], shape).astype(np.float64)
            if self.data_dims[key].get("transform") is not None:
                transform_fn = globals().get(self.data_dims[key]["transform"])
                self.data_dims[key]["values"] = transform_fn(self.data_dims[key]["values"], **self.data_dims[key].get("transform_kwargs", {}))
        self.params_train, self.targets_train = gen_training_data(self.params_train, self.targets_train, copy.deepcopy(self.data_dims), n_jobs=-1, verbose=True)
        self.params_val, self.targets_val = prepare_validation_data(params=self.params_val, targets=self.targets_val, data_dims=copy.deepcopy(self.data_dims))

        #normalize parameters
        self.params_mins = self.params_train.min()
        self.params_maxs = self.params_train.max()
        self.params_means = self.params_train.mean()
        self.params_stds = self.params_train.std()
        self.params_norm = self.params_opt.get("normalization", "norm_minmax")
        self.params_train = getattr(self, self.params_norm)(self.params_train)
        self.params_val = getattr(self, self.params_norm)(self.params_val)

        self.build_param_stats()

        # Create PyTorch datasets for training and validation
        self.train_dataset = SimpleDataset(self.params_train, self.targets_train)
        self.val_dataset = SimpleDataset(self.params_val, self.targets_val)

    def norm_minmax(self, x):
        return (x - self.params_mins) / (self.params_maxs - self.params_mins)
    
    def norm_standard(self, x):
        return (x - self.params_means) / self.params_stds

    def norm_minmax_extended(self, x):
        return (x - self.params_mins) / (self.params_maxs - self.params_mins) * 2 - 1
    
    def build_param_stats(self):
        self.param_stats = OrderedDict()
        for param, minimum, maximum, mean, std in zip(self.params_train.columns, self.params_mins, self.params_maxs, self.params_means, self.params_stds):
            self.param_stats[param] = {}
            self.param_stats[param]["min"] = float(minimum)
            self.param_stats[param]["max"] = float(maximum)
            self.param_stats[param]["mean"] = float(mean)
            self.param_stats[param]["std"] = float(std)
            #self.param_stats[param]["prior"] = [float(self.params_train[param].min()), float(self.params_train[param].max())]

    def discard_nan(self):
        nan_indices = np.argwhere(np.isnan(self.targets))
        nan_indices = np.unique(nan_indices[:,0])
        print(f"Removing {len(nan_indices)} simulations from shape {self.targets.shape}")
        self.targets = np.delete(self.targets, nan_indices, axis=0)
        self.params = np.delete(self.params, nan_indices, axis=0)

    def prepare_data(self):
        """
        Prepares the input parameters by discarding specified parameters and log-transforming others.
        Also applies offset and log transformation to the target values if specified.
        """
        self.params = self.params.drop(columns=self.params_opt.get("discard", []))
        for param in self.params_opt.get("log", []):
            self.params[param] = np.log10(self.params[param])
            self.params.rename(columns={param: f"log10{param}"}, inplace=True)
        
        # Ensure targets are numeric and handle offset
        offset = self.targets_opt.get("offset", 0)
        if offset != 0:
            # Convert offset to float to ensure proper type handling
            offset = float(offset)
            # Ensure targets are float type
            self.targets = self.targets.astype(np.float64)
            self.targets = self.targets + offset
            
        if self.targets_opt.get("log", False):
            # Ensure targets are positive before log transform
            if np.any(self.targets <= 0):
                print(f"Warning: Found non-positive values in targets. Min value: {self.targets.min()}")
                # Add small epsilon to avoid log(0)
                self.targets = np.maximum(self.targets, 1e-10)
            self.targets = np.log10(self.targets)
        

def gen_training_data(params, targets, data_dims, n_jobs=-1, verbose=False, **kwargs):
    """
    Generate training data by interpolating along data dimensions.
    Parameters:
    -----------
    params : pd.DataFrame
        DataFrame containing the params for interpolation.
    targets : np.ndarray
        Array containing the target values to be interpolated.
    data_dims : dict
        Dictionary containing the data dimensions.
    n_jobs : int, optional
        Number of jobs to run in parallel (default is -1, which uses all available processors).
    verbose : bool, optional
        Boolean indicating whether to display progress information (default is False).
    Returns:
    --------
    params : pd.DataFrame
        DataFrame containing the interpolated params.
    data : np.ndarray
        Array containing the interpolated data.
    """
    params_columns = list(params.columns)
    params = params.to_numpy()

    if np.all([data_dims[k]['lims_nsample'][2] >= 1 for k in data_dims]):
        data_dims_columns = []
        data_dims_ = []
        lims_nsample = []
        for key in data_dims.keys():
            values = data_dims[key]["values"]
            lim_min = data_dims[key]["lims_nsample"][0]
            lim_max = data_dims[key]["lims_nsample"][1]

            # Validate that lims_nsample is within the data dimension bounds
            if lim_min < values.min() or np.isnan(lim_min):
                print(f"lims_nsample lower bound ({lim_min}) for '{key}' is below "
                f"data minimum ({values.min()}) or it is nan. Using data minimum instead.")
                lim_min = values.min()
            if lim_max > values.max() or np.isnan(lim_max):
                print(f"lims_nsample upper bound ({lim_max}) for '{key}' is above "
                f"data maximum ({values.max()}) or it is nan. Using data maximum instead.")
                lim_max = values.max()

            if data_dims[key]["log"]:
                # Use the prior limits from lims_nsample, applying log transform
                lims_nsample.append([
                    np.log10(lim_min),
                    np.log10(lim_max),
                    data_dims[key]["lims_nsample"][2]
                ])
                data_dims_columns.append(f"log10{key}")
                data_dims_.append(np.log10(values))
            else:
                # Use the updated limits with the number of samples
                lims_nsample.append([
                    lim_min,
                    lim_max,
                    data_dims[key]["lims_nsample"][2]
                ])
                data_dims_columns.append(key)
                data_dims_.append(values)

        results = Parallel(n_jobs=n_jobs)(
            delayed(random_grid_interpolation)(params=p, data_dims=data_dims_, data=data_i, lims_nsample=lims_nsample)
            for p, data_i in tqdm(zip(params, targets), total=len(params), desc="Interpolating", disable=not verbose)
        )
        params, targets = zip(*results)
        params = np.vstack(params)
        targets = np.hstack(targets)
        
        # section to disable parallel for easier debugging
        #for i,(p, data_i) in enumerate(tqdm(zip(params, targets), total=len(params), desc="Interpolating", disable=not verbose)):
        #    params_i, targets_i = random_grid_interpolation(params=p, data_dims=data_dims_, data=data_i, lims_nsample=lims_nsample)
        #    if i==0:
        #        params = params_i
        #        targets = targets_i
        #    else:
        #        params = np.vstack((params, params_i))
        #        targets = np.hstack((targets, targets_i))
    else:
        # No interpolation - stack raw data within lims_nsample bounds
        data_dims_columns = []
        data_dims_ = []
        masks = []
        for key in data_dims.keys():
            values = data_dims[key]["values"]
            lim_min = data_dims[key]["lims_nsample"][0]
            lim_max = data_dims[key]["lims_nsample"][1]
            if np.isnan(lim_min):
                lim_min = values.min()
            if np.isnan(lim_max):
                lim_max = values.max()

            # Create mask for values within bounds
            mask = (values >= lim_min) & (values <= lim_max)
            masks.append(mask)

            if data_dims[key]["log"]:
                data_dims_columns.append(f"log10{key}")
                data_dims_.append(np.log10(values[mask]))
            else:
                data_dims_columns.append(key)
                data_dims_.append(values[mask])

        # Apply masks to targets along each data dimension axis
        for i, mask in enumerate(masks):
            targets = np.compress(mask, targets, axis=i + 1)

        # Create meshgrid of filtered data dimension values
        grids = np.meshgrid(*data_dims_, indexing='ij')
        combinations = np.vstack([grid.ravel() for grid in grids]).T

        num_params = len(params)
        num_combinations = len(combinations)

        # Repeat params for each combination and tile combinations for each param
        params = np.repeat(params, num_combinations, axis=0)
        combinations = np.tile(combinations, (num_params, 1))
        params = np.hstack((combinations, params))

        # Flatten targets
        targets = targets.ravel()
    
    #print("Final params shape:", params.shape, "Final targets shape:", targets.shape)
    params = pd.DataFrame(params, columns=data_dims_columns + params_columns)

    return params, targets

def random_grid_interpolation(params, data_dims, data, lims_nsample):
    try:
        priors = [np.random.uniform(*var) for var in lims_nsample]

        priors = [np.sort(prior) for prior in priors]
        interp = RegularGridInterpolator(data_dims, data, method='linear')
        grids = np.meshgrid(*priors)
        interp = interp(tuple(grids))
        stacked = np.stack((*grids, interp), axis=-1)
        stacked = stacked.reshape(-1, stacked.shape[-1])
        params = np.tile(params, (stacked.shape[0], 1))
        interp = stacked[:,-1]
        params = np.hstack((stacked[:, :-1], params))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno, flush=True) 
        assert False
    return params, interp

def prepare_validation_data(params, targets, data_dims, **kwargs):
    """
    Flattens the given data and combines it with the parameters and data dimensions.

    Parameters:
    -----------
    parameters : pandas.DataFrame
        A DataFrame containing the parameters to be combined with the data.
    data : numpy.ndarray
        The data array to be flattened and combined with the parameters.
    data_opt : dict
        A dictionary containing the data options.
    Returns:
    --------
    parameters : pandas.DataFrame
        A DataFrame containing the flattened data combined with the parameters and data dimensions.
    data : numpy.ndarray
        The flattened data array.

    Raises:
    -------
    AssertionError
        If the length of the combined parameters and data does not match the length of the flattened data.
    """
    params_columns = list(params.columns)
    params = params.to_numpy()

    data_dims_columns = []
    data_dims_ = []
    lims = []
    for key in data_dims.keys():
        values = data_dims[key]["values"]
        lim_min = data_dims[key]["lims_nsample"][0]
        lim_max = data_dims[key]["lims_nsample"][1]

        # Handle nan values by using data min/max
        if np.isnan(lim_min):
            lim_min = values.min()
        if np.isnan(lim_max):
            lim_max = values.max()

        if data_dims[key]["log"]:
            lims.append([np.log10(lim_min), np.log10(lim_max)])
            data_dims_columns.append(f"log10{key}")
            data_dims_.append(np.log10(values))
        else:
            lims.append([lim_min, lim_max])
            data_dims_columns.append(key)
            data_dims_.append(values)

    #log lims
    masks = [np.logical_and(data_dim >= lim[0], data_dim <= lim[1]) for data_dim, lim in zip(data_dims_, lims)]
    drop_rows = [np.where(~mask)[0] for mask in masks]
    data_dims_ = [data_dim[mask] for data_dim, mask in zip(data_dims_, masks)]

    grids = np.meshgrid(*data_dims_, indexing='ij')
    combinations = np.vstack([grid.ravel() for grid in grids]).T

    num_params = len(params)
    num_combinations = len(combinations)

    params = np.repeat(params, num_combinations, axis=0)
    combinations = np.tile(combinations, (num_params, 1))
    params = np.hstack((combinations, params))

    params = pd.DataFrame(params, columns=data_dims_columns + params_columns)
    for i,drop_row in enumerate(drop_rows):
        targets = np.delete(targets, drop_row, axis=i+1)

    targets = targets.ravel()
    assert len(params) == len(targets), f"Length of parameter={params.shape} and data={targets.shape} do not match."

    return params, targets

def transform_dim_little_h(input, h=0.674, **kwargs):
    """
    Transform the input values by dividing by little h.

    Parameters:
    -----------
    input : numpy.ndarray
        The input array to be transformed.
    h : float, optional
        The value of little h (default is 0.674).

    Returns:
    --------
    output : numpy.ndarray
        The transformed array.
    """
    return input / h

def get_radio_sed(sed_type, power=-0.7):

    '''
    Load raw SED
    Eqn. 5 in "What does the first highly-redshifted 21-cm detection tell us about early galaxies?"
    Gives the valie at 150 MHz, extrapolate to lower frequencies with spectral index = -0.7
    '''
    if sed_type == 'power_law':
        nu = np.logspace(6,13,1000)
        sed = (nu/(150*10**6))**(power) # times frad times SFR/(m_solar yr-1)
        log_sed = 22 + np.log10(sed)
    else:
        nu, sed = 0,0
        
    return np.log10(nu), log_sed

def Hubble_const(z):
    # from itamar cosmo.json
    H0 = 67.04 
    Om = 0.3168681398488275
    #OLambda = 0.6831318601511724
    Hz = H0*np.sqrt(Om*(1+z)**3) # EdS
    return Hz

def transform_T_today(targets, params, z_file, z_key):
    '''
    Returns the T_radio today by integrating the redshifting SED across all zs,
    scaled by the SFR at the given z and frad.
    Note: modified from Itamar's code above to be much more efficient (x2000 faster)
    '''
    frad = params['fradio'].values
    z_array = load_fn(z_file, z_key)[0].astype(float)
    dz = abs(z_array[1] - z_array[0])
    
    nu_today = np.logspace(-2, 1.1, 100)*1e9*u.Hz
    log_nu, log_sed = get_radio_sed('power_law')
    log_sed_interp = interp1d(log_nu, log_sed, kind='linear')

    constants = 1/(8*np.pi*c.k_B) * (c.c**3/nu_today**2)
    A = 1/(Hubble_const(z_array)*u.km/u.s/u.Mpc) * 1/(1+z_array) * dz # dz/(H*(1+z))
    redshifted_sed = 10**log_sed_interp(np.log10(np.outer(1+z_array,nu_today.value))) * (u.W/u.Hz)

    T_at_z = constants[None, :] * A[:, None] * redshifted_sed * frad[:,None,None] * targets[:,:,None]/(u.Mpc**3)

    T_today = np.sum(T_at_z,axis=1).to_value(u.K)
    return T_today

if __name__ == "__main__":
    opt = {
        "params_opt": {
            "file": "/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat",
            "key": "parameters",
            "names": ['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'],
            "log": ['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'],
            "discard": ['zeta', 'feed', 'delay'],
        },
        "targets_opt": {
            "file": "/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Deltak_mat.mat",
            "key": "combined_Deltaks",
            "offset": 1e-6,
            "log": True,
        },
        "data_dims": {
            "z": {
                "file": "/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat",
                "key": "z21cm",
                "log": False,
                "lims_nsample": [6, 27, 20]
            },
            "k": {
                "file": "/Users/simonpochinda/Documents/PhD/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_k_mat.mat",
                "key": "ks",
                "log": True,
                "lims_nsample": [0.1, 0.99, 20],
                "transform": "transform_little_h",
            },
        },
    }
    
    dataset = BaseDataset(opt)

    
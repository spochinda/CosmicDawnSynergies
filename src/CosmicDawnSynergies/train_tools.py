import os
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import numpy as np
import torch.distributed
from tqdm import tqdm

import pandas as pd
import torch 
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
import time 
from collections import OrderedDict

import matplotlib.pyplot as plt



def random_grid_interpolation(parameters, data_dims, data, lims_nsample):
    #p = parameters[i]
    #data_dims = tuple of data axes e.g. (x (redshift), y (wavevector))
    #data = dsq[i]
    #lims_nsample = list of lists with options for each sample axes, min and max and num_samples for each parameter [[min, max, num_samples], [min, max, num_samples]]
    #var2 = wavenumber sample

    priors = [np.random.uniform(*var) for var in lims_nsample]
    
    #priors[0][0] = 10 # for testing dsq
    #priors[1][0] = np.log10(k_array[19]) # for testing dsq

    priors = [np.sort(prior) for prior in priors]
    interp = RegularGridInterpolator(data_dims, data, method='linear')
    grids = np.meshgrid(*priors)
    interp = interp(tuple(grids))
    stacked = np.stack((*grids, interp), axis=-1)
    stacked = stacked.reshape(-1, stacked.shape[-1])
    parameters = np.tile(parameters, (stacked.shape[0], 1))
    interp = stacked[:,-1]
    parameters = np.hstack((stacked[:, :-1], parameters))
    return parameters, interp



def gen_training_data(parameters, data_dims, data, lims_nsample, data_dims_log, data_log, n_jobs=-1, verbose=False):
    """
    Generate training data by interpolating along data dimensions.
    Parameters:
    -----------
    parameters : pd.DataFrame
        DataFrame containing the parameters for interpolation.
    data_dims : list of pd.DataFrame
        List of DataFrames containing the data dimensions.
    data : np.ndarray
        Array containing the data to be interpolated.
    lims_nsample : list of lists
        List of lists containing the limits and number of samples to interpolate for each data dimension.
    data_dims_log : list of bool
        List indicating whether to apply log10 transformation to each data dimension.
    data_log : bool
        Boolean indicating whether to apply log10 transformation to the data.
    n_jobs : int, optional
        Number of jobs to run in parallel (default is -1, which uses all available processors).
    verbose : bool, optional
        Boolean indicating whether to display progress information (default is False).
    Returns:
    --------
    parameters : pd.DataFrame
        DataFrame containing the interpolated parameters.
    data : np.ndarray
        Array containing the interpolated data.
    """
    
    parameters_columns = list(parameters.columns)
    parameters = parameters.to_numpy()

    data_dims_columns = [data_dim.columns[0] for data_dim in data_dims]
    data_dims_columns = [f"log10{data_dim}" if data_dim_log else data_dim for i, (data_dim, data_dim_log) in enumerate(zip(data_dims_columns, data_dims_log))]

    data_dims = [np.log10(data_dim) if data_dim_log else data_dim for data_dim, data_dim_log in zip(data_dims, data_dims_log)]
    data_dims = [data_dim.to_numpy().ravel() for data_dim in data_dims]

    data = np.log10(data) if data_log else data
    lims_nsample = [[np.log10(var[0]), np.log10(var[1]), var[2]] if data_dims_log[i] else var for i, var in enumerate(lims_nsample)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(random_grid_interpolation)(parameters=p, data_dims=data_dims, data=data_i, lims_nsample=lims_nsample)
        for p, data_i in tqdm(zip(parameters, data), total=len(parameters), desc="Interpolating", disable=not verbose)
    )

    parameters, data = zip(*results)
    parameters = np.vstack(parameters)
    data = np.hstack(data)

    parameters = pd.DataFrame(parameters, columns=data_dims_columns + parameters_columns)

    return parameters, data





def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"], flush=True)
    except:
        print("Did not find master address variable. Setting manually...", flush=True)
        os.environ["MASTER_ADDR"] = "localhost"

    os.environ["MASTER_PORT"] = "2595"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus? nccl for gpus

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, parameters, target, device="cpu", **kwargs):
        self.device = device
        self.target = torch.from_numpy(target).to(torch.float32).to(self.device)

        if isinstance(parameters, np.ndarray):
            self.parameters = torch.from_numpy(parameters).to(torch.float32).to(self.device)
        elif isinstance(parameters, pd.DataFrame):
            self.parameters = torch.from_numpy(parameters.to_numpy()).to(torch.float32).to(self.device)
        else:
            raise ValueError("parameters must be a numpy array or pandas DataFrame")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        parameters = self.parameters[idx]
        target = self.target[idx]

        #parameters = torch.from_numpy(parameters).to(torch.float32)
        #target = torch.tensor(target, dtype=torch.float32)        
        #parameters = parameters.to(self.device)
        #target = target.to(self.device)
        return parameters, target


class MLP(nn.Module):
    def __init__(self, in_dim = 11, hidden_dim = 100, n_hidden = 1, out_dim = 1, dropout = 0.2, use_norm_dropout = False, use_attn = True):
        super(MLP, self).__init__()
        
        self.use_attn = use_attn
        self.use_norm_dropout = use_norm_dropout

        self.bool = True
        if self.bool:
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_norm_dropout:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if self.use_norm_dropout:
                layers.append(nn.Dropout(dropout)) ###
            for _ in range(n_hidden):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.use_norm_dropout:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                if self.use_norm_dropout:
                    layers.append(nn.Dropout(dropout)) ###
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.fc1 = nn.Linear(1, 64)
            #self.norm1 = nn.LayerNorm(64)
            self.act1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            
            self.fc2 = nn.Linear(64, 128)
            #self.norm2 = nn.LayerNorm(128)
            self.act2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            
            transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=256, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2, enable_nested_tensor=False)
            
            self.fc3 = nn.Linear(128, 64)
            #self.norm3 = nn.LayerNorm(64)
            self.act3 = nn.ReLU()
            self.dropout3 = nn.Dropout(dropout)
            
            self.fc4 = nn.Linear(64,32)
            #self.norm4 = nn.LayerNorm(32)
            self.act4 = nn.ReLU()
            self.dropout4 = nn.Dropout(dropout)

            self.fc5 = nn.Linear(32, 16)
            #self.norm5 = nn.LayerNorm(16)
            self.act5 = nn.ReLU()
            self.dropout5 = nn.Dropout(dropout)

            self.fc6 = nn.Linear(16, 8)
            #self.norm6 = nn.LayerNorm(8)
            self.act6 = nn.ReLU()
            self.dropout6 = nn.Dropout(dropout)

            self.fc7 = nn.Linear(8, 4)
            #self.norm7 = nn.LayerNorm(4)
            self.act7 = nn.ReLU()
            self.dropout7 = nn.Dropout(dropout)

            self.fc8 = nn.Linear(4, 1)
            self.act8 = nn.ReLU()
            self.dropout8 = nn.Dropout(dropout)

            self.fc9 = nn.Linear(11, 8)
            #self.norm9 = nn.LayerNorm(8)
            self.act9 = nn.ReLU()
            self.dropout9 = nn.Dropout(dropout)

            self.fc10 = nn.Linear(8, 4)
            #self.norm10 = nn.LayerNorm(4)
            self.act10 = nn.ReLU()
            self.dropout10 = nn.Dropout(dropout)

            self.fc11 = nn.Linear(4, 1)

    def forward(self, x):
        if self.bool:
            x = self.mlp(x)
            x = x.squeeze(-1)
        else:
            x = x.unsqueeze_(-1)
            x = self.dropout1(self.act1(self.fc1(x))) #64
            x = self.dropout2(self.act2(self.fc2(x))) #128
            x = self.transformer(x) #128
            x = self.dropout3(self.act3(self.fc3(x))) #64
            x = self.dropout4(self.act4(self.fc4(x))) #32
            x = self.dropout5(self.act5(self.fc5(x))) #16
            x = self.dropout6(self.act6(self.fc6(x))) #8
            x = self.dropout7(self.act7(self.fc7(x))) #4
            x = self.dropout8(self.act8(self.fc8(x))) #1
            x = x.squeeze(-1)
            x = self.dropout9(self.act9(self.fc9(x))) #8
            x = self.dropout10(self.act10(self.fc10(x))) #4
            x = self.fc11(x) #1
            x = x.squeeze(-1)

        return x

class poweremu_torch(nn.Module):
    def __init__(self, 
                 network, network_opt, 
                 optimizer, optimizer_opt, 
                 train_opt, scale_opt, 
                 device="cpu"):
        super(poweremu_torch, self).__init__()
        self.device = device
        self.network = network #MLP
        self.network_opt = network_opt #dictionary of MLP args
        self.model = self.network(**self.network_opt).to(self.device)
        self.multi_gpu = torch.cuda.device_count() > 1 and self.device!="cpu"
        if self.multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index])
        self.optimizer_opt = optimizer_opt #dictionary of optimizer args
        self.optimizer = optimizer
        self.opt = optimizer(self.model.parameters(), **optimizer_opt)
        self.train_opt = train_opt
        self.scale_opt = scale_opt


        self.epoch = 0
        self.loss = []
        self.validation_loss = []
        
    def train(self, train_dataloader, validation_dataloader, **kwargs):
        epochs = self.train_opt.pop("epochs", 100)
        profiling = self.train_opt.pop("profiling", False)
        loss_fn = self.train_opt.pop("loss_fn", torch.nn.MSELoss())
        save_after_epochs = self.train_opt.pop("save_after_epochs", 5)
        save_progress_plots_path = self.train_opt.pop("save_progress_plots_path", False)
        save_model_path = self.train_opt.pop("save_model_path", None)
        
        parameters_validation = validation_dataloader.dataset.parameters
        target_validation = validation_dataloader.dataset.target

        if loss_fn == "HuberLoss":
            loss_fn = torch.nn.HuberLoss()
        elif loss_fn == "L1Loss":
            loss_fn = torch.nn.L1Loss()
        elif loss_fn == "KLDivLoss":
            loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        elif loss_fn == "MSELoss":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.MSELoss()
        
        if self.device.index == 0 or self.device.type=="cpu":
            print(self.model, flush=True)
        for e in range(self.epoch, self.epoch+epochs):
            stime = time.time()
            for i,(parameters_train,target_train) in enumerate(train_dataloader):
                
                if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_push("predict-loss-backward-step")
                
                self.model.train()
                self.opt.zero_grad()
                pred_train = self.model(parameters_train)
                loss = loss_fn(pred_train, target_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                
                if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_pop()
                

                with torch.no_grad():
                    self.model.eval()

                    if self.multi_gpu:  torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                                        
                    _loss = loss.clone().detach().cpu().item()
                    pred_validation = self.model(parameters_validation)
                    _vresidlog = pred_validation - target_validation
                    _vrmselog = 10**torch.sqrt(torch.mean(torch.square(_vresidlog)))

                    if self.multi_gpu:  torch.distributed.all_reduce(_vrmselog, op=torch.distributed.ReduceOp.AVG)

                    _vrmselog = _vrmselog.clone().detach().cpu().item()
                
                    self.loss.append(_loss)
                    self.validation_loss.append(_vrmselog)
                    
                    _residlog = pred_train - target_train
                    _rmselog = 10**torch.sqrt(torch.mean(torch.square(_residlog))).detach().item()
                    _q95log = 10**torch.quantile(torch.sqrt(torch.square(_residlog)), 0.95).item()
                    _percentlog = 100 * torch.abs(_residlog) / target_train
                    _percentlog = _percentlog[~torch.isinf(_percentlog)]
                    _percentlogq95 = torch.nanquantile(_percentlog, 0.95)
                    
                    if self.device.index == 0 or self.device.type=="cpu":
                        
                        print(f"[{str(self.device)}] "
                              f"Epoch {e} | "
                              f"Batch {i+1} out of {train_dataloader.__len__()} | "
                              f"Time: {time.time()-stime:.2f} | "
                              f"Train: RMSE={_rmselog:.2f} mK2 q95<={_q95log:,.2f} mK2, pct<={_percentlogq95:,.2f}% | "
                              f"Validation: {_vrmselog:,.2f} mK^2" 
                              ,flush=True)
                        
                        if (_vrmselog == min(self.validation_loss)) and (self.device.index == 0 or self.device.type=="cpu") and (e >= save_after_epochs):
                            print(f"Saving model with validation loss: {_vrmselog:,.2f}", flush=True)
                            self.save_network(save_model_path)
                            self.epoch = e

                        if save_progress_plots_path:

                            fig, axes = plt.subplots(1,2, figsize=(12,6))
                            hist, edges = np.histogram(_residlog.abs().detach().cpu().numpy(), bins=100)
                            axes[0].bar(edges[:-1], hist, width=np.diff(edges))
                            axes[0].grid()
                            axes[0].set_xlabel('absResiduals')
                            axes[0].set_ylabel('Frequency')
                            hist, edges = np.histogram(_percentlog.detach().cpu().numpy(), bins=100)
                            axes[1].bar(edges[:-1], hist, width=np.diff(edges))
                            axes[1].grid()
                            axes[1].set_xlabel('Percent Error')
                            axes[1].set_ylabel('Frequency')                            
                            
                            save_path = os.path.join(save_progress_plots_path, "residuals_perc_histogram.png")
                            plt.savefig(save_path)
                            plt.close()

                            fig, axes = plt.subplots(1,1, figsize=(12,6))
                            bin_min = min(pred_validation.detach().cpu().numpy().min(), target_validation.detach().cpu().numpy().min())
                            bin_max = max(pred_validation.detach().cpu().max(), target_validation.detach().cpu().max())
                            bins = np.linspace(bin_min, bin_max, 100)
                            hist, edges = np.histogram(pred_validation.detach().cpu().numpy(), bins=bins)
                            axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
                            hist, edges = np.histogram(target_validation.detach().cpu().numpy(), bins=bins)
                            axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
                            #axes.hist(target.detach().cpu().numpy(), bins=100, alpha=0.5, label='Targets')
                            axes.set_xlabel('logDelta^2')
                            axes.set_ylabel('Frequency')
                            axes.legend()
                            
                            save_path = os.path.join(save_progress_plots_path, "validation_progress_hist.png")
                            plt.savefig(save_path)
                            plt.close()




    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.opt.state_dict(),
                    optimizer_opt = self.optimizer_opt,
                    train_opt = self.train_opt,
                    scale_opt = self.scale_opt,
                    loss = self.loss,
                    validation_loss = self.validation_loss,
                    epoch = self.epoch,
                    ),
                    f = path
                    )
        else:
            if str(self.device) == "cuda:0":
                #print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        network_opt = self.network_opt,
                        model = self.model.module.state_dict(), 
                        optimizer = self.opt.state_dict(),
                        optimizer_opt = self.optimizer_opt,
                        train_opt = self.train_opt,
                        scale_opt = self.scale_opt,
                        loss = self.loss,
                        validation_loss = self.validation_loss,
                        epoch = self.epoch,
                        ),
                        f = path
                        )

    def load_network(self, path):
        loaded_state = torch.load(path, map_location=self.device)
        self.network_opt = loaded_state['network_opt']
        self.model = self.network(**self.network_opt)
        self.model.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        try:
            self.optimizer_opt = loaded_state['optimizer_opt']
        except:
            self.optimizer_opt = {}
        self.opt = self.optimizer(self.model.parameters(), **self.optimizer_opt)
        self.opt.load_state_dict(loaded_state['optimizer'])
        self.train_opt = loaded_state['train_opt']
        self.scale_opt = loaded_state['scale_opt']
        self.loss = loaded_state['loss']
        self.validation_loss = loaded_state['validation_loss']
        try:
            self.epoch = loaded_state['epoch']
        except:
            pass
             
def prepare_parameters(parameters, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay']):
    """
    Prepares the input parameters by discarding specified parameters and log-transforming others.
    Args:
        parameters (pd.DataFrame): The input parameters to be prepared.
        transform_params (list, optional): The parameters to be log-transformed. Defaults to ['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'].
        discard_params (list, optional): The parameters to be discarded. Defaults to ['zeta', 'feed', 'delay'].
    Returns:
        pd.DataFrame: The prepared parameters with specified parameters discarded and others log-transformed.
    """
    
    parameters = parameters.drop(columns=discard_params)
    for param in transform_params:
        parameters[param] = np.log10(parameters[param])
        parameters.rename(columns={param: f"log10{param}"}, inplace=True)

    return parameters

def train_model(rank, multi_gpu, parameters_train, target_train, parameters_validation, target_validation, network_opt, optimizer_opt, train_opt, scale_opt, **kwargs):
    batch_size = train_opt.get("batch_size", 10000)

    if multi_gpu:
        world_size = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank}")
        ddp_setup(rank, world_size=world_size)
    elif not multi_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")
    
    train_data_module = Dataloader(parameters=parameters_train, target=target_train, device=device)#, data_dims=data_dims, data_dims_log=data_dims_log, vars=vars, data_log=data_log)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data_module, shuffle=True, seed=0) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=(train_sampler is None), sampler = train_sampler,)

    validation_data_module = Dataloader(parameters=parameters_validation, target=target_validation, device=device)#, data_dims=data_dims, data_dims_log=data_dims_log, vars=vars, data_log=data_log)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=validation_data_module, shuffle=True, seed=0) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=(validation_sampler is None), sampler = validation_sampler,)


    emu = poweremu_torch(network=MLP, network_opt=network_opt,
                         optimizer=torch.optim.Adam, optimizer_opt=optimizer_opt,
                         train_opt=train_opt, scale_opt=scale_opt,
                         device=device)


    #train
    if train_opt["profiling"] and torch.cuda.is_available():
        with torch.autograd.profiler.emit_nvtx():
            emu.train(train_dataloader, validation_dataloader)
    else:
        emu.train(train_dataloader, validation_dataloader)

    if multi_gpu:
        torch.distributed.barrier()
        destroy_process_group()

    
def prepare_validation_data(parameters, data, data_dims, data_dims_log, lims, data_log):
    """
    Flattens the given data and combines it with the parameters and data dimensions.

    Parameters:
    -----------
    parameters : pandas.DataFrame
        A DataFrame containing the parameters to be combined with the data.
    data : numpy.ndarray
        The data array to be flattened and combined with the parameters.
    data_dims : list of pd.DataFrame
        List of DataFrames containing the data dimensions.
    data_dims_log : list of bool
        List indicating whether to apply log10 transformation to each data dimension.
    lims : list of lists
        List of lists containing the limits for each data dimension.
    data_log : bool
        Boolean indicating whether to apply log10 transformation to the data.    

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
    data = np.log10(data) if data_log else data

    parameters_columns = list(parameters.columns)
    parameters = parameters.to_numpy()
    data_dims_columns = [data_dim.columns[0] for data_dim in data_dims]
    data_dims_columns = [f"log10{data_dim}" if data_dim_log else data_dim for i, (data_dim, data_dim_log) in enumerate(zip(data_dims_columns, data_dims_log))]

    
    #log lims
    masks = [np.logical_and(data_dim >= lim[0], data_dim <= lim[1]) for data_dim, lim in zip(data_dims, lims)]
    drop_rows = [np.where(~mask)[0] for mask in masks]
    data_dims = [data_dim.drop(drop_row) for drop_row,data_dim in zip(drop_rows,data_dims)]
    #data_dims = [data_dim[mask] for data_dim, mask in zip(data_dims, masks)]
    data_dims = [np.log10(data_dim) if data_dim_log else data_dim for data_dim, data_dim_log in zip(data_dims, data_dims_log)]
    data_dims = [data_dim.to_numpy().ravel() for data_dim in data_dims]
    
    grids = np.meshgrid(*data_dims, indexing='ij')
    combinations = np.vstack([grid.ravel() for grid in grids]).T

    num_parameters = len(parameters)
    num_combinations = len(combinations)

    parameters = np.repeat(parameters, num_combinations, axis=0)
    combinations = np.tile(combinations, (num_parameters, 1))
    parameters = np.hstack((combinations, parameters))

    parameters = pd.DataFrame(parameters, columns=data_dims_columns + parameters_columns)

    for i,drop_row in enumerate(drop_rows):
        data = np.delete(data, drop_row, axis=i+1)

    data = data.ravel()
    assert len(parameters) == len(data), f"Length of parameter={parameters.shape} and data={data.shape} do not match."

    return parameters, data

def uniform_rebin_data(data):
    """
    Rebin data uniformly into a specified number of samples.
    The number of samples in each bin is determined by the number of samples in the smallest bin.

    Parameters:
    data (numpy.ndarray): The input data to be rebinned.

    Returns:
    list: Indices of the rebinned data.
    
    Usage:
    indices = rebin_data(data)
    data = data[indices]
    parameters = parameters[indices]
    """
    bin_min = data.min()
    bin_max = data.max()
    bins = np.array([bin_min, *np.linspace(0., 5, 19), bin_max])
    bin_indices = np.digitize(data, bins)
    bin_nmin = np.array([np.sum(bin_indices == i) for i in range(1, len(bins))]).min()
    indices = []
    for i in range(1, len(bins)):
        indices.extend(np.random.choice(np.where(bin_indices == i)[0], bin_nmin, replace=False))
    return indices

def prepare_scale_opt(parameters, method):        
    mean = parameters.mean(axis=0)
    std = parameters.std(axis=0)
    minimum = parameters.min(axis=0)
    maximum = parameters.max(axis=0)

    parameter_names = parameters.columns
    scale_opt = OrderedDict()
    for parameter, min_, max_, mean_, std_, in zip(parameter_names, minimum, maximum, mean, std):
        method_ = method.pop(parameter, "standardize")
        scale_opt[parameter] = dict(method = method_, stats = dict(minimum = min_, maximum = max_, mean = mean_, std = std_))
    
    if method.keys():
        print(f"Warning: Unused keys in method dictionary: {method.keys()}")

    return scale_opt

class Scaler:
    def __init__(
            self, 
            scale_opt):  

        self.scale_opt = scale_opt

    def standardize(self, data, **kwargs):
        minimum = kwargs.pop("minimum", data.min())
        maximum = kwargs.pop("maximum", data.max())
        return (2 * (data - minimum) / (maximum - minimum)) - 1

    def normalize(self, data, **kwargs):
        mean = kwargs.pop("mean", data.mean())
        std = kwargs.pop("std", data.std())
        return (data - mean) / std
    
    def identity(self, data):
        return data

    def inverse_standardize(self, data, minimum, maximum):
        return 0.5 * (data + 1) * (maximum - minimum) + minimum

    def inverse_normalize(self, data, mean, std):
        return (data * std) + mean

    def inverse_identity(self, data):
        return data
    
    def transform(self, data, use_scale_opt = True, **kwargs):
        """
        data: pd.DataFrame
        """
        if not use_scale_opt:
            minimum = kwargs.pop("minimum", data.min(axis=0))
            maximum = kwargs.pop("maximum", data.max(axis=0))
            mean = kwargs.pop("mean", data.mean(axis=0))
            std = kwargs.pop("std", data.std(axis=0))

        params = data.columns
        n_sim = len(data)
        n_params = len(params)
        
        scale_opt_keys = np.array([*self.scale_opt.keys()])
        scale_opt_key_in_params = np.array([key in params for key in scale_opt_keys])
        missing_keys = scale_opt_keys[~scale_opt_key_in_params]
        
        assert len(missing_keys) == 0, f"Missing keys in data: {missing_keys}"
        assert n_params == len(scale_opt_keys), "number of features and number of transforms in scale_opt must be the same"
        
        for i,key in enumerate(scale_opt_keys):
            if self.scale_opt[key]["method"] == 'standardize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"minimum": minimum[i], "maximum": maximum[i]}
                data[key] = self.standardize(data[key], **stats)
            elif self.scale_opt[key]["method"] == 'normalize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"mean": mean[i], "std": std[i]}
                data[key] = self.normalize(data[key], **stats)
            else:
                data[key] = self.identity(data[key])
        return data
    
    def inverse_transform(self, data, use_scale_opt = True, **kwargs):
        minimum = kwargs.pop("minimum", data.min(axis=0))
        maximum = kwargs.pop("maximum", data.max(axis=0))
        mean = kwargs.pop("mean", data.mean(axis=0))
        std = kwargs.pop("std", data.std(axis=0))

        n_sim, n_params = data.shape
        assert n_params == len(self.axes_transform), "Length of data and transform must be the same"
        for i,key in self.scale_opt.keys():
            if self.scale_opt[key]["method"] == 'standardize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"minimum": minimum[i], "maximum": maximum[i]}
                data[:,i] = self.inverse_standardize(data[:,i], **stats)
            elif self.scale_opt[key]["method"] == 'normalize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"mean": mean[i], "std": std[i]}
                data[:,i] = self.inverse_normalize(data[:,i], **stats)
            else:
                data[:,i] = self.inverse_identity(data[:,i])
        return data
    
def shuffle_data(parameters, data):
    """
    Shuffle the given parameters and data in unison.

    This function shuffles the rows of the `parameters` DataFrame and the corresponding
    elements in the `data` array using the same random permutation.

    Parameters:
    parameters (pd.DataFrame): The DataFrame containing the parameters to be shuffled.
    data (np.ndarray): The array containing the data to be shuffled.

    Returns:
    tuple: A tuple containing the shuffled `parameters` DataFrame and the shuffled `data` array.
    """
    indices = np.random.permutation(parameters.index)
    parameters = parameters.reindex(indices).reset_index(drop=True)
    data = data[indices]
    return parameters, data

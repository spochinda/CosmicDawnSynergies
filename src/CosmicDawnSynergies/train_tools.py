import os
import sys
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
import CosmicDawnSynergies.models as models



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



def gen_training_data(parameters, data, data_opt, n_jobs=-1, verbose=False, **kwargs):
    """
    Generate training data by interpolating along data dimensions.
    Parameters:
    -----------
    parameters : pd.DataFrame
        DataFrame containing the parameters for interpolation.
    data : np.ndarray
        Array containing the data to be interpolated.
    data_opt : dict
        Dictionary containing the data options.
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

    data_dims_columns = []
    data_dims = []
    lims_nsample = []
    for dim in data_opt["data_dims"]:
        key = list(dim.keys())[0]
        min, max = dim[key]["lims"]
        nsample = dim[key]["nsample"]
        if dim[key]["log"]:
            data_dims_columns.append(f"log10{key}")
            data_dims.append(np.log10(dim[key]["values"]))
            lims_nsample.append([float(np.log10(min)), float(np.log10(max)), nsample])
        else:
            data_dims_columns.append(key)
            data_dims.append(dim[key]["values"])
            nsample = dim[key]["nsample"]
            lims_nsample.append([min, max, nsample])



    #data_dims_columns = [data_dim.columns[0] for data_dim in data_dims]
    #data_dims_columns = [f"log10{data_dim}" if data_dim_log else data_dim for i, (data_dim, data_dim_log) in enumerate(zip(data_dims_columns, data_dims_log))]

    #data_dims = [np.log10(data_dim) if data_dim_log else data_dim for data_dim, data_dim_log in zip(data_dims, data_dims_log)]
    #data_dims = [data_dim.to_numpy().ravel() for data_dim in data_dims]

    data = np.log10(data) if data_opt["data_log"] else data

    

    #lims_nsample = [[np.log10(var[0]), np.log10(var[1]), var[2]] if data_dims_log[i] else var for i, var in enumerate(lims_nsample)]

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

    os.environ["MASTER_PORT"] = "2599"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus? nccl for gpus

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, parameters, target, device="cpu", **kwargs):
        self.device = device
        # Keep on CPU — moving to device per-batch in the training loop avoids
        # the MPS/CUDA collation bottleneck (torch.stack on device tensors is slow).
        self.target = torch.from_numpy(target).to(torch.float32)

        if isinstance(parameters, np.ndarray):
            self.parameters = torch.from_numpy(parameters).to(torch.float32)
        elif isinstance(parameters, pd.DataFrame):
            self.parameters = torch.from_numpy(parameters.to_numpy()).to(torch.float32)
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

class poweremu_torch(nn.Module):
    def __init__(self, 
                 network_opt={}, 
                 optimizer_opt={}, 
                 train_opt={}, scale_opt={}, data_opt={},
                 device="cpu",
                 **kwargs):
        super(poweremu_torch, self).__init__()
        self.device = device
        self.multi_gpu = torch.cuda.device_count() > 1 and self.device!="cpu"
        
        self.network_opt = network_opt
        self.network_name, network_opt_ = [(key, value) for key, value in network_opt.items()][0] if len(network_opt) == 1 else ("MLP", {})    
        self.network = getattr(models, self.network_name)
        self.model = self.network(**network_opt_).to(self.device)
        
        if self.multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index])
        
        self.optimizer_opt = optimizer_opt
        self.optimizer_name, optimizer_opt_ = [(key, value) for key, value in optimizer_opt.items()][0] if len(optimizer_opt) == 1 else ("Adam", {})
        self.optimizer = getattr(torch.optim, self.optimizer_name)
        self.opt = self.optimizer(self.model.parameters(), **optimizer_opt_)
        
        self.train_opt = train_opt
        self.data_opt = data_opt
        self.scale_opt = scale_opt
        
        self.epoch = 0
        #self.loss = []
        #self.validation_loss = []
        self.loss = torch.tensor([], device=self.device)
        self.validation_loss = torch.tensor([], device=self.device)

        
    def train(self, train_dataloader, validation_dataloader, **kwargs):
        print_freq = kwargs.get("print_freq", 20)
        epochs = self.train_opt.get("epochs", 100)
        profiling = self.train_opt.get("profiling", False)
        loss_fn = self.train_opt.get("loss_fn", "MSELoss")
        save_after_epochs = self.train_opt.get("save_after_epochs", 5)
        save_progress_plots_path = self.train_opt.get("save_progress_plots_path", False)
        save_model_path = self.train_opt.get("save_model_path", None)
        terminate_time = self.train_opt.get("terminate_time", False)
        model_id = self.train_opt.get("model_id", "")
        
        parameters_validation = validation_dataloader.dataset.parameters.to(self.device)
        target_validation = validation_dataloader.dataset.target.to(self.device)

        loss_fn = getattr(torch.nn, loss_fn)()
        
        if self.device.type in ("cpu", "mps") or self.device.index == 0:
            print(self.model, flush=True)
        training_stime = time.time()
        _print_every = None  # auto-calibrated after first batch
        for e in range(self.epoch, self.epoch+epochs):
            stime = time.time()
            for i,(parameters_train,target_train) in enumerate(train_dataloader):
                _batch_t = time.time()
                parameters_train = parameters_train.to(self.device)
                target_train = target_train.to(self.device)
                if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_push(f"[{self.device.index}] predict-loss-backward-step-validation")
                self.model.train()
                self.opt.zero_grad()
                pred_train = self.model(parameters_train)
                loss = loss_fn(pred_train, target_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                if self.multi_gpu:  torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

                # calibrate print interval after first batch
                if _print_every is None and e == self.epoch and i == 0:
                    _batch_wall = time.time() - _batch_t
                    _print_every = max(1, int(20.0 / _batch_wall))
                    print(f"[diag] first batch: {_batch_wall:.3f}s/batch  →  printing every {_print_every} batches (~20s)", flush=True)

                with torch.no_grad():
                    self.model.eval()
                    pred_validation = self.model(parameters_validation)
                    if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_pop()    
                    
                    if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_push(f"[{self.device.index}] neccessary stats")
                    _vrmse = pred_validation - target_validation
                    _vrmse = torch.sqrt(torch.mean(torch.square(_vrmse)))
                    _vrmse = 10**_vrmse if self.data_opt["data_log"] else _vrmse
                    if self.multi_gpu:  torch.distributed.all_reduce(_vrmse, op=torch.distributed.ReduceOp.AVG)
                    _loss = loss.clone().detach()#.cpu().item() 
                    _vrmse = _vrmse.clone()#.detach()#.cpu().item() #expensive
                    self.loss = torch.cat((self.loss, _loss.unsqueeze(0)))
                    self.validation_loss = torch.cat((self.validation_loss, _vrmse.unsqueeze(0)))
                    if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_pop()
                    
                    if self.device.type in ("cpu", "mps") or self.device.index == 0:
                        if (_vrmse == min(self.validation_loss)) and (e >= save_after_epochs):
                            print(f"Saving model with validation loss: {_vrmse:,.2f}", flush=True)
                            self.save_network(save_model_path)
                            self.epoch = e

                            if save_progress_plots_path:
                                fig, axes = plt.subplots(1,1, figsize=(12,6))
                                pred_validation_ = pred_validation.clone().cpu().numpy() #clone tensors to avoid errors in next iteration
                                target_validation_ = target_validation.clone().cpu().numpy() #clone tensors to avoid errors in next iteration
                                bin_min = min(pred_validation_.min(), target_validation_.min())
                                bin_max = max(pred_validation_.max(), target_validation_.max())
                                bins = np.linspace(bin_min, bin_max, 100)
                                hist, edges = np.histogram(pred_validation_, bins=bins)
                                axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
                                hist, edges = np.histogram(target_validation_, bins=bins)
                                axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
                                axes.set_xlabel('Target')
                                axes.set_ylabel('Frequency')
                                axes.legend()
                                axes.set_yscale('log')#'symlog', linthresh=1e-3)
                                
                                save_path = os.path.join(save_progress_plots_path, f"validation_progress_hist{model_id}.png")
                                plt.savefig(save_path)
                                plt.close()

                    _should_print = (_print_every is not None and i % _print_every == 0) or ((_vrmse == min(self.validation_loss)) and (e >= save_after_epochs))
                    if (self.device.type in ("cpu", "mps") or self.device.index == 0) and _should_print:
                        if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_push(f"[{self.device.index}] optional stats")
                        _resid = (pred_train - target_train).detach()
                        _rmse = torch.sqrt(torch.mean(torch.square(_resid)))#.item()
                        _rmse = 10**_rmse if self.data_opt["data_log"] else _rmse
                        _q95 = torch.quantile(torch.sqrt(torch.square(_resid)), 0.95)
                        _q95 = 10**_q95 if self.data_opt["data_log"] else _q95
                        if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_pop()
                    
                        if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_push(f"[{self.device.index}] print")
                        device_str = f"[{str(self.device)}] "
                        epoch_str = f"Epoch {e} | "
                        batch_str = f"Batch {i+1} out of {train_dataloader.__len__()} | "
                        time_str = f"Time: {time.time()-stime:.2f} | "
                        train_str = f"Train: RMSE={_rmse:.4f} "
                        q95_str = f"q95<={_q95:,.4f} | "
                        validation_str = f"Validation: {_vrmse:,.4f} "
                        print(device_str + epoch_str + batch_str + time_str + train_str + q95_str + validation_str, flush=True)
                        if profiling and torch.cuda.is_available():   torch.cuda.nvtx.range_pop()
                    
                    if i == 20 and profiling: break

            #end training if time exceeds terminate_time
            if (terminate_time!=False) and (time.time()-training_stime > terminate_time):
                print(f"Training time exceeded {terminate_time} seconds. Terminating training.", flush=True)
                break
                    

    def predict(self, params, data_dims, **kwargs):
        self.model.eval()
        
        ###### get input indices ######
        emulator_indices = list(range(len(self.scale_opt)))
        emulator_keys = list(self.scale_opt.keys())
        data_dims_items = list(data_dims.items())
        for key,value in data_dims_items:
            if key in emulator_keys:
                pass
            elif f"log10{key}" in emulator_keys:
                data_dims[f"log10{key}"] = np.log10(data_dims.pop(key))
                key = f"log10{key}"
            else:
                assert False, f"{key} or log10{key} not found in emulator parameters."
            index = emulator_keys.index(key)
            emulator_indices.remove(index)
        ###### determine tiling ######
        tile = 1
        for key,value in data_dims.items():
            index = emulator_keys.index(key)
            data_dims[key] = [index, value]
            if hasattr(value, "__len__"):
                tile = max(tile, len(value))
        #####################################################

        params_ = np.empty((tile, params.size + len(data_dims)))
        for key,(index,value) in data_dims.items():
            params_[:,index] = value
        params_[:,emulator_indices] = params
        params = params_
        
        with torch.no_grad():
            params = self.scaler.transform(params, use_scale_opt=True)
            params = torch.from_numpy(params).to(dtype=torch.float32)
            pred = self.model(params)
            pred = pred.detach().cpu().numpy()
            if self.data_opt["data_log"]:
                pred = 10**pred

            #if self.convert_mK_to_K:
            #    pred *= 1e-3
        return data_dims, pred


    def save_network(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.opt.state_dict(),
                    optimizer_opt = self.optimizer_opt,
                    train_opt = self.train_opt,
                    data_opt = self.data_opt,
                    scale_opt = self.scale_opt,
                    loss=self.loss,
                    validation_loss=self.validation_loss, 
                    #loss = self.loss,
                    #validation_loss = self.validation_loss,
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
                        data_opt = self.data_opt,
                        scale_opt = self.scale_opt,
                        #loss = self.loss,
                        #validation_loss = self.validation_loss,
                        loss=self.loss,
                        validation_loss=self.validation_loss,
                        epoch = self.epoch,
                        ),
                        f = path
                        )

    def load_network(self, path):
        loaded_state = torch.load(path, map_location=self.device, weights_only=False)
        self.network_opt = loaded_state['network_opt']
        self.network_name, network_opt_ = [(key, value) for key, value in self.network_opt.items()][0] if len(self.network_opt) == 1 else ("MLP", self.network_opt)
        self.network = getattr(models, self.network_name)
        self.model = self.network(**network_opt_)
        self.model.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        try:
            self.optimizer_opt = loaded_state['optimizer_opt']
        except:
            self.optimizer_opt = {}
        self.optimizer_name, optimizer_opt_ = [(key, value) for key, value in self.optimizer_opt.items()][0] if len(self.optimizer_opt) == 1 else ("Adam", self.optimizer_opt)
        self.optimizer = getattr(torch.optim, self.optimizer_name)
        self.opt = self.optimizer(self.model.parameters(), **optimizer_opt_)
        self.opt.load_state_dict(loaded_state['optimizer'])
        self.train_opt = loaded_state['train_opt']
        try:
            self.data_opt = loaded_state['data_opt']
        except:
            self.data_opt = {}
        self.scale_opt = loaded_state['scale_opt']
        self.scaler = Scaler(self.scale_opt)
        self.loss = loaded_state['loss']
        self.validation_loss = loaded_state['validation_loss']
        try:
            self.epoch = loaded_state['epoch']
        except:
            pass
             
def prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=["alpha", "nu_0", "pop"]):
    """
    Prepares the input parameters by discarding specified parameters and log-transforming others.
    Args:
        parameters (pd.DataFrame): The input parameters to be prepared.
        transform_params (list, optional): The parameters to be log-transformed. Defaults to ['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'].
        discard_params (list, optional): The parameters to be discarded. Defaults to ['zeta', 'feed', 'delay'].
    Returns:
        pd.DataFrame: The prepared parameters with specified parameters discarded and others log-transformed.
    """
    data_opt["discrete_params"] = {}
    for param in discrete_params:
        data_opt["discrete_params"][param] = np.unique(parameters[param])
        if param in transform_params:
            data_opt["discrete_params"][param] = np.log10(data_opt["discrete_params"][param])
            data_opt["discrete_params"][f"log10{param}"] = data_opt["discrete_params"].pop(param)

    parameters = parameters.drop(columns=discard_params)
    for param in transform_params:
        parameters[param] = np.log10(parameters[param])
        parameters.rename(columns={param: f"log10{param}"}, inplace=True)
       
    return parameters, data_opt

def train_model(rank, multi_gpu, parameters_train, target_train, parameters_validation, target_validation, network_opt, optimizer_opt, train_opt, scale_opt, data_opt, **kwargs):
    batch_size = train_opt.get("batch_size", 10000)

    if multi_gpu:
        world_size = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank}")
        ddp_setup(rank, world_size=world_size)
    elif not multi_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[diag] device={device}  train_size={len(parameters_train)}  val_size={len(parameters_validation)}", flush=True)

    t0 = time.time()
    train_data_module = Dataloader(parameters=parameters_train, target=target_train, device=device)
    print(f"[diag] train Dataloader: {time.time()-t0:.2f}s  target shape={train_data_module.target.shape}", flush=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data_module, shuffle=True, seed=0) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=(train_sampler is None), sampler = train_sampler,)
    print(f"[diag] train DataLoader: {len(train_dataloader)} batches of {batch_size}", flush=True)

    t0 = time.time()
    validation_data_module = Dataloader(parameters=parameters_validation, target=target_validation, device=device)
    print(f"[diag] val Dataloader: {time.time()-t0:.2f}s  target shape={validation_data_module.target.shape}", flush=True)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=validation_data_module, shuffle=True, seed=0) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=(validation_sampler is None), sampler = validation_sampler,)

    t0 = time.time()
    emu = poweremu_torch(network_opt=network_opt,
                         optimizer_opt=optimizer_opt,
                         train_opt=train_opt, scale_opt=scale_opt, data_opt=data_opt,
                         device=device)
    print(f"[diag] poweremu_torch init: {time.time()-t0:.2f}s", flush=True)


    #train
    if train_opt["profiling"] and torch.cuda.is_available():
        with torch.autograd.profiler.emit_nvtx():
            emu.train(train_dataloader, validation_dataloader)
    else:
        emu.train(train_dataloader, validation_dataloader)

    if multi_gpu:
        torch.distributed.barrier()
        destroy_process_group()

    
def prepare_validation_data(parameters, data, data_opt, **kwargs):
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
    data = np.log10(data) if data_opt["data_log"] else data

    parameters_columns = list(parameters.columns)
    parameters = parameters.to_numpy()

    data_dims_columns = []
    data_dims = []
    lims = []
    for dim in data_opt["data_dims"]:
        key = list(dim.keys())[0]
        min, max = dim[key]["lims"]
        if dim[key]["log"]:
            data_dims_columns.append(f"log10{key}")
            data_dims.append(np.log10(dim[key]["values"]))
            lims.append([float(np.log10(min)), float(np.log10(max))])
        else:
            data_dims_columns.append(key)
            data_dims.append(dim[key]["values"])
            lims.append([min, max])


    
    #log lims
    masks = [np.logical_and(data_dim >= lim[0], data_dim <= lim[1]) for data_dim, lim in zip(data_dims, lims)]
    drop_rows = [np.where(~mask)[0] for mask in masks]
    data_dims = [data_dim[mask] for data_dim, mask in zip(data_dims, masks)]
    
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

def prepare_scale_opt(parameters, method, **kwargs):        
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
        minimum = kwargs.get("minimum", data.min())
        maximum = kwargs.get("maximum", data.max())
        return (2 * (data - minimum) / (maximum - minimum)) - 1

    def normalize(self, data, **kwargs):
        mean = kwargs.get("mean", data.mean())
        std = kwargs.get("std", data.std())
        return (data - mean) / std
    
    def identity(self, data, **kwargs):
        return data

    def inverse_standardize(self, data, minimum, maximum):
        return 0.5 * (data + 1) * (maximum - minimum) + minimum

    def inverse_normalize(self, data, mean, std):
        return (data * std) + mean

    def inverse_identity(self, data):
        return data
    
    def transform(self, parameters, use_scale_opt = True, **kwargs):
        """
        parameters: pd.DataFrame or np.ndarray
        """
        if not use_scale_opt:
            minimum = kwargs.get("minimum", parameters.min(axis=0))
            maximum = kwargs.get("maximum", parameters.max(axis=0))
            mean = kwargs.get("mean", parameters.mean(axis=0))
            std = kwargs.get("std", parameters.std(axis=0))
        
        scale_opt_keys = np.array([*self.scale_opt.keys()])
        
        if isinstance(parameters, pd.DataFrame):        
            params = parameters.columns
            scale_opt_key_in_params = np.array([key in params for key in scale_opt_keys])
            missing_keys = scale_opt_keys[~scale_opt_key_in_params]
            
            assert len(missing_keys) == 0, f"Missing keys in data: {missing_keys}"
            assert len(params) == len(scale_opt_keys), "number of features and number of transforms in scale_opt must be the same"
            
            for i,key in enumerate(scale_opt_keys):
                scale_fn = getattr(self, self.scale_opt[key]["method"])
                parameters[key] = scale_fn(parameters[key], **self.scale_opt[key]["stats"])

        elif isinstance(parameters, np.ndarray):
            n_params = parameters.shape[1]
            assert n_params == len(self.scale_opt), "Length of data and transform must be the same"
            for i,key in enumerate(scale_opt_keys):
                scale_fn = getattr(self, self.scale_opt[key]["method"])
                parameters[:,i] = scale_fn(parameters[:,i], **self.scale_opt[key]["stats"])

        return parameters

    
    def inverse_transform(self, data, use_scale_opt = True, **kwargs):
        minimum = kwargs.get("minimum", data.min(axis=0))
        maximum = kwargs.get("maximum", data.max(axis=0))
        mean = kwargs.get("mean", data.mean(axis=0))
        std = kwargs.get("std", data.std(axis=0))

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
    
def shuffle_data(parameters, data, seed=42):
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
    np.random.seed(seed)
    indices = np.random.permutation(parameters.index)
    parameters = parameters.reindex(indices).reset_index(drop=True)
    data = data[indices]
    return parameters, data

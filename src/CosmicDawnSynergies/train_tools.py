import os
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import numpy as np
import torch.distributed
from tqdm import tqdm

import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__).split('CosmicDawnSynergies')[0] + 'CosmicDawnSynergies'
path = "/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')

z_array = loadmat(path+'hera_z_mat.mat')['z21cm'][0]
k_array = loadmat(path+'hera_k_mat.mat')['ks'][0]
dsq = loadmat(path+'hera_Deltak_mat.mat')['combined_Deltaks']
XRB = loadmat(path + "hera_XRB_mat.mat")['combined_XRBs']
nu_keV = loadmat(path + "hera_nu_mat.mat")['nu_keV'][0]
parameters = loadmat(path+'hera_parameters_mat.mat')['parameters']

def random_grid_interpolation(parameters, data_dims, data, vars):
    #p = parameters[i]
    #data_dims = tuple of data axes e.g. (x (redshift), y (wavevector))
    #data = dsq[i]
    #vars = list of lists with options for each sample axes, min and max and num_samples for each parameter [[min, max, num_samples], [min, max, num_samples]]
    #var2 = wavenumber sample

    priors = [np.random.uniform(*var) for var in vars]
    
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



def gen_training_data(parameters, data_dims, data, vars, data_dims_log, data_log, n_jobs=-1, verbose=False):
    
    data_dims = [np.log10(data_dims[i]) if data_dims_log[i] else data_dims[i] for i in range(len(data_dims))]
    data = np.log10(data) if data_log else data
    vars = [[np.log10(var[0]), np.log10(var[1]), var[2]] if data_dims_log[i] else var for i, var in enumerate(vars)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(random_grid_interpolation)(parameters=p, data_dims=data_dims, data=data_i, vars=vars)
        for p, data_i in tqdm(zip(parameters, data), total=len(parameters), desc="Interpolating", disable=not verbose)
    )

    p_list, interp_list = zip(*results)
    p_list = np.vstack(p_list)
    interp_list = np.hstack(interp_list)

    return p_list, interp_list

#minimum = dsq[dsq!=0].min()
#dsq[dsq==0] = minimum * 1e-3
#p, logdsq_interp = gen_training_data(parameters=parameters, data_dims=(z_array, k_array), data=dsq, vars=[[6, 38, 140], [3e-2, 0.99, 140]], data_dims_log=[False, True], data_log=True, n_jobs=-1)

#print(p.shape, logdsq_interp.shape)


### test dsq interpolation (add hardcoded values for testing in random_grid_interpolation)
"""
minimum = dsq[dsq!=0].min()
dsq[dsq==0] = minimum * 1e-3
p, logdsq_interp = gen_training_data(parameters=parameters[:3], data_dims=(z_array, k_array), data=dsq[:3], vars=[[6, 38, 20], [3e-2, 0.99, 20]], data_dims_log=[False, True], data_log=True)

z_idx = np.where(p[:,0] == 6)[0]
k_idx = np.where(p[:,1] == np.log10(k_array[19]))[0]

dsq_test = dsq[:3]
minimum = dsq_test[dsq_test!=0].min()
dsq_test[dsq_test==0] = minimum * 1e-3
fig,axes = plt.subplots(1,2, figsize=(12,6))
axes[0].loglog(k_array, dsq_test[:3,0].T, 'o', alpha=0.5)
axes[0].loglog(10**p[z_idx,1], 10**logdsq_interp[z_idx], 'x',alpha=0.8)
axes[0].set_xlabel('k')
axes[0].set_ylabel('Delta^2')

axes[1].plot(z_array, dsq_test[:3,:,19].T, 'o', alpha=0.5)
axes[1].plot(p[k_idx,0], 10**logdsq_interp[k_idx], 'x',alpha=0.8)
axes[1].set_xlabel('z')
axes[1].set_ylabel('Delta^2')
axes[1].set_yscale('log')

plt.savefig(current_dir+'/images/dsq_interpolation_test.png')
#plt.show()

"""

"""
p, logXRB = gen_training_data(parameters=parameters[:3], data_dims=(nu_keV,), data=XRB[:3], vars=[[2e-1, 1e3, 20],], data_dims_log=[True,], data_log=True)

plt.loglog(nu_keV, XRB[:3].T, ls='solid', alpha=0.8, label='Data')
plt.loglog(10**p[:,0], 10**logXRB.T, 'x', alpha=0.5, label='Interpolated')
plt.xlabel('Frequency (keV)')
plt.ylabel('XRB')
plt.savefig(current_dir+'/images/XRB_interpolation_test.png')

#plt.show()
"""


#build MLP regressor in pytorch
import pandas as pd
import torch 
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from sklearn.model_selection import train_test_split
import time 


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
    def __init__(self, parameters, target, fullDataset=False, device="cpu", **kwargs):
        #convert np array data to dataframe
        self.fullDataset = fullDataset
        self.device = device
        self.parameters = torch.from_numpy(parameters).to(torch.float32).to(self.device)
        self.target = torch.from_numpy(target).to(torch.float32).to(self.device)
        
        self.data_dims = kwargs.pop("data_dims", None)
        self.data_dims_log = kwargs.pop("data_dims_log", None)
        self.vars = kwargs.pop("vars", None)
        self.data_log = kwargs.pop("data_log", None)

        assert not (not fullDataset and (self.data_dims is None or self.data_dims_log is None or self.vars is None or self.data_log is None)), "If fullDataset=False, data_dims, data_dims_log, vars, data_log must be provided"

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        parameters = self.parameters[idx]
        target = self.target[idx]
        if not self.fullDataset:
            parameters, target = gen_training_data(parameters=parameters, data_dims=self.data_dims, data=self.target, vars=self.vars, data_dims_log=self.data_dims_log, data_log=self.data_log, n_jobs=-1, verbose=False)

        #parameters = torch.from_numpy(parameters).to(torch.float32)
        #target = torch.tensor(target, dtype=torch.float32)        
        #parameters = parameters.to(self.device)
        #target = target.to(self.device)
        return parameters, target
"""
            if self.use_attn:
                self.layernorm_attn = nn.LayerNorm(in_dim)
                self.query = nn.Linear(in_dim, in_dim)
                self.key = nn.Linear(in_dim, in_dim)
                self.value = nn.Linear(in_dim, in_dim)
                self.softmax = nn.Softmax(dim=-1)        

            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_norm_dropout:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            if self.use_norm_dropout:
                layers.append(nn.Dropout(dropout)) ###
            for _ in range(n_hidden):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.use_norm_dropout:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Tanh())
                if self.use_norm_dropout:
                    layers.append(nn.Dropout(dropout)) ###
            layers.append(nn.Linear(hidden_dim, out_dim))
            #layers.append(nn.ReLU()) ###
            self.fc = nn.Sequential(*layers)

            
                x = self.layernorm_attn(x)
                Q = self.query(x)  # Shape: [batch_size, features]
                K = self.key(x)    # Shape: [batch_size, features]
                V = self.value(x)  # Shape: [batch_size, features]
                # Compute attention scores between features
                attention_scores = torch.matmul(Q.unsqueeze(2), K.unsqueeze(1)) / (x.size(-1) ** 0.5)
                # Shape: [batch_size, features, features]        
                attention_weights = self.softmax(attention_scores)  # Shape: [batch_size, features, features]
                # Apply attention weights to values
                x = torch.matmul(attention_weights, V.unsqueeze(2)).squeeze(2)  # Shape: [batch_size, features]


            #if torch.cuda.current_device() == 0:    print("[5] After attention, x shape:", x.shape, flush=True)      # Check x shape
            x = self.fc(x)
            #if torch.cuda.current_device() == 0:    print("[6] After model, x shape:", x.shape, flush=True)      # Check x shape
            x = torch.squeeze(x)
            #if torch.cuda.current_device() == 0:    print("[7] After squeeze, x shape:", x.shape, flush=True)      # Check x shape
"""
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden = 1, out_dim = 1, dropout = 0.2, use_norm_dropout = False, use_attn = True):
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
        self.optimizer = optimizer(self.model.parameters(), **optimizer_opt)
        self.train_opt = train_opt
        self.scale_opt = scale_opt



        self.loss = []
        self.validation_loss = []
        
    def train(self, train_dataloader, validation_dataloader, **kwargs):
        epochs = self.train_opt.pop("epochs", 100)
        profiling = self.train_opt.pop("profiling", False)
        loss_fn = self.train_opt.pop("loss_fn", torch.nn.MSELoss())
        
        parameters_validation = validation_dataloader.dataset.parameters
        target_validation = validation_dataloader.dataset.target

        #loss_fn = torch.nn.HuberLoss(delta=10.)
        #loss_fn = torch.nn.L1Loss()
        #loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        
        if self.device.index == 0 or self.device.type=="cpu":
            print(self.model, flush=True)
        for e in range(epochs):
            stime = time.time()
            for i,(parameters,target) in enumerate(train_dataloader):
                
                if profiling:   torch.cuda.nvtx.range_push("predict-loss-backward-step")
                self.model.train()
                self.optimizer.zero_grad()
                pred_train = self.model(parameters)
                loss = loss_fn(pred_train, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                if profiling:   torch.cuda.nvtx.range_pop()
                

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
                    
                    _residlog = pred_train - target
                    _rmselog = 10**torch.sqrt(torch.mean(torch.square(_residlog))).detach().item()
                    _q95log = 10**torch.quantile(torch.sqrt(torch.square(_residlog)), 0.95).item()
                    _percentlog = 100 * torch.abs(_residlog) / target
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
                        
                        if (_vrmselog == min(self.validation_loss)) and (self.device.index == 0 or self.device.type=="cpu") and (e >= 5):
                            print(f"Saving model with validation loss: {_vrmselog:,.2f}", flush=True)
                            self.save_network("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/trained_emulators_poweremu/MLP_7_2.pth")

                        #if (i == 0) or (i == train_dataloader.__len__() // 2):
                            fig, ax = plt.subplots(1,1, figsize=(6,6))
                            bin_min = min(pred_train.detach().cpu().numpy().min(), target.detach().cpu().numpy().min())
                            bin_max = max(pred_train.detach().cpu().max(), target.detach().cpu().max())
                            bins = np.linspace(bin_min, bin_max, 100)
                            hist, edges = np.histogram(pred_train.detach().cpu().numpy(), bins=bins)
                            ax.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
                            hist, edges = np.histogram(target.detach().cpu().numpy(), bins=bins)
                            ax.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
                            ax.set_xlabel('logDelta^2')
                            ax.set_ylabel('Frequency')
                            ax.legend()
                            plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/histval_7_2.png")
                            plt.close()

                            #fig, axes = plt.subplots(1,11, figsize=(6*11,6))
                            #labels = ["z", "k", "logf_star_II", "logf_star_III", "logVc", "logfX", "alpha", "nu_0", "tau", "logfrad", "pop_trans_model"]
                            #for i in range(11):
                            #    axes[i].hist(parameters_validation[:,i].detach().cpu().numpy(), bins=100)
                            #    axes[i].set_xlabel(f'{labels[i]}')
                            #    axes[i].set_ylabel('Frequency')
                            #plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/parameters_hist_7_2.png")

                            fig, axes = plt.subplots(1,2, figsize=(12,6))
                            hist, edges = np.histogram(_residlog.abs().detach().cpu().numpy(), bins=100)
                            axes[0].bar(edges[:-1], hist, width=np.diff(edges))
                            axes[0].grid()
                            axes[0].set_xlabel('absResiduals [mK^2]')
                            axes[0].set_ylabel('Frequency')
                            hist, edges = np.histogram(_percentlog.detach().cpu().numpy(), bins=100)
                            axes[1].bar(edges[:-1], hist, width=np.diff(edges))
                            axes[1].grid()
                            axes[1].set_xlabel('Percent Error')
                            axes[1].set_ylabel('Frequency')                            
                            
                            plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/residuals_perc_7_2.png")
                            plt.close()

                            fig, axes = plt.subplots(1,1, figsize=(12,6))
                            bin_min = min(pred_train.detach().cpu().numpy().min(), target.detach().cpu().numpy().min())
                            bin_max = max(pred_train.detach().cpu().max(), target.detach().cpu().max())
                            bins = np.linspace(bin_min, bin_max, 100)
                            hist, edges = np.histogram(pred_train.detach().cpu().numpy(), bins=bins)
                            axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
                            hist, edges = np.histogram(target.detach().cpu().numpy(), bins=bins)
                            axes.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
                            #axes.hist(target.detach().cpu().numpy(), bins=100, alpha=0.5, label='Targets')
                            axes.set_xlabel('logDelta^2')
                            axes.set_ylabel('Frequency')
                            axes.legend()
                            
                            plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/residuals_7_2.png")
                            plt.close()

                            #fig, axes = plt.subplots(1,11, figsize=(11*6,6))
                            #labels = ["z", "k", "logf_star_II", "logf_star_III", "logVc", "logfX", "alpha", "nu_0", "tau", "logfrad", "pop_trans_model"]
                            #for i in range(11):
                            #    axes[i].hist(parameters[:,i].detach().cpu().numpy(), bins=100)
                            #    axes[i].set_xlabel(f'{labels[i]}')
                            #    axes[i].set_ylabel('Frequency')
                            #plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/parameters_hist_4_2.png")




    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.optimizer.state_dict(),
                    train_opt = self.train_opt,
                    #ema = self.ema.state_dict(),
                    scale_opt = self.scale_opt,
                    loss = self.loss,
                    validation_loss = self.validation_loss,
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
                        optimizer = self.optimizer.state_dict(),
                        train_opt = self.train_opt,
                        #ema = self.ema.state_dict(),
                        scale_opt = self.scale_opt,
                        loss = self.loss,
                        validation_loss = self.validation_loss,
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
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        #self.ema.load_state_dict(loaded_state['ema'])
        self.train_opt = loaded_state['train_opt']
        self.scale_opt = loaded_state['scale_opt']
        self.loss = loaded_state['loss']
        self.validation_loss = loaded_state['validation_loss']
             
def prepare_parameters(parameters, log_indices=[0,1,2,3,8], discard_indices=[6,10,11]):
    if log_indices is not None:
        parameters[:, log_indices] = np.log10(parameters[:, log_indices])
    parameters = np.delete(parameters, discard_indices, axis=1)
    return parameters

def train(rank, multi_gpu, parameters_train, target_train, parameters_validation, target_validation, batch_size, scale_opt, fullDataset=False, profiling=False, **kwargs):
    
    if multi_gpu:
        world_size = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank}")
        ddp_setup(rank, world_size=world_size)
    elif not multi_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")
    
    train_data_module = Dataloader(parameters=parameters_train, target=target_train, fullDataset=fullDataset, device=device)#, data_dims=data_dims, data_dims_log=data_dims_log, vars=vars, data_log=data_log)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data_module, shuffle=True, seed=0) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=(train_sampler is None), sampler = train_sampler,)

    validation_data_module = Dataloader(parameters=parameters_validation, target=target_validation, fullDataset=fullDataset, device=device)#, data_dims=data_dims, data_dims_log=data_dims_log, vars=vars, data_log=data_log)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=validation_data_module, shuffle=True, seed=0) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=(validation_sampler is None), sampler = validation_sampler,)

    network_opt = dict(in_dim=parameters_train.shape[1], hidden_dim=100, n_hidden = 6, out_dim = 1, dropout = 0.1, use_norm_dropout = False, use_attn = False) #use_norm_dropout = True, use_attn = True)
    optimizer_opt = dict(lr=1e-3, weight_decay=1e-4)
    train_opt = dict(epochs=10000, profiling=profiling, loss_fn=torch.nn.MSELoss())
    scale_opt = scale_opt
    emu = poweremu_torch(network=MLP, network_opt=network_opt,
                         optimizer=torch.optim.Adam, optimizer_opt=optimizer_opt,
                         train_opt=train_opt, scale_opt=scale_opt,
                         device=device)


    #train
    if train_opt["profiling"]:
        with torch.autograd.profiler.emit_nvtx():
            emu.train(train_dataloader, validation_dataloader)
    else:
        emu.train(train_dataloader, validation_dataloader)

    if multi_gpu:
        torch.distributed.barrier()
        destroy_process_group()

    
def flatten_data(parameters, data, data_dims):

    grids = np.meshgrid(*data_dims, indexing='ij')
    combinations = np.vstack([grid.ravel() for grid in grids]).T

    num_parameters = len(parameters)
    num_combinations = len(combinations)
    parameters = np.repeat(parameters, num_combinations, axis=0)
    combinations = np.tile(combinations, (num_parameters, 1))
    parameters = np.hstack((combinations, parameters))
    
    data = data.ravel()
    assert len(parameters) == len(data), "Length of parameters and data must be the same"

    return parameters, data


#m_idx = 1000
#z_idx = 20
#k_start = 0
#idx_start = np.ravel_multi_index((m_idx, z_idx, k_start), dsq_original.shape)
#plt.figure()
#plt.loglog(k_array, dsq_flattened[idx_start:idx_start+len(k_array)], 'o', alpha=0.5, label='Flattened Data')
#plt.loglog(k_array, dsq_original[m_idx, z_idx], 'x', alpha=0.8, label='Original Data')
#plt.xlabel('k')
#plt.ylabel('dsq')
#plt.legend()
#plt.title('dsq vs k')
#plt.savefig("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/images/dsq_flatten_test.png")
#plt.close()    

#labels = 
# ["z", "k", "logf_star_II", 
# "logf_star_III", "logVc", "logfX", 
# "alpha", "nu_0", "tau", 
# "logfrad", "pop_trans_model"]
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
        minimum = kwargs.pop("minimum", data.min(axis=0))
        maximum = kwargs.pop("maximum", data.max(axis=0))
        mean = kwargs.pop("mean", data.mean(axis=0))
        std = kwargs.pop("std", data.std(axis=0))

        n_sim, n_params = data.shape
        assert n_params == len(self.scale_opt.keys()), "number of features and number of transforms in scale_opt must be the same"
        for i,key in enumerate(self.scale_opt.keys()):
            if self.scale_opt[key]["method"] == 'standardize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"minimum": minimum[i], "maximum": maximum[i]}
                data[:,i] = self.standardize(data[:,i], **stats)
            elif self.scale_opt[key]["method"] == 'normalize':
                stats = self.scale_opt[key]["stats"] if use_scale_opt else {"mean": mean[i], "std": std[i]}
                data[:,i] = self.normalize(data[:,i], **stats)
            else:
                data[:,i] = self.identity(data[:,i])
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

            




if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    little_h = 0.6704
    parameters = prepare_parameters(parameters)
    minimum = dsq[dsq!=0].min()
    dsq[dsq==0] = minimum * 10**np.random.uniform(-3, 0, dsq[dsq==0].shape) #1e-3
    #dsq[dsq<1.] = 1.

    #train_test_split for parameters and dsq
    parameters_train, parameters_validation, dsq_train, dsq_validation = train_test_split(parameters, dsq, test_size=0.2, train_size=0.8, random_state=42)

    zmask = np.logical_and(z_array >= 6, z_array <= 27)
    kmask = np.logical_and(k_array / little_h >= 3e-2 / little_h, k_array / little_h <= 0.99 / little_h)
    dsq_validation = dsq_validation[:, zmask, :][:, :, kmask]
    parameters_validation, dsq_validation = flatten_data(parameters=parameters_validation, data=dsq_validation, data_dims=(z_array[zmask], np.log10(k_array[kmask] / little_h)))
    logdsq_validation = np.log10(dsq_validation)
    #np.savez_compressed("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/validation_data.npz", parameters=parameters_validation, dsq=dsq_validation)
    #load
    #data = np.load("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/validation_data.npz")
    #parameters_validation = data['parameters']
    #logdsq_validation = np.log10(data['dsq'])
    
    #shuffle parameters_validation and logdsq_validation with torch
    indices = torch.randperm(len(parameters_validation))
    parameters_validation = parameters_validation[indices]
    logdsq_validation = logdsq_validation[indices]
    print(f"Shapes: parameters_train: {parameters_train.shape}, dsq_train: {dsq_train.shape}, parameters_validation: {parameters_validation.shape}, logdsq_validation: {logdsq_validation.shape}, z_array[zmask]: {z_array[zmask].shape}, k_array[kmask]: {k_array[kmask].shape}", flush=True)
    parameters_train, logdsq_train = gen_training_data(parameters=parameters_train, data_dims=(z_array, k_array / little_h), data=dsq_train, vars=[[6, 27, 10], [3e-2 / little_h, 0.99 / little_h, 10]], data_dims_log=[False, True], data_log=True, verbose=True)# if torch.cuda.current_device() == 0 else False)
    np.savez_compressed("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data_7_2.npz", parameters_train=parameters_train, logdsq_train=logdsq_train, parameters_validation=parameters_validation, logdsq_validation=logdsq_validation)
    #load training data
    #data = np.load("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data.npz")
    #parameters_train = data['parameters']
    #logdsq_train = data['logdsq']
    #standardize/normalize parameters_train and parameters_validation
    mean = parameters_train.mean(axis=0)
    std = parameters_train.std(axis=0)
    minimum = parameters_train.min(axis=0)
    maximum = parameters_train.max(axis=0)
    #labels = ["z", "k", "logf_star_II", "logf_star_III", "logVc", 
    # "logfX", "alpha", "nu_0", "tau", "logfrad", "pop_trans_model"]
    from collections import OrderedDict
    scale_opt = OrderedDict(
                     z               = dict(method = "standardize", stats = dict(minimum = minimum[0], maximum = maximum[0])),
                     logk            = dict(method = "standardize", stats = dict(minimum = minimum[1], maximum = maximum[1])),
                     logf_star_II    = dict(method = "standardize", stats = dict(minimum = minimum[2], maximum = maximum[2])),
                     logf_star_III   = dict(method = "standardize", stats = dict(minimum = minimum[3], maximum = maximum[3])),
                     logVc           = dict(method = "standardize", stats = dict(minimum = minimum[4], maximum = maximum[4])),
                     logfx           = dict(method = "standardize", stats = dict(minimum = minimum[5], maximum = maximum[5])),
                     alpha           = dict(method = "standardize", stats = dict(minimum = minimum[6], maximum = maximum[6])),
                     nu_0            = dict(method = "standardize", stats = dict(minimum = minimum[7], maximum = maximum[7])),
                     tau             = dict(method = "normalize",   stats = dict(mean = mean[8], std = std[8])),
                     logfrad         = dict(method = "standardize", stats = dict(minimum = minimum[9], maximum = maximum[9])),
                     pop_trans_model = dict(method = "standardize", stats = dict(minimum = minimum[10], maximum = maximum[10])),
                     )
    scaler = Scaler(scale_opt)
    parameters_train = scaler.transform(parameters_train)
    parameters_validation = scaler.transform(parameters_validation)
    ##min-max normalise all features except feature 8 
    #parameters_train[:, :8] = (2 * (parameters_train[:, :8] - features_min[:8]) / (features_max[:8] - features_min[:8])) - 1
    #parameters_validation[:, :8] = (2 * (parameters_validation[:, :8] - features_min[:8]) / (features_max[:8] - features_min[:8])) - 1
    #parameters_train[:, 9:] = (2 * (parameters_train[:, 9:] - features_min[9:]) / (features_max[9:] - features_min[9:])) - 1
    #parameters_validation[:, 9:] = (2 * (parameters_validation[:, 9:] - features_min[9:]) / (features_max[9:] - features_min[9:])) - 1
    ##standardise feature 8
    #parameters_train[:, 8] = (parameters_train[:, 8] - mean[8]) / std[8]
    #parameters_validation[:, 8] = (parameters_validation[:, 8] - mean[8]) / std[8]

    #remove outliers
    #mask = logdsq_train < -4
    #parameters_train = np.delete(parameters_train, mask, axis=0)
    #logdsq_train = np.delete(logdsq_train, mask, axis=0)
    #mask = logdsq_validation < -4
    #parameters_validation = np.delete(parameters_validation, mask, axis=0)
    #logdsq_validation = np.delete(logdsq_validation, mask, axis=0)
    """
    #number of data pointd before
    nbefore = len(logdsq_train)
    #bin data and uniformly draw an equal amount of samples from each bin
    def rebin_data(data, num_samples):
        bin_min = data.min()
        bin_max = data.max()
        bins = np.array([bin_min, *np.linspace(0., 5, 19), bin_max])
        bin_indices = np.digitize(data, bins)
        bin_nmin = np.array([np.sum(bin_indices == i) for i in range(1, len(bins))]).min()
        indices = []
        for i in range(1, len(bins)):
            indices.extend(np.random.choice(np.where(bin_indices == i)[0], bin_nmin, replace=False))
        return indices
    indices = rebin_data(logdsq_train, 10000)
    bins = [logdsq_train.min(), *np.linspace(-2, 5, 19), logdsq_train.max()]
    parameters_train = parameters_train[indices]
    logdsq_train = logdsq_train[indices]
    nafter = len(logdsq_train)
    print(f"Number of samples before: {nbefore:,}, after: {nafter:,}. Percentage of data kept: {100*nafter/nbefore:.2f}%", flush=True)
    
    #shuffle parameters_train and logdsq_train with torch
    indices = torch.randperm(len(parameters_train))
    parameters_train = parameters_train[indices]
    logdsq_train = logdsq_train[indices]
    
    #plot histograms of logdsq_train
    #fig, ax = plt.subplots(1,1, figsize=(6,6))
    #hist, edges = np.histogram(logdsq_train, bins=bins)
    #bin_centers = 0.5 * (edges[:-1] + edges[1:])
    #ax.bar(bin_centers, hist, width=np.diff(edges), alpha=0.5, label='Training Data')
    #ax.legend()
    #ax.set_xlabel('logdsq')
    #ax.set_ylabel('Frequency')
    #plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/logdsq_histogram_rebinned.png")

    #fig, axes = plt.subplots(1,11, figsize=(6*11,6))
    #labels = ["z", "k", "logf_star_II", "logf_star_III", "logVc", "logfX", "alpha", "nu_0", "tau", "logfrad", "pop_trans_model"]
    #for i in range(11):
    #    axes[i].hist(parameters_train[:,i], bins=100)
    #    axes[i].set_xlabel(f'{labels[i]}')
    #    axes[i].set_ylabel('Frequency')
    #plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/parameters_histogram_scaled.png")
    
    """
    if multi_gpu:
        print(f"Using multi-gpu with {world_size} GPUs", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
        torch.multiprocessing.spawn(train, args=(multi_gpu, parameters_train, logdsq_train, parameters_validation, logdsq_validation, 20000, scale_opt, True, False), nprocs=world_size)
        print("Training complete", flush=True)
    else:
        print("Not using multi-gpu")
        train(0, multi_gpu=multi_gpu, parameters_train=parameters_train, target_train=logdsq_train, parameters_validation=parameters_validation, target_validation=logdsq_validation, batch_size=10000, scale_opt=scale_opt, fullDataset=True, profiling=True)#, data_dims=(z_array, k_array), data_dims_log=[False, True], vars=[[6, 27, 10], [3e-2, 0.99, 10]], data_log=True)

    
    

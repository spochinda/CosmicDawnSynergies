import os
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import numpy as np
import torch.distributed
from tqdm import tqdm

import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__).split('CosmicDawnSynergies')[0] + 'CosmicDawnSynergies'
path = "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')

z_array = loadmat(path+'hera_z_mat.mat')['z21cm'][0]
k_array = loadmat(path+'hera_k_mat.mat')['ks'][0]
dsq = loadmat(path+'hera_Deltak_mat.mat')['combined_Deltaks']
XRB = loadmat(path + "hera_XRB_mat.mat")['combined_XRBs']
nu_keV = loadmat(path + "hera_nu_mat.mat")['nu_keV'][0]
parameters = loadmat(path+'hera_parameters_mat.mat')['parameters']

def random_grid_interpolation(p, data_dims, data, vars):
    #p = parameters[i]
    #data_dims = tuple of data axes e.g. (x (redshift), y (wavevector))
    #data = dsq[i]
    #vars = list of lists with options for each sample axes, min and max and num_samples for each parameter [[min, max, num_samples], [min, max, num_samples]]
    #var2 = wavenumber sample

    priors = [np.random.uniform(*var) for var in vars]
    
    #priors[0][0] = 6 # for testing dsq
    #priors[1][0] = np.log10(k_array[19]) # for testing dsq

    priors = [np.sort(prior) for prior in priors]
    interp = RegularGridInterpolator(data_dims, data, method='linear')
    grids = np.meshgrid(*priors)
    interp = interp(tuple(grids))
    interp = interp.flatten()
    grids = [grid.flatten()[:, None] for grid in grids]
    #grids_dim = [grid[:, None] for grid in grids]
    p = np.tile(p, (len(grids[0]), 1))
    p = np.hstack((*grids, p))

    return p, interp



def gen_training_data(parameters, data_dims, data, vars, data_dims_log, data_log, n_jobs=-1, verbose=False):
    
    data_dims = [np.log10(data_dims[i]) if data_dims_log[i] else data_dims[i] for i in range(len(data_dims))]
    data = np.log10(data) if data_log else data
    vars = [[np.log10(var[0]), np.log10(var[1]), var[2]] if data_dims_log[i] else var for i, var in enumerate(vars)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(random_grid_interpolation)(p=p, data_dims=data_dims, data=data_i, vars=vars)
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

    os.environ["MASTER_PORT"] = "2594"#"12355" 
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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden = 1, out_dim = 1):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

        ## Initialize weights
        #self._initialize_weights()

    #def _initialize_weights(self):
    #    for m in self.model:
    #        if isinstance(m, nn.Linear):
    #            nn.init.xavier_uniform_(m.weight)
    #            if m.bias is not None:
    #                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
        return x

class poweremu_torch(nn.Module):
    def __init__(self, network, network_opt, optimizer_opt, device="cpu"):
        super(poweremu_torch, self).__init__()
        self.device = device
        self.network = network #MLP
        self.network_opt = network_opt #dictionary of MLP args
        self.model = self.network(**self.network_opt).to(self.device)
        self.multi_gpu = torch.cuda.device_count() > 1
        if self.multi_gpu:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index])
        self.optimizer_opt = optimizer_opt #dictionary of optimizer args
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_opt)



        self.loss = []
        
    def train(self, train_dataloader, validation_dataloader, epochs, accu_frac=0.02, profiling=False):
        
        self.model.train()
        for e in range(epochs):
            loss_epoch = torch.tensor(0., device=self.device)
            validation_epoch = torch.tensor(0., device=self.device)
            q50 = torch.tensor(0., device=self.device)
            stime = time.time()
            for i,(parameters,target) in enumerate(train_dataloader):
                
                if profiling:   torch.cuda.nvtx.range_push("predict-loss-backward")
                pred_train = self.model(parameters)
                loss = torch.square(pred_train - target)


                with torch.no_grad():
                    resid = loss.clone().detach()
                    resid = torch.sqrt(resid)
                    q50 += torch.quantile(resid, 0.5) / train_dataloader.__len__()
                    if torch.cuda.current_device() == 0:
                        if i==0:
                            plt.figure()
                            plt.hist(resid.cpu().numpy(), bins=100)
                            plt.xlabel('Residuals')
                            plt.ylabel('Frequency')
                            plt.title('Residuals')
                            plt.savefig("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/images/residuals.png")
                            plt.close()


                loss = torch.sqrt(torch.mean(loss))
                loss.backward()
                if profiling:   torch.cuda.nvtx.range_pop()
                loss_epoch += loss.clone().detach()
                
                if ((i+1) % (train_dataloader.__len__()//(accu_frac**-1)) == 0) and (i != 0):
                    loss_epoch /= (train_dataloader.__len__()//(accu_frac**-1))
                    if self.multi_gpu:
                        torch.distributed.all_reduce(tensor=loss_epoch, op=torch.distributed.ReduceOp.AVG)
                        torch.distributed.all_reduce(tensor=q50, op=torch.distributed.ReduceOp.AVG)
                    q50 = 10**q50.item()
                    loss_epoch = 10**loss_epoch.item()

                    if self.device.index == 0 or self.device.type=="cpu":
                        print(f"[{str(self.device)}] Epoch {e} | Batch {i+1} out of {train_dataloader.__len__()} | Time: {time.time()-stime:.2f} | Loss: dDsq = {loss_epoch:.0f} ({q50:.0f}) mK^2 ", flush=True)
                    loss_epoch = torch.tensor(0., device=self.device)

                    
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                    
                    if profiling:   torch.cuda.nvtx.range_push("optimizer")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if profiling:   torch.cuda.nvtx.range_pop()                

            #if loss_epoch == torch.min(torch.tensor(self.loss)):
            #    print("Saving model! (train print)", flush=True)
            #    #self.save_network("best_model.pth")
            #else:
            #    print("Not saving model! (train print)", flush=True)
            #
            #if self.multi_gpu:
            #    torch.distributed.barrier()
    
    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        if self.train_opt.get("log_indices") is not None:
            x[:, self.train_opt["log_indices"]] = torch.log10(x[:, self.train_opt["log_indices"]])

        y = self.model(x)
        
        if (self.train_opt.get("log_output") is not None) and (self.train_opt.get("log_output")):
            y = 10**y
        return y

    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.optimizer.state_dict(),
                    train_opt = self.train_opt,
                    #ema = self.ema.state_dict(),
                    loss = self.loss,
                    ),
                    f = path
                    )
        else:
            if str(self.device) == "cuda:0":
                print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        network_opt = self.network_opt,
                        model = self.model.module.state_dict(), 
                        optimizer = self.optimizer.state_dict(),
                        train_opt = self.train_opt,
                        #ema = self.ema.state_dict(),
                        loss = self.loss,
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
        self.train_opt = loaded_state['train_opt']
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.loss = loaded_state['loss']
             
def prepare_parameters(parameters, log_indices=[0,1,2,3,8], discard_indices=[6,10,11]):
    if log_indices is not None:
        parameters[:, log_indices] = np.log10(parameters[:, log_indices])
    parameters = np.delete(parameters, discard_indices, axis=1)
    return parameters

def train(rank, multi_gpu, parameters_train, target_train, parameters_validation, target_validation, batch_size, fullDataset=False, profiling=False, **kwargs):
    data_dims = kwargs.pop("data_dims", None)
    data_dims_log = kwargs.pop("data_dims_log", None)
    vars = kwargs.pop("vars", None)
    data_log = kwargs.pop("data_log", None)
    
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

    network_opt = dict(in_dim=parameters_train.shape[1], hidden_dim=100, n_hidden = 4, out_dim = 1)
    optimizer_opt = dict(lr=1e-4)
    emu = poweremu_torch(network=MLP, network_opt=network_opt, optimizer_opt=optimizer_opt, device=device)


    #train
    if profiling:
        with torch.autograd.profiler.emit_nvtx():
            emu.train(train_dataloader, validation_dataloader, epochs=1, accu_frac=1, profiling=profiling)
    else:
        emu.train(train_dataloader, validation_dataloader, epochs=100, accu_frac=1, profiling=profiling)

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



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    parameters = prepare_parameters(parameters)
    minimum = dsq[dsq!=0].min()
    dsq[dsq==0] = minimum * 1e-3

    #train_test_split for parameters and dsq
    parameters_train, parameters_validation, dsq_train, dsq_validation = train_test_split(parameters, dsq, test_size=0.2, train_size=0.8, random_state=42)

    #parameters_validation, dsq_validation = flatten_data(parameters=parameters_validation, data=dsq_validation, data_dims=(z_array, k_array))
    #np.savez_compressed("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/validation_data.npz", parameters=parameters_validation_2, dsq=dsq_validation_2)
    #load
    data = np.load("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/validation_data.npz")
    parameters_validation = data['parameters']
    logdsq_validation = np.log10(data['dsq'])

    #parameters_train, logdsq_train = gen_training_data(parameters=parameters_train, data_dims=(z_array, k_array), data=dsq, vars=[[6, 27, 10], [3e-2, 0.99, 10]], data_dims_log=[False, True], data_log=True, verbose=True)# if torch.cuda.current_device() == 0 else False)
    #np.savez_compressed("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data.npz", parameters=parameters_train, logdsq=logdsq_train)
    #load training data
    data = np.load("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data.npz")
    parameters_train = data['parameters']
    logdsq_train = data['logdsq']

    print(parameters_train.shape, logdsq_train.shape, parameters_validation.shape, logdsq_validation.shape)

    
    if multi_gpu:
        print(f"Using multi-gpu with {world_size} GPUs", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
        torch.multiprocessing.spawn(train, args=(multi_gpu, parameters_train, logdsq_train, parameters_validation, logdsq_validation, 50000, True, False), nprocs=world_size)
        print("Training complete", flush=True)
    else:
        print("Not using multi-gpu")
        train(0, multi_gpu=multi_gpu, parameters_train=parameters_train, target_train=logdsq_train, parameters_validation=parameters_validation, target_validation=logdsq_validation, batch_size=10000, fullDataset=True, profiling=True)#, data_dims=(z_array, k_array), data_dims_log=[False, True], vars=[[6, 27, 10], [3e-2, 0.99, 10]], data_log=True)

    
    

import os
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
import numpy as np
import torch.distributed
from tqdm import tqdm

import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__).split('CosmicDawnSynergies')[0] + 'CosmicDawnSynergies'
path = os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')

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



def gen_training_data(parameters, data_dims, data, vars, data_dims_log, data_log, n_jobs=-1):
    
    data_dims = [np.log10(data_dims[i]) if data_dims_log[i] else data_dims[i] for i in range(len(data_dims))]
    data = np.log10(data) if data_log else data
    vars = [[np.log10(var[0]), np.log10(var[1]), var[2]] if data_dims_log[i] else var for i, var in enumerate(vars)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(random_grid_interpolation)(p=p, data_dims=data_dims, data=data_i, vars=vars)
        for p, data_i in tqdm(zip(parameters, data), total=len(parameters))
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


def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"])
    except:
        print("Did not find master address variable. Setting manually...")
        os.environ["MASTER_ADDR"] = "localhost"

    os.environ["MASTER_PORT"] = "2594"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus? nccl for gpus

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, parameters, target, device="cpu"):
        #convert np array data to dataframe
        self.parameters = parameters
        self.target = target
        self.device = device

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        parameters = self.parameters[idx]
        target = self.target[idx]
        parameters = torch.from_numpy(parameters).to(torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        parameters = parameters.to(self.device)
        target = target.to(self.device)
        return parameters, target


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden = 1, out_dim = 1):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        #self.fc_hidden = [nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)]
        self.fc_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for fc in self.fc_hidden:
            x = self.relu(fc(x))
        x = self.fc_out(x)

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
        
    def train(self, train_dataloader, epochs):
        
        self.model.train()
        for e in range(epochs):
            loss_epoch = torch.tensor(0., device=self.device)
            for i,(parameters,target) in enumerate(train_dataloader):
                print(f"[{str(self.device)}] Epoch {e} | Batch {i} out of {train_dataloader.__len__()}", flush=True)
                print(f"data device {parameters.device} | target device {target.device} | model device {self.model.device}", flush=True)
                predicted = self.model(parameters)
                
                loss_batch = torch.nanmean(torch.square(predicted - target))
                
                self.optimizer.zero_grad()
                loss_batch.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()

                if self.multi_gpu:
                    torch.distributed.all_reduce(tensor=loss_batch, op=torch.distributed.ReduceOp.AVG)
                
                loss_epoch += loss_batch.detach() / train_dataloader.__len__() # maybe not right


            self.loss.append(loss_epoch.item())

            print(f"[{self.device}] Epoch {e} | Loss: {loss_epoch.item()}", flush=True)

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

def train(rank, multi_gpu, parameters, logdsq_interp):
    
    if multi_gpu:
        world_size = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank}")
        ddp_setup(rank, world_size=world_size)
    elif not multi_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_data_module = Dataloader(parameters=parameters, target=logdsq_interp, device=device)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_data_module, shuffle=True, seed=0) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=10000, shuffle=(train_sampler is None), sampler = train_sampler)

    network_opt = dict(in_dim=parameters.shape[1], hidden_dim=100, n_hidden = 4, out_dim = 1)
    optimizer_opt = dict(lr=1e-3)
    emu = poweremu_torch(network=MLP, network_opt=network_opt, optimizer_opt=optimizer_opt, device=device)

    #train
    emu.train(train_dataloader, epochs=10)

    if multi_gpu:
        torch.distributed.barrier()
        destroy_process_group()

    


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1


    parameters = prepare_parameters(parameters)
    #print([(parameters[:,i].min(),parameters[:,i].max()) for i in range(parameters.shape[1])])
    minimum = dsq[dsq!=0].min()
    dsq[dsq==0] = minimum * 1e-3
    parameters, logdsq_interp = gen_training_data(parameters=parameters, data_dims=(z_array, k_array), data=dsq, vars=[[6, 27, 10], [3e-2, 0.99, 10]], data_dims_log=[False, True], data_log=True)

    if multi_gpu:
        print("Using multi-gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
        torch.multiprocessing.spawn(train, args=(multi_gpu, parameters, logdsq_interp), nprocs=world_size)
    else:
        print("Not using multi-gpu")
        train(0, multi_gpu, parameters, logdsq_interp)



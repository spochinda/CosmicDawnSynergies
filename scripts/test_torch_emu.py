from CosmicDawnSynergies.train_tools import poweremu_torch, MLP
from CosmicDawnSynergies.train_tools import prepare_parameters, flatten_data, gen_training_data
from CosmicDawnSynergies.train_tools import Scaler
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat

path = "/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/"

little_h = 0.6704

z_array = loadmat(path+'hera_z_mat.mat')['z21cm'][0]
k_array = loadmat(path+'hera_k_mat.mat')['ks'][0]
dsq = loadmat(path+'hera_Deltak_mat.mat')['combined_Deltaks']
XRB = loadmat(path + "hera_XRB_mat.mat")['combined_XRBs']
nu_keV = loadmat(path + "hera_nu_mat.mat")['nu_keV'][0]
parameters = loadmat(path+'hera_parameters_mat.mat')['parameters']


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
#np.savez_compressed("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data_7_2.npz", parameters=parameters_train, logdsq=logdsq_train, parameters_validation=parameters_validation, logdsq_validation=logdsq_validation)
#load training data
#data = np.load("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/training_data.npz")
#print(data.keys())
#parameters_train = data['parameters']
#logdsq_train = data['logdsq']
#parameters_validation = data['parameters_validation']
#logdsq_validation = data['logdsq_validation']

#convert to tensor and float32
parameters_train = torch.from_numpy(parameters_train).to(torch.float32)
logdsq_train = torch.from_numpy(logdsq_train).to(torch.float32)
parameters_validation = torch.from_numpy(parameters_validation).to(torch.float32)
logdsq_validation = torch.from_numpy(logdsq_validation).to(torch.float32)


network_opt = dict(in_dim=11, hidden_dim=100, n_hidden = 6, out_dim = 1, dropout = 0.1, use_norm_dropout = False, use_attn = False)
optimizer_opt = dict(lr=1e-3, weight_decay=1e-4)
train_opt = dict(epochs=10000, profiling=False, loss_fn=torch.nn.MSELoss())
scale_opt = dict()

poweremu = poweremu_torch(network=MLP, network_opt=network_opt, 
                          optimizer=torch.optim.Adam, optimizer_opt=optimizer_opt, 
                          train_opt=train_opt, scale_opt=scale_opt,
                          device='cpu')
poweremu.load_network("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/trained_emulators_poweremu/MLP_7_2.pth")
print("loss: ", np.shape(poweremu.loss))
poweremu.model.eval()

redshift = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
masks = [parameters_validation[:,0] == z for z in redshift]

scaler = Scaler(poweremu.scale_opt)
parameters_train = scaler.transform(parameters_train)
parameters_validation = scaler.transform(parameters_validation)

with torch.no_grad():
    pred_validation = poweremu.model(parameters_validation)
    resid = pred_validation - logdsq_validation

    """
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    bin_min = min(pred_validation.detach().cpu().numpy().min(), logdsq_validation.detach().cpu().numpy().min())
    bin_max = max(pred_validation.detach().cpu().max(), logdsq_validation.detach().cpu().max())
    bins = np.linspace(bin_min, bin_max, 100)
    hist, edges = np.histogram(pred_validation.detach().cpu().numpy(), bins=bins)
    ax.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
    hist, edges = np.histogram(logdsq_validation.detach().cpu().numpy(), bins=bins)
    ax.bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
    ax.set_xlabel('logDelta^2')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/histval_7_2_loaded.png")
    plt.close()
    """

    print(parameters_validation[:,0].min(), parameters_validation[:,0].max(), poweremu.scale_opt["z"]["stats"])

    #plot histograms by redshift from 6 to 27
    
    fig,axes = plt.subplots(6, 4, figsize=(14,18))
    for i,z in enumerate(redshift):
        ind = np.unravel_index(i, (6,4))
        mask = masks[i]
        pred_validation_temp = pred_validation[mask]
        logdsq_validation_temp = logdsq_validation[mask]
        rmse = 10**torch.sqrt(torch.mean((pred_validation_temp - logdsq_validation_temp)**2)).item()
        bin_min = min(pred_validation_temp.detach().cpu().numpy().min(), logdsq_validation_temp.detach().cpu().numpy().min())
        bin_max = max(pred_validation_temp.detach().cpu().max(), logdsq_validation_temp.detach().cpu().max())
        bins = np.linspace(bin_min, bin_max, 100)
        hist, edges = np.histogram(pred_validation_temp.detach().cpu().numpy(), bins=bins)
        axes[ind].bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Predictions')
        hist, edges = np.histogram(logdsq_validation_temp.detach().cpu().numpy(), bins=bins)
        axes[ind].bar(edges[:-1], hist, width=np.diff(edges), alpha=0.5, label='Targets')
        axes[ind].set_xlabel('logDelta^2')
        axes[ind].set_ylabel('Frequency')
        axes[ind].legend(title=f"z={z}, rmse={rmse:.2f} mK^2")
    plt.tight_layout()
    plt.savefig("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/images/histval_7_2_loaded_z.png")
    plt.close()
    





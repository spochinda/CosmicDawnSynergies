from CosmicDawnSynergies.train_tools import poweremu_torch
from CosmicDawnSynergies.train_tools import prepare_parameters, gen_training_data, prepare_validation_data, prepare_scale_opt, shuffle_data
from CosmicDawnSynergies.train_tools import Scaler
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.io import loadmat

if __name__ == '__main__':
    path = "/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/CosmicDawnSynergies/" #"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')
    model_path = path + "data/trained_emulators_poweremu/MLP_0.pth"
    model_name = model_path.split("/")[-1].split(".")[0]
    z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0]
    k_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_k_mat.mat')['ks'][0]
    dsq = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Deltak_mat.mat')['combined_Deltaks']
    parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']

    #convert to pandas dataframe
    z_array = pd.DataFrame(z_array, columns=['z'])
    k_array = pd.DataFrame(k_array, columns=['k'])
    parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])

    #determine ranges for z and k
    little_h = 0.6704
    k_min = 3e-2 / little_h
    k_max = 0.99 / little_h
    k_array = k_array / little_h
    z_min = 6
    z_max = 27

    #discard irrelevant parameters and log-transform selected parameters    
    parameters = prepare_parameters(parameters)

    minimum = dsq[dsq!=0].min()
    dsq[dsq==0] = minimum * 10**np.random.uniform(-3, 0, dsq[dsq==0].shape) #randomly truncate zeros to below 0-3 orders of magnitude below minimum

    #Split data and parameters into training set and test set
    parameters_train, parameters_validation, dsq_train, dsq_validation = train_test_split(parameters, dsq, test_size=0.2, train_size=0.8, random_state=42)

    #generate training data and prepare validation data
    parameters_train, logdsq_train = gen_training_data(parameters=parameters_train, data_dims=[z_array, k_array], data=dsq_train, lims_nsample=[[z_min, z_max, 10], [k_min, k_max, 10]], data_dims_log=[False, True], data_log=True, verbose=True)# if torch.cuda.current_device() == 0 else False)
    parameters_validation, logdsq_validation = prepare_validation_data(parameters=parameters_validation, data=dsq_validation, data_dims=[z_array, k_array], data_dims_log=[False, True], lims=[[z_min, z_max], [k_min, k_max]], data_log=True)

    #scale (standardize (default) or normalize) parameters
    method = dict(tau = "normalize")
    scale_opt = prepare_scale_opt(parameters_train, method)
    scaler = Scaler(scale_opt)
    parameters_train = scaler.transform(parameters_train)
    parameters_validation = scaler.transform(parameters_validation)

    #shuffle
    parameters_validation, logdsq_validation = shuffle_data(parameters_validation, logdsq_validation)
    parameters_train, logdsq_train = shuffle_data(parameters_train, logdsq_train)

    network_opt = dict()
    optimizer_opt = dict()
    train_opt = dict()
    scale_opt = dict()

    poweremu = poweremu_torch(network_opt=network_opt, 
                            optimizer_opt=optimizer_opt, 
                            train_opt=train_opt, scale_opt=scale_opt,
                            device='cpu')
    poweremu.load_network(model_path)
    print("network_opt: ", poweremu.network_opt)
    scaler = Scaler(poweremu.scale_opt)
    assert False

    parameters_validation = torch.from_numpy(parameters_validation.to_numpy()).to(torch.float32)
    redshifts = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    #redshifts = pd.DataFrame(redshift, columns=['z'])
    redshifts = scaler.standardize(redshifts, minimum=poweremu.scale_opt["z"]["stats"]["minimum"], maximum=poweremu.scale_opt["z"]["stats"]["maximum"])
    ##unique redshifts in parameters_validation
    #unique_redshifts = np.unique(parameters_validation[:,0].detach().cpu().numpy())

    #mask for redshift almost equal to 
    masks = [np.isclose(parameters_validation[:,0].detach().cpu().numpy(), redshift) for redshift in redshifts]
    print([mask.sum() for mask in masks])
    poweremu.model.eval()

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


        #plot histograms by redshift from 6 to 27
        
        fig,axes = plt.subplots(6, 4, figsize=(14,18))
        for i,z in enumerate(redshifts):
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
        plt.savefig(path + f"images/histval_z_{model_name}.png")
        plt.close()
        





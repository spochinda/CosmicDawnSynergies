from scipy.io import loadmat
import pandas as pd
from CosmicDawnSynergies.train_tools import prepare_parameters, gen_training_data, prepare_validation_data, prepare_scale_opt, Scaler, shuffle_data, train_model
import numpy as np
from sklearn.model_selection import train_test_split
import torch

if __name__ == '__main__':
    path = "/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/CosmicDawnSynergies/" #"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')
    
    little_h = 0.6704
    #define network, optimizer, training, and data options
    network_opt = {"MLP": {"in_dim": 11, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
    optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
    train_opt = dict(epochs=10000, batch_size=10000, profiling=False, loss_fn="MSELoss", save_after_epochs=5, 
                        save_model_path=path+"data/trained_emulators_poweremu/MLP_0.pth",
                        save_progress_plots_path=path+"images/",)
    data_opt = dict(data_log=True, 
                    data_dims_log=[False, True], #
                    lims_nsample=[[6, 27, 10], [3e-2/little_h, 0.99/little_h, 10]],
                    lims=[[6, 27], [3e-2/little_h, 0.99/little_h]],
                    train_test_split_opt = dict(test_size=0.2, train_size=0.8, random_state=42),
                    scale_method=dict(tau="normalize")
                    )

    #load data
    z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0]
    k_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_k_mat.mat')['ks'][0]
    k_array = k_array / little_h
    target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Deltak_mat.mat')['combined_Deltaks']
    parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']

    #convert to pandas dataframe
    z_array = pd.DataFrame(z_array, columns=['z'])
    k_array = pd.DataFrame(k_array, columns=['k'])
    data_dims = [z_array, k_array]
    parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
    
    #discard irrelevant parameters, log-transform selected parameters, and randomly truncate zeros down to 3 orders of magnitude below minimum
    parameters = prepare_parameters(parameters)
    minimum = target[target!=0].min()
    target[target==0] = minimum * 10**np.random.uniform(-3, 0, target[target==0].shape)

    ############################################################## EDIT ABOVE THIS LINE ##############################################################

    #Split data and parameters into training set and test set
    parameters_train, parameters_validation, target_train, target_validation = train_test_split(parameters, target, **data_opt["train_test_split_opt"])

    #generate training data and prepare validation data
    assert (len(target.shape)-1) == len(data_dims) == len(data_opt["lims_nsample"]) == len(data_opt["lims"]) == len(data_opt["data_dims_log"]), "Number of data dimensions does not match number of limits and transformation options in data options"
    parameters_train, target_train = gen_training_data(parameters=parameters_train, data=target_train, data_dims=data_dims, verbose=True, **data_opt)
    assert len(parameters_train.columns) == network_opt["MLP"]["in_dim"], f"Number of input parameters ({len(parameters_train.columns)}) does not match network input dimension ({network_opt['MLP']['in_dim']})"
    parameters_validation, target_validation = prepare_validation_data(parameters=parameters_validation, data=target_validation, data_dims=data_dims, **data_opt)

    #scale (standardize (default) or normalize) parameters
    scale_opt = prepare_scale_opt(parameters_train, data_opt["scale_method"])
    scaler = Scaler(scale_opt)
    parameters_train = scaler.transform(parameters_train)
    parameters_validation = scaler.transform(parameters_validation)

    #shuffle
    parameters_validation, target_validation = shuffle_data(parameters_validation, target_validation)
    parameters_train, target_train = shuffle_data(parameters_train, target_train)



    #launch training process
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1
    if multi_gpu:
        print(f"Using multi-gpu with {world_size} GPUs", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
        torch.multiprocessing.spawn(train_model, args=(multi_gpu, parameters_train, target_train, parameters_validation, target_validation, network_opt, optimizer_opt, train_opt, scale_opt, data_opt), nprocs=world_size)
        print("Training complete", flush=True)
    else:
        print("Not using multi-gpu")
        train_model(0, multi_gpu=multi_gpu, parameters_train=parameters_train, target_train=target_train, parameters_validation=parameters_validation, target_validation=target_validation, network_opt=network_opt, optimizer_opt=optimizer_opt, train_opt=train_opt, scale_opt=scale_opt, data_opt=data_opt)
from scipy.io import loadmat
import pandas as pd
from CosmicDawnSynergies.train_tools import prepare_parameters, gen_training_data, prepare_validation_data, prepare_scale_opt, Scaler, shuffle_data, train_model
import numpy as np
from sklearn.model_selection import train_test_split

import torch

#load data
path = "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/" #"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/" #os.path.join(current_dir, 'data/models_21cmSim/HERA_IDR4_Emulator_Data/')
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
#np.savez_compressed("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/data.npz", parameters_train=parameters_train, dsq_train=dsq_train, parameters_validation=parameters_validation, dsq_validation=dsq_validation)
#load
#data = np.load("/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/data.npz")
#parameters_train = data['parameters_train']
#logdsq_train = data['dsq_train']
#parameters_validation = data['parameters_validation']
#logdsq_validation = data['dsq_validation']

#scale (standardize (default) or normalize) parameters
method = dict(tau = "normalize")
scale_opt = prepare_scale_opt(parameters_train, method)
scaler = Scaler(scale_opt)
parameters_train = scaler.transform(parameters_train)
parameters_validation = scaler.transform(parameters_validation)

#shuffle
parameters_validation, logdsq_validation = shuffle_data(parameters_validation, logdsq_validation)
parameters_train, logdsq_train = shuffle_data(parameters_train, logdsq_train)

network_opt = dict(in_dim=len(parameters_train.columns), hidden_dim=100, n_hidden = 6, out_dim = 1, dropout = 0.1, use_norm_dropout = False, use_attn = False) #use_norm_dropout = True, use_attn = True)
optimizer_opt = dict(lr=1e-3, weight_decay=1e-4)
train_opt = dict(epochs=10000, batch_size=10000, profiling=False, loss_fn="MSELoss", save_after_epochs=5, 
                    save_model_path=path+"data/trained_emulators_poweremu/MLP_0.pth",
                    save_progress_plots_path=path+"images/",)

#launch training process
world_size = torch.cuda.device_count()
multi_gpu = world_size > 1
if multi_gpu:
    print(f"Using multi-gpu with {world_size} GPUs", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
    torch.multiprocessing.spawn(train_model, args=(multi_gpu, parameters_train, logdsq_train, parameters_validation, logdsq_validation, network_opt, optimizer_opt, train_opt, scale_opt), nprocs=world_size)
    print("Training complete", flush=True)
else:
    print("Not using multi-gpu")
    train_model(0, multi_gpu=multi_gpu, parameters_train=parameters_train, target_train=logdsq_train, parameters_validation=parameters_validation, target_validation=logdsq_validation, network_opt=network_opt, optimizer_opt=optimizer_opt, train_opt=train_opt, scale_opt=scale_opt)
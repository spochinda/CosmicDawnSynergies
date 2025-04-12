from scipy.io import loadmat
import pandas as pd
from CosmicDawnSynergies.train_tools import prepare_parameters, gen_training_data, prepare_validation_data, prepare_scale_opt, Scaler, shuffle_data, train_model
from CosmicDawnSynergies.likelihood import LikelihoodRadioBackground
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from copy import deepcopy
from collections import OrderedDict
import argparse


if __name__ == '__main__':
    path = "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/" # "/Users/simonpochinda/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies/"
    parser = argparse.ArgumentParser(description="Train different emulators.")
    parser.add_argument('--emulator', type=str, choices=['Delta21', 'XRB', 'T_today', 'T21', 'Ts', 'TK', 'Trad'], default='Ts', help="Choose which emulator to train.")
    args = parser.parse_args()

    emulator_choice = args.emulator
    print(f"Training {emulator_choice} emulator")

    ############################################################## Delta21^2 emu ##############################################################
    if emulator_choice == 'Delta21':
        #define network, optimizer, training, and data options
        little_h = 0.6704
        network_opt = {"MLP": {"in_dim": 10, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=10, 
                            save_model_path=path+"data/trained_emulators_poweremu/dsq_frad_Stefan_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"z": {"log": False, "lims": [6., 27.], "nsample": 40}},
                        {"k": {"log": True, "lims": [3e-2/little_h, 0.99/little_h], "nsample": 40}}
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }

        #load data
        #z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        #z_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_z_mat.mat')['z21cm'][0].astype(float)
        z_array = np.arange(6, 51, 1, dtype=np.float32)
        #k_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_k_mat.mat')['ks'][0]
        #k_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_k_mat.mat')['ks'][0]
        k_array = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/K_sims_fRad.mat')['Kout'][0]
        k_array = k_array / little_h
        
        data_opt["data_dims"][0]["z"]["values"] = z_array
        data_opt["data_dims"][1]["k"]["values"] = k_array
        
        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Deltak_mat.mat')['combined_Deltaks']
        #target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_Deltak_mat.mat')['combined_Deltaks']
        target = np.real(loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/Pk_sims_fRad.mat')['PKout'])
        #find simulations with nans and remove them
        nan_indices = np.argwhere(np.isnan(target))
        nan_indices = np.unique(nan_indices[:,0])
        print(f"Removing {len(nan_indices)} simulations from shape {target.shape}")
        target = np.delete(target, nan_indices, axis=0)
        
        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        parameters = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/PT_sims_fRad.mat')['PTout']
        parameters = np.delete(parameters, nan_indices, axis=0)

        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['Rmfp', 'fstar', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio'])
                
        #discard irrelevant parameters, annd log-transform selected parameters
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstar', 'Vc', 'fX', 'fradio'], discard_params=['zeta'], discrete_params=['alpha', 'nu_0'])

        ############################################################## XRB emu ##############################################################
    if emulator_choice == 'XRB':
        network_opt = {"MLP": {"in_dim": 10, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=5, 
                            save_model_path=path+"data/trained_emulators_poweremu/xrb_Arad_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"E_min": {"log": True, "lims": [0.8, 70.], "nsample": 200}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }
        
        #nu_keV = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_nu_mat.mat')['nu_keV'][0]
        nu_keV = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_nu_mat.mat')['nu_keV'][0]
        data_opt["data_dims"][0]["E_min"]["values"] = nu_keV

        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_XRB_mat.mat')['combined_XRBs']
        target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_XRB_mat.mat')['combined_XRBs']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        
        
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])

        ############################################################## T_today emu ##############################################################
    if emulator_choice == 'T_today':
        network_opt = {"MLP": {"in_dim": 10, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=50, 
                            save_model_path=path+"data/trained_emulators_poweremu/T_today_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"nu_today": {"log": True, "lims": [1e7, 1.1e10], "nsample": 200}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }

        
        nu_today = np.logspace(-2, 1.1, 100)*1e9
        data_opt["data_dims"][0]["nu_today"]["values"] = nu_today
        
        target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_SFR_mat.mat')['combined_SFRs']
        parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])

        #calculate T_today for LWA/ARCADE constraints emulator
        z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        target = np.array([LikelihoodRadioBackground().get_T_radio_today_Jiten(zs=z_array, sfr=sfr, frad=10**frad)[1] for sfr,frad in zip(target, parameters['log10fradio'].values)])

    ############################################################## T21 emu ##############################################################
    if emulator_choice == 'T21':
        network_opt = {"MLP": {"in_dim": 10, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=5, 
                            save_model_path=path+"data/trained_emulators_poweremu/T21_Arad_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": False,
                    "data_dims":[ 
                        {"z": {"log": False, "lims": [6., 27.], "nsample": 200}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }
        
        #z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        z_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_z_mat.mat')['z21cm'][0].astype(float)
        data_opt["data_dims"][0]["z"]["values"] = z_array

        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_T21_mat.mat')['combined_T21s']
        target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_T21_mat.mat')['combined_T21s']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])

    ############################################################## Ts emu ##############################################################
    if emulator_choice == 'Ts':
        network_opt = {"MLP": {"in_dim": 9, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=10, 
                            save_model_path=path+"data/trained_emulators_poweremu/Ts_frad_Stefan_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"z": {"log": False, "lims": [6., 27.], "nsample": 400}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }
        
        #z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        #z_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_z_mat.mat')['z21cm'][0].astype(float)
        z_array = np.arange(6, 51, 1, dtype=np.float32)
        data_opt["data_dims"][0]["z"]["values"] = z_array

        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Ts_mat.mat')['combined_Tss']
        #target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_Ts_mat.mat')['combined_Tss']
        target = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/TS_sims_fRad.mat')['TSout']        
        #find simulations with nans and remove them
        nan_indices = np.argwhere(np.isnan(target))
        nan_indices = np.unique(nan_indices[:,0])
        print(f"Removing {len(nan_indices)} simulations from shape {target.shape}")
        target = np.delete(target, nan_indices, axis=0)
        
        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        parameters = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/PT_sims_fRad.mat')['PTout']
        parameters = np.delete(parameters, nan_indices, axis=0)
        
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['Rmfp', 'fstar', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio'])

        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstar', 'Vc', 'fX', 'fradio'], discard_params=['zeta'], discrete_params=['alpha', 'nu_0'])
        

    ############################################################## TK emu ##############################################################
    if emulator_choice == 'TK':
        network_opt = {"MLP": {"in_dim": 9, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=10, 
                            save_model_path=path+"data/trained_emulators_poweremu/TK_Arad_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"z": {"log": False, "lims": [6., 27.], "nsample": 200}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }
        
        #z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        #z_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_z_mat.mat')['z21cm'][0].astype(float)
        z_array = np.arange(6, 51, 1, dtype=np.float32)
        data_opt["data_dims"][0]["z"]["values"] = z_array

        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_TK_mat.mat')['combined_TKs']
        #target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_TK_mat.mat')['combined_TKs']
        target = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/TK_sims_fRad.mat')['TKout']
        #find simulations with nans and remove them
        nan_indices = np.argwhere(np.isnan(target))
        nan_indices = np.unique(nan_indices[:,0])
        target = np.delete(target, nan_indices, axis=0)
        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        parameters = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/PT_sims_fRad.mat')['PTout']
        parameters = np.delete(parameters, nan_indices, axis=0)
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['Rmfp', 'fstar', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio'])

        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstar', 'Vc', 'fX', 'fradio'], discard_params=['zeta'], discrete_params=['alpha', 'nu_0'])


    ############################################################## Trad emu ##############################################################
    if emulator_choice == 'Trad':
        network_opt = {"MLP": {"in_dim": 9, "hidden_dim": 100, "n_hidden": 6, "out_dim": 1}}
        optimizer_opt = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
        train_opt = dict(epochs=10000, batch_size=20000, profiling=False, loss_fn="MSELoss", 
                            save_after_epochs=50, 
                            save_model_path=path+"data/trained_emulators_poweremu/Trad_frad_Stefan_emu.pth",
                            save_progress_plots_path=path+"images/",
                            terminate_time=3600*2,
                            model_id=f"_{emulator_choice}",
                            )
        data_opt = {"data_log": True,
                    "data_dims":[ 
                        {"z": {"log": False, "lims": [6., 27.], "nsample": 400}},
                        ],
                    "train_test_split_opt": {"test_size": 0.2, "train_size": 0.8, "random_state": 42},
                    "scale_method": {"tau": "normalize"},
                    }
        
        #z_array = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat')['z21cm'][0].astype(float)
        #z_array = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_z_mat.mat')['z21cm'][0].astype(float)
        z_array = np.arange(6, 51, 1, dtype=np.float32)
        data_opt["data_dims"][0]["z"]["values"] = z_array

        #target = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_TradLOS_mat.mat')['combined_TradLOSs']
        #target = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_TradLOS_mat.mat')['combined_TradLOSs']
        target = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/Trad_sims_fRad.mat')['Tradout']
        #find simulations with nans and remove them
        nan_indices = np.argwhere(np.isnan(target))
        nan_indices = np.unique(nan_indices[:,0])
        print(f"Removing {len(nan_indices)} simulations from shape {target.shape}")
        target = np.delete(target, nan_indices, axis=0)

        #parameters = loadmat(path+'data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat')['parameters']
        #parameters = loadmat(path+'data/models_21cmSim/HERA_Cosmic_String_Dataset/hera_parameters_mat.mat')['parameters']
        parameters = loadmat('/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/H1C_Stefan/Sims2021/PT_sims_fRad.mat')['PTout']
        parameters = np.delete(parameters, nan_indices, axis=0)
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio', 'pop', 'feed', 'delay'])
        #parameters = pd.DataFrame(parameters, columns=['fstarII', 'fstarIII', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'Arad', 'pop', 'feed', 'delay'])
        parameters = pd.DataFrame(parameters, columns=['Rmfp', 'fstar', 'Vc', 'fX', 'alpha', 'nu_0', 'zeta', 'tau', 'fradio'])

        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'fradio'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        #parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstarII', 'fstarIII', 'Vc', 'fX', 'Arad'], discard_params=['zeta', 'feed', 'delay'], discrete_params=['alpha', 'nu_0', 'pop'])
        parameters, data_opt = prepare_parameters(parameters, data_opt, transform_params=['fstar', 'Vc', 'fX', 'fradio'], discard_params=['zeta'], discrete_params=['alpha', 'nu_0'])

    ############################################################## EDIT ABOVE THIS LINE ##############################################################

    #randomly truncate zeros down to 3 orders of magnitude below minimum if data_log is True
    if data_opt["data_log"]:
        minimum = target[target!=0].min()
        target[target==0] = minimum * 10**np.random.uniform(-3, 0, target[target==0].shape)

    #Split data and parameters into training set and test set
    parameters_train, parameters_validation, target_train, target_validation = train_test_split(parameters, target, **data_opt["train_test_split_opt"])
    #generate training data and prepare validation data
    assert (len(target.shape)-1) == len(data_opt["data_dims"]), "Number of data dimensions does not match number of limits and transformation options in data options"
    parameters_train, target_train = gen_training_data(parameters=parameters_train, data=target_train, data_opt=data_opt, verbose=True)
    assert len(parameters_train.columns) == network_opt["MLP"]["in_dim"], f"Number of input parameters ({len(parameters_train.columns)}) does not match network input dimension ({network_opt['MLP']['in_dim']})"
    parameters_validation, target_validation = prepare_validation_data(parameters=parameters_validation, data=target_validation, data_opt=data_opt)
    ##save
    #parameters_train.to_csv(path+"data/parameters_train.csv", index=False)
    #parameters_validation.to_csv(path+"data/parameters_validation.csv", index=False)
    #np.savez(path+"data/target.npz", target_train=target_train, target_validation=target_validation)
    ##load
    #parameters_train = pd.read_csv(path+"data/parameters_train.csv")
    #parameters_validation = pd.read_csv(path+"data/parameters_validation.csv")
    #target = np.load(path+"data/target.npz")
    #target_train = target["target_train"]
    #target_validation = target["target_validation"]

    #scale (standardize (default) or normalize) parameters
    scale_opt = prepare_scale_opt(parameters_train, deepcopy(data_opt["scale_method"]))
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
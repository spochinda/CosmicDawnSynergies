import numpy as np
from mpi4py import MPI
from codes.tools import *
import codes.itamar.radio_cutoff_calc as rad
import time
import os
import psutil

import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

path="/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

Trad = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TradLOS_", name="hera", key='combined_TradLOSs', endings=["mat"])
TK = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TK_", name="hera", key='combined_TKs', endings=["mat"])
Ts = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_Ts_", name="hera", key='combined_Tss', endings=["mat"])
parameters = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_parameters_", name="hera", key='parameters', endings=["mat"])
z_array = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_z_", name="hera", key='z21cm', endings=["mat"])[0]


nan_samples = np.unique(
    np.concatenate([
        np.where(np.isnan(Trad))[0],
        np.where(np.isnan(TK))[0],
        np.where(np.isnan(Ts))[0],
    ])
)

SFR = load_files(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_SFR_", name="hera", key='combined_SFRs', endings=["mat"])
SFR_HERA4 = np.delete(SFR, nan_samples, axis=0)
parameters = np.delete(parameters, nan_samples, axis=0)

def PT12_to_PL9(PT12):
    # Convert 12 parameters to usually used parameters, with appropriate logs
    features = np.zeros((PT12.shape[0], 9))

    features[:,0] = np.log10(PT12[:,0]) #fstarII
    features[:,1] = np.log10(PT12[:,1]) #fStarIII
    features[:,2] = np.log10(PT12[:,2]) #Vc
    features[:,3] = np.log10(PT12[:,3]) #fX
    features[:,4] = PT12[:,4] #alpha #positive [1,1.3,1.5]
    features[:,5] = PT12[:,5] #nu_0
    #features[:,5] = PT10[:,6] #zeta
    features[:,6] = PT12[:,7] #tau
    features[:,7] = np.log10(PT12[:,8]) #fradio
    features[:,8] = PT12[:,9] #pop
    #features[:,9] = PT12[:,10] #feed
    #features[:,10] = PT12[:,11] #delay
    return features


parameters_HERA4 = PT12_to_PL9(parameters)

def Tradio(params, sfr, z_cutoff=6.01):
    z_dense = np.linspace(z_cutoff - 0.01, z_cutoff + 0.01, 2)

    # Split the work among processes
    chunked_params = np.array_split(params, size)
    my_params = chunked_params[rank]
    
    T_model = np.zeros(shape=(len(my_params), 100))    

    # Initialize progress tracking for each rank
    last_integer_percentages = [-20] * size  # Start with -20 for all ranks to ensure 0% is displayed
    start_times = [time.time()] * size
    
    # Loop and simulate work
    for i,p in enumerate(my_params):
        fr = 10 ** p[7]
        nu_today, T_today = rad.get_T_radio_today(np.array([6.0, 6.01]), 10 ** np.interp(z_dense, z_array, np.log10(sfr[i, :])))
        T_model[i, :] = np.mean(T_today, axis=0) * fr

        # Calculate progress percentage for this process
        progress = 100*(i + 1) / len(my_params)
        # Display progress at every X%
        integer_percentage = int(progress)
        if integer_percentage % 20 == 0 and integer_percentage > last_integer_percentages[rank]:
            elapsed_time = time.time() - start_times[rank]
            last_integer_percentages[rank] = integer_percentage
            print(f"Progress (Rank {rank}): {integer_percentage}% - Elapsed Time: {elapsed_time:.2f} seconds", flush=True)
        
        # Memory check for each rank
        if False:#i % 10 ==0:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
            print(f"Rank {rank} Memory Usage: {memory_usage:.2f} MB")

    # Gather results from all processes
    all_T_model = comm.gather(T_model, root=0)

    if rank == 0:
        print(all_T_model)
        # Combine results from all processes
        print("Combining results...")
        combined_T_model = np.concatenate(all_T_model, axis=0)
        print("Combined results!")

        return nu_today, combined_T_model #nu_today, combined_T_model, indices
    else:
        return None,None

if __name__ == "__main__":

    
    if os.path.exists(path+"T_model.npz"):
        if rank ==0: 
            print("File exists!")
            data = np.load(path+"T_model.npz")
            nu_today = data["nu_today"]
            T_model = data["T_model"]
            for i,T in enumerate(T_model):
                if (np.all(T == 0)): #and (i%10000==0):
                    print("Found all zeros at i={0}, note: {1}".format(i, i%10000==0))

    else:
        print("File does not exist.")
        nu_today, T_model = Tradio(params=parameters_HERA4, sfr=SFR_HERA4)
        if rank==0:
            for i,T in enumerate(T_model):
                if np.all(T == 0):
                    print(i)
            np.savez(path+"T_model", T_model=T_model, nu_today=nu_today)
            #np.save(path+"T_model.npy", T_model)
            print("Finished saving")


    if False:#rank==0:
        [nu_obs, T_obs, dT_obs] = np.load(path+"codes/itamar/LWA1_with_err.npy")
        #for T in T_model:
        plt.loglog(nu_today/1e9, T_model[::100,:].T, ls = "solid",c="k",alpha=0.05)
        plt.loglog(nu_obs/1e9, T_obs, c="g",marker="o",ls="solid")
        plt.axvline(max(nu_today[::2]/1e9))
        plt.show()
    if rank ==0:
        #test emulator
        from codes.emulator_poweremu import *

        def emulatorModel1d(emu, arr, p):
            par0 = np.array([np.NaN, *p])
            params=np.tile(par0, (len(arr), 1))
            params[:,0] = arr
            return 10**emu.predict(params)

        parameters_HERA4_train, parameters_HERA4_test, T_model_HERA4_train, T_model_HERA4_test = train_test_split(parameters_HERA4, T_model, test_size=0.2, random_state=42)
        emu_T = poweremu(loadfile=path + "data/trained_emulators_poweremu/T_emu2_n100_l100100100100_t0.0001_o0.pkl", tol=1e-4, n_iter_no_change=99999, preprocesss_log_x=False,preprocess_y=False, offset=0)


        fig, ax = plt.subplots(3,1, sharex=True)
        alpha = 0.5
        for i in np.arange(len(T_model_HERA4_test)-10, len(T_model_HERA4_test), 1):
            T_model_emu = emulatorModel1d(emu=emu_T, arr = np.log10(nu_today[::2]/1e9), p = parameters_HERA4_test[i,:])
            ax[0].loglog(nu_today/1e9, T_model_HERA4_test[i,:], c="k", alpha=alpha)
            ax[0].loglog(nu_today[::2]/1e9, T_model_emu, c="r", alpha=alpha, ls="dashed")
            ax[1].loglog(nu_today[::2]/1e9, 100*(T_model_HERA4_test[i,::2] - T_model_emu)/T_model_HERA4_test[i,::2], c="k", alpha=alpha)    
            ax[2].loglog(nu_today[::2]/1e9, (T_model_HERA4_test[i,::2] - T_model_emu), c="k", alpha=alpha)
        
        quants = np.nanpercentile(T_model_HERA4_test[:,::2] - T_model_emu, [16,84],axis=0)
        ax[2].plot(nu_today[::2]/1e9, quants.T,  c="r", alpha=0.5)
        print([[q.min(), q.max()] for q in quants])
        
        ax[1].set_yscale("symlog")
        ax[2].set_yscale("symlog")

        ax[0].set_ylabel("Tradio")
        ax[1].set_ylabel("Percentage difference")
        ax[2].set_ylabel("Difference")
        ax[2].set_xlabel("Frequency")

        plt.show()


        #print(emu_T.predict([np.log10(nu_today[10]/1e9), *parameters_HERA4_test[0,:]]))
        #print(emulatorModel1d(emu=emu_T, arr = np.log10(nu_today/1e9), p = parameters_HERA4_test[0,:]))

    #comm.barrier()
    MPI.Finalize()
        # Process the combined results as needed
        # ...
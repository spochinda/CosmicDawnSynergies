print("Script started", flush=True)

from codes.emulator_poweremu import *
from codes.loader_21cmSim import *
from codes.tools import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from mpi4py import MPI
print("Check 1", flush=True)
comm = MPI.COMM_WORLD
print("Check 2", flush=True)
rank = comm.Get_rank()
print("Check 3", flush=True)
size = comm.Get_size()
print("Check 4", flush=True)

import time
print("RANK: ", rank, flush=True)
#time.sleep(5)

# Redshift and k ranges used in data, and load params and powerspectra
## Finally get the wavenumbers [1/cMpc] from the files. They
## should be all identical but double check for new data.
path = 'data/models_21cmSim/HERA_IDR4_Emulator_Data/'

z_array = load_files(path, middle="_z_", name="hera", key='z21cm', endings=["mat"])[0]
k_array = load_files(path, middle="_k_", name="hera", key='ks', endings=["mat"])[0]

Deltak = load_files(path, middle="_Deltak_", name="hera", key='combined_Deltaks', endings=["mat"])
parameters = load_files(path, middle="_parameters_", name="hera", key='parameters', endings=["mat"])
Trad = load_files(path, middle="_TradLOS_", name="hera", key='combined_TradLOSs', endings=["mat"])
TK = load_files(path, middle="_TK_", name="hera", key='combined_TKs', endings=["mat"])
Ts = load_files(path, middle="_Ts_", name="hera", key='combined_Tss', endings=["mat"])
#XRB = load_files(path, middle="_XRB_", name="hera", key='combined_XRBs', endings=["mat"])

nan_samples = np.unique(
    np.concatenate([
        np.where(np.isnan(Trad))[0],
        np.where(np.isnan(TK))[0],
        np.where(np.isnan(Ts))[0],
    ])
)
print("Dropping {0} samples since they contain NaNs at relevant places. This is {1}% of the data, number of samples dropped. Applying z-mask and k-mask".format(len(nan_samples), np.round(len(nan_samples)/len(Deltak)*100),2))
zmask = np.array(z_array >= 7) & (z_array <= 26)
kmask = np.array(k_array >= 8.5e-2) & (k_array <= 1)

Deltak_HERA4 = np.delete(Deltak, nan_samples, axis=0)[:,zmask,:][:,:,kmask]

parameters = np.delete(parameters, nan_samples, axis=0)

Trad_HERA4 = np.delete(Trad, nan_samples, axis=0)[:,zmask]

TK_HERA4 = np.delete(TK, nan_samples, axis=0)[:,zmask]

Ts_HERA4 = np.delete(Ts, nan_samples, axis=0)[:,zmask]

zarr = z_array[zmask]
karr = k_array[kmask]

def PT12_to_PL9(PT12):
    # Convert 12 parameters to usually used parameters, with appropriate logs
    features = np.zeros((PT12.shape[0], 9))
    
    features[:,0] = np.log(PT12[:,0]) #fstarII
    features[:,1] = np.log10(PT12[:,1]) #fStarIII
    features[:,2] = np.log10(PT12[:,2]) #Vc
    features[:,3] = np.log10(PT12[:,3]) #fX
    features[:,4] = PT12[:,4] #alpha
    features[:,5] = PT12[:,5] #nu_0
    #features[:,5] = PT10[:,5] #zeta
    features[:,6] = PT12[:,7] #tau
    features[:,7] = np.log10(PT12[:,8]) #fradio
    features[:,8] = PT12[:,9] #pop
    #features[:,9] = PT12[:,10] #feed
    #features[:,10] = PT12[:,11] #delay
    return features


parameters_HERA4 = PT12_to_PL9(parameters)

#with some extras
#Pk_HERA4, [PT, Trad_HERA4, TK_HERA4, T21_HERA4, xA_HERA4, xHI_HERA4, SFR_HERA4, Ts_HERA4, fII_HERA4, JA_HERA4, Mcrit_HERA4, SFRII_HERA4, SFRIII_HERA4, XRB_HERA4] = remove_powerspectra_nans(Pk, [PT, Trad, TK, T21, xA, xHI, SFR, Ts, fII, JA, Mcrit, SFRII, SFRIII, XRB])
#PL_HERA4 = PT9_to_PL8(PT)
#print(Deltak_HERA4.shape,parameters_HERA4.shape ,Trad_HERA4.shape, TK_HERA4.shape, Ts_HERA4.shape, zarr.shape)

# emulators
#Pk_emu = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_adaptive.pkl",preprocesss_log_x=False)
#TS_emu = poweremu(loadfile="data/trained_emulators_poweremu/TS_emu_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#TK_emu = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#TR_emu = poweremu(loadfile="data/trained_emulators_poweremu/TR_emu_converged.pkl", preprocesss_log_x=False, offset=1e-3)
#SFR_emu = poweremu(loadfile="data/trained_emulators_poweremu/SFR_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#extras
#xA_emu = poweremu(loadfile="data/trained_emulators_poweremu/xA_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#xHI_emu = poweremu(loadfile="data/trained_emulators_poweremu/xHI_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#fII_emu = poweremu(loadfile="data/trained_emulators_poweremu/fII_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#JA_emu = poweremu(loadfile="data/trained_emulators_poweremu/JA_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#Mcrit_emu = poweremu(loadfile="data/trained_emulators_poweremu/Mcrit_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#SFRII_emu = poweremu(loadfile="data/trained_emulators_poweremu/SFRII_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#SFRIII_emu = poweremu(loadfile="data/trained_emulators_poweremu/SFRIII_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)
#XRB_emu = poweremu(loadfile="data/trained_emulators_poweremu/XRB_emu_converged.pkl", preprocesss_log_x=False, offset=1e-25)

# Training data
# Little h for wave number conversions, use h from simulation
h=0.6704
n_over = 100

#Delta emulator
if False: #(rank==0) or (size==1):
    parameters_HERA4_train, parameters_HERA4_test, Deltak_HERA4_train, Deltak_HERA4_test = train_test_split(parameters_HERA4, Deltak_HERA4, test_size=0.2, random_state=42)
    print(parameters_HERA4_train.shape, Deltak_HERA4_train.shape)
    print("Checkpoint 1.0: Gen PS training, rank={0}".format(rank), flush=True)
    train_x, train_y = gen_training(n_over=n_over, params=parameters_HERA4_train, data=Deltak_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), klow=karr.min()/h, khigh=karr.max()/h)
    train_y[train_y<1] = 1
   
    if False:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        #emu = make_pipeline(StandardScaler(), MLPRegressor(
        #        # Changeable non-defaults
        #        hidden_layer_sizes=layers, max_iter=9999, tol=1e-9,
        #        # Mandatory non-defaults
        #        verbose=True, validation_fraction=0, warm_start=True,
        #        # Defaults
        #        #activation='relu', early_stopping=False,
        #        #alpha=1e-4, solver='adam', learning_rate='constant',
        #        #**kwargs
        #))
        offset = 0
        tol = 1e-4
        input_1 = train_x
        output_1 = np.log(train_y+offset)
        mlp = MLPRegressor(
        max_iter=50, tol=tol,
        #hidden_layer_sizes=layers,
        validation_fraction=0, warm_start=True,
        #verbose=True, 
        )
        parameter_space = {
        'hidden_layer_sizes': [(100,100,100,100), (200, 200, 200, 200), (400, 300, 200, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
        }
        from sklearn.model_selection import GridSearchCV
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=3)
        clf.fit(input_1,output_1)
        #Best paramete set
        print('Best parameters found:\n', clf.best_params_, flush=True)
        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), flush=True)
        np.save("gridsearch_results_2.npy", clf.cv_results_)

    if True:
        # Train & Save
        layers = (100, 100, 100, 100) #(100, 30, 10, 5)
        tol = 1e-5#1e-5
        offset = 1#1e-8#1
        emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=310, solver="adam", offset=offset,tol=tol) #offset 10-100
        #with open('output_train.txt', 'a+') as f:
            #f.write("Checkpoint 1.1: PS training\n")
        print("Checkpoint 1.1: PS training", flush=True)
        emu.train(train_x, train_y)
        emu.save("data/trained_emulators_poweremu/Deltasq_emu_PL9_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol), str(offset)))
########################################################################

#XRB emulator
if True:
    XRB = load_files("data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_XRB_", name="hera", key='combined_XRBs', endings=["mat"])
    nu_keV = load_files("data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_nu_", name="hera", key='nu_keV', endings=["mat"])[0]
    nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.85)
    XRB = XRB[:,nu_mask]
    nu_keV = nu_keV[nu_mask]
    
    XRB_HERA4 = np.delete(XRB, nan_samples, axis=0)
    parameters_HERA4_train, parameters_HERA4_test, XRB_HERA4_train, XRB_HERA4_test = train_test_split(parameters_HERA4, np.log(XRB_HERA4), test_size=0.2, random_state=42)
    print("Checkpoint: XRB generate training set", flush=True)
    train_x, train_y = gen_training_1d(n_over=500, params=parameters_HERA4_train, data=XRB_HERA4_train, zlow=np.log( nu_keV ).min(), zhigh=np.log( nu_keV ).max(), zarr=np.log(nu_keV) )
    train_y = np.exp(train_y)
    layers = (100, 100, 100, 100)
    tol = 1e-5
    offset = 0
    emu = poweremu(loadfile=None,preprocesss_log_x=False, preprocess_y=True, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=0,tol=tol)
    print("Checkpoint: XRB training", flush=True)
    emu.train(train_x, train_y)
    emu.save("data/trained_emulators_poweremu/XRB_emu_PL9_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(500, layers[0], layers[1], layers[2], layers[3], str(tol), str(offset)))


#Trad emulator
if False: #(rank==1) or (size==1):
    Trad_noNaNs = Trad_HERA4#[:,:31]
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, Trad_noNaNs, test_size=0.2, random_state=42)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 2.0: Gen Trad training\n")
    print("Checkpoint 2.0: Gen Trad training, rank={0}".format(rank), flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    #ptrain_x, ptrain_y = gen_training_1d(1, PL_HERA4_train, T_HERA4_train, fix_z=8, zarr=zarr)
    #test_x, test_y = gen_training_1d(10, PL_HERA4_test, T_HERA4_test, seed=0, zarr=zarr)
    #ptest_x, ptest_y = gen_training_1d(1, PL_HERA4_test, T_HERA4_test, seed=1, fix_z=8, zarr=zarr)
    # Train & Save
    layers = (100, 100, 100, 100)#
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 2.1: Trad training\n")
    print("Checkpoint 2.1: Trad training", flush=True)
    emu.train(train_x, train_y)
    emu.save("data/trained_emulators_poweremu/Trad_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T)))

#TK emulator
if False: #(rank==2) or (size==1):
    TK_noNaNs = TK_HERA4#[:,:31]
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, TK_noNaNs, test_size=0.2, random_state=42)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 3.0: Gen TK training\n")
    print("Checkpoint 3.0: Gen TK training, rank={0}".format(rank), flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    #ptrain_x, ptrain_y = gen_training_1d(1, PL_HERA4_train, T_HERA4_train, fix_z=8, zarr=zarr)
    #test_x, test_y = gen_training_1d(10, PL_HERA4_test, T_HERA4_test, seed=0, zarr=zarr)
    #ptest_x, ptest_y = gen_training_1d(1, PL_HERA4_test, T_HERA4_test, seed=1, fix_z=8, zarr=zarr)
    layers = (100, 100, 100, 100)#(100, 30, 10, 5)
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 3.1: TK training\n")
    print("Checkpoint 3.1: TK training", flush=True)
    emu.train(train_x, train_y)
    emu.save("data/trained_emulators_poweremu/TK_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T)))

#Ts emulator
if False: #(rank==3) or (size==1):
    Ts_noNaNs = Ts_HERA4#[:,:31]
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, Ts_noNaNs, test_size=0.2, random_state=42)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 4.0: Gen Ts training\n")
    print("Checkpoint 4.0: Gen Ts training, rank={0}".format(rank), flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    #ptrain_x, ptrain_y = gen_training_1d(1, PL_HERA4_train, T_HERA4_train, fix_z=8, zarr=zarr)
    #test_x, test_y = gen_training_1d(10, PL_HERA4_test, T_HERA4_test, seed=0, zarr=zarr)
    #ptest_x, ptest_y = gen_training_1d(1, PL_HERA4_test, T_HERA4_test, seed=1, fix_z=8, zarr=zarr)
    layers = (100, 100, 100, 100)#(100, 30, 10, 5)
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    #with open('output_train.txt', 'a+') as f:
        #f.write("Checkpoint 4.1: Ts training\n")
    print("Checkpoint 4.1: Ts training", flush=True)
    emu.train(train_x, train_y)
    emu.save("data/trained_emulators_poweremu/Ts_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T) ))

print("Finished training emulator. Evaluating quality...", flush=True)


def calculate_accuracy(emu, test_x, test_y, add_rsd=None):
    if add_rsd is None:
        pred_y = emu.predict(test_x)
    else:
        pred_y = emu.predict(np.hstack((test_x, np.ones([len(test_x), 1])*add_rsd)))

    deltas = np.log10((test_y+offset)/(pred_y+offset))
    return deltas

def score(emu, test_x, test_y, add_rsd=None):
    deltas = calculate_accuracy(emu, test_x, test_y, add_rsd=add_rsd)
    limit68 = 10**confidence_level(deltas, level=0.68)
    limit95 = 10**confidence_level(deltas, level=0.95)
    limit997 = 10**confidence_level(deltas, level=0.997)
    print("68% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit68-1)), np.sum(np.abs(limit68-1)))+"\n95% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit95-1)), np.sum(np.abs(limit95-1)))+"\n99.7% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit997-1)), np.sum(np.abs(limit997-1)))+"\n(assuming {0} mK² level threshold; +% means test>pred)".format(offset))
    return limit68, limit95, limit997

def hist(emu, test_x, test_y, add_rsd=None):
    deltas = calculate_accuracy(emu, test_x, test_y, add_rsd=add_rsd)
    plt.hist(deltas, bins=100)
    plt.show()
    return deltas

def zkmap(emu, full_x, full_y, zmin=7, zmax=26, add_rsd=None):
    # Make a colormap showing emulator error as a function of k and z
    def test_emu_kz(emu,z,k, PL=full_x, Pk=full_y):
        test_x, test_y = gen_training(1, PL, Pk, seed=0, fix_k=k, fix_z=z)
        test_y[test_y<1] = 1 #Added by SP
        limit68, limit95, limit997 = score(emu, test_x, test_y, add_rsd=add_rsd)
        return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
    # Compute values
    zarr = z_array[zmask][:] #z_array[0][ np.where((z_array[0] > zmin) & (z_array[0] < zmax)) ]
    karr = k_array[kmask][:]/h #k_array[0][ np.where((k_array[0] > 5e-2) & (k_array[0] < 1)) ][::3]
    #zarr = np.arange(zmin, zmax+0.1, 1)
    #karr = np.arange(4e-2, 1, 0.1)
    tarr1 = np.ones([len(zarr), len(karr)])
    tarr2 = np.ones([len(zarr), len(karr)])
    tarr3 = np.ones([len(zarr), len(karr)])
    for i in range(len(zarr)):
        for j in range(len(karr)):
            print("Index ({0},{1}). Progress: {2}/{3} \n z={4}, k [h/cMpc]={5}".format(i,j, np.ravel_multi_index((i,j),(zarr.size,karr.size)), zarr.size*karr.size,zarr[i], np.round(karr[j],4) ))
            tarr1[i,j], tarr2[i,j], tarr3[i,j] = test_emu_kz(emu,zarr[i],karr[j])#added /h
    # Make plot
    zax, kax = make_axes_pcolor(zarr, np.array(karr))
    #print(karr,kax)
    plt.subplot(311)
    plt.suptitle("Emulator average CL size")#s (e.g. +15/-5% is 0.1)")
    plt.title("68% CLs")
    plt.pcolormesh(zarr, karr, tarr1.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr1.T)
    plt.ylabel("k [h/cMpc]")
    plt.colorbar()
    plt.subplot(312)
    plt.title("95% CLs")
    plt.pcolormesh(zarr, karr, tarr2.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr2.T)
    plt.ylabel("k [h/cMpc]")
    plt.colorbar()
    plt.subplot(313)
    plt.title("99.7% CLs")
    plt.pcolormesh(zarr, karr, tarr3.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr3.T)
    plt.xlabel("Redshift z")
    plt.ylabel("k [h/cMpc]")
    plt.colorbar()
    #plt.show()
    #plt.savefig("zkmap.png")

#emu = poweremu(loadfile="data/trained_emulators_poweremu/Deltasq_emu_n400_l100100100100_t1e-06_o0.pkl", tol=1e-6, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
#zkmap(emu=emu, full_x=parameters_HERA4, full_y=Deltak_HERA4, zmin=7, zmax=26)
#plt.savefig("zkmap_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T)))

#def zkmap(emu=Pk_emu_m_Sims, full_x=PL_Sims_test, full_y=Pk_Sims_test, zmin=7, zmax=11, add_rsd=None):
#    # Make a colormap showing emulator error as a function of k and z
#    def test_emu_kz(emu,z,k, PL=full_x, Pk=full_y):
#        test_x, test_y = gen_training(1, PL, Pk, seed=0, fix_k=k, fix_z=z)
#        limit68, limit95, limit997 = score(emu, test_x, test_y, add_rsd=add_rsd)
#        return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
        # Compute values

#score(TR_emu0, ptest_x[:,1:], ptest_y) #0.09
#score(TR_emu1, ptest_x, ptest_y) #0.10
#score(TR_emu1, test_x, test_y) #0.06
#score(TS_emu0, ptest_x[:,1:], ptest_y) #0.10
#score(TS_emu1, ptest_x, ptest_y) #0.16 0.12
#score(TS_emu1, test_x, test_y) #0.10
#score(TK_emu0, ptest_x[:,1:], ptest_y) #0.10
#score(TK_emu1, ptest_x, ptest_y) #0.20 0.1...
#score(TK_emu1, test_x, test_y) #0.09


#zkmap()
#zkmap(emu=Pk_emu_RadLyA_m, full_x=PL_RSD_Itamar, full_y=Pk_RSD_Itamar, add_rsd=True)
#zkmap(emu=Pk_emu_RadLyA_m4, full_x=PL_RSD_Itamar, full_y=Pk_RSD_Itamar, add_rsd=True, zmax=30)


# Make a new emulator

## Adaptive. Temperature:
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers,
#    max_iter=9999, learning_rate="adaptive", solver="sgd", n_iter_no_change=5,
#    tol=0.00001, offset=offset)
# currently m2 converged runs forgot offset! Otherwise really good after ~75 it (TS)
# emu_m3_converged done
#emu.save("data/trained_emulators_poweremu/"+key+"emu_m3_converged.pkl")
#emu.train(train_x, train_y)
#T21
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers,
#    max_iter=9999, learning_rate="adaptive", solver="sgd", n_iter_no_change=5,
#    tol=0.00001, offset=1e-3, preprocess_y=False)
#emu.train(train_x, train_y)
#emu.save("data/trained_emulators_poweremu/T_emu_test.pkl")
# v1 converged: Good models with

## Constant

#layers = (100, 30, 10, 5)
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=1e-3,tol=1e-5)

# Train & Save
#emu.train(ptrain_x, ptrain_y)
#emu.save("data/trained_emulators_poweremu/Pk_emu_test.pkl")

#emu.train(mtrain_x, mtrain_y)
#emu.save("data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl")


# Discussion, here focused on RadLyA models:
# Question: What's the maximum we can achieve at fix_k=0.192, fix_z=8? [emu_f]
#               68% samples within +14% / -8% of true --> 0.22
#               95% samples within +44% / -33% of true --> 0.77
#           Do we achieve this with the general emulator? [emu_a]
#               68% samples within +24% / -5% of true --> 0.30
#               95% samples within +59% / -19% of true --> 0.77
#           No. Let's run an emulator with 1k oversampling [emu_m]
#               68% samples within +13% / -8% of true --> 0.21
#               95% samples within +54% / -19% of true --> 0.73
#           Awesome! Can we increase the z bounds and still achieve this?
#           Now with 1k oversampling up to z=21, converged @ 232 iterations
#               68% samples within +15% / -9% of true --> 0.24
#               95% samples within +61% / -19% of true --> 0.80
#           That's alright for now --> Do SARAS+HERA tests with Pk_emu_evenmorez_RadLyA_adaptive.
#           Ah wait need to z=31. Here we go Pk_emu_m3_RadLyA_adaptive.pkl @ 234 converged
#               68% samples within +22% / -11% of true --> 0.33
#               95% samples within +68% / -30% of true --> 0.98
#           Hmm OK. But give it one more try with less k space (m4train) and slightly larger layers (150, 50, 15, 5)
#               ("data/trained_emulators_poweremu/Pk_emu_m4_RadLyA_adaptive.pkl")
#               68% samples within +16% / -10% of true --> 0.26
#               95% samples within +54% / -22% of true --> 0.77
#           Yeah here we go.

#           Let's apply this to Sims though! Let's see what is the best score we can get with fixed k & z:
#               (100, 30, 10, 5), adaptive
#                   68% samples within +23% / -15% of true --> 0.38
#                   95% samples within +87% / -53% of true --> 1.40
#                   99.7% samples within +529% / -82% of true --> 6.11
#               (200, 100, 50, 25), adaptive
#                   68% samples within +18% / -14% of true --> 0.32
#                   95% samples within +87% / -45% of true --> 1.32
#                   99.7% samples within +282% / -79% of true --> 3.61
#               repeat
#                   68% samples within +18% / -15% of true --> 0.33
#                   95% samples within +75% / -48% of true --> 1.24
#                   99.7% samples within +190% / -74% of true --> 2.64
#               Pk_emuL_fixkz_Sims_adaptive
#               (200, 100, 50, 25), adam
#                   68% samples within +15% / -11% of true --> 0.26
#                   95% samples within +60% / -38% of true --> 0.98
#                   99.7% samples within +281% / -69% of true --> 3.51
#               repeat (save as Pk_emuL_fixkz_Sims_adam_2001005025)
#                   68% samples within +19% / -11% of true --> 0.30
#                   95% samples within +65% / -42% of true --> 1.07
#                   99.7% samples within +206% / -80% of true --> 2.86
#               repeat (save as Pk_emuL_fixkz_Sims_adam_2001005025_v2)
#                   68% samples within +17% / -8% of true --> 0.26
#                   95% samples within +67% / -37% of true --> 1.04
#                   99.7% samples within +323% / -75% of true --> 3.98
#              (400, 200, 100, 50), adam
#                  68% samples within +13% / -9% of true --> 0.22
#                  95% samples within +57% / -35% of true --> 0.92
#                  99.7% samples within +179% / -86% of true --> 2.66
#           Okay, (200, 100, 50, 25) w/ adam is the score to beat, saved as Pk_emuL_fixkz_Sims_adam_2001005025_v2
#           Run Sims emulator aiming for 68% ~ 0.26 and 95% ~ 1 scores.
#               Here we go! (100, 30, 10, 5) layers with adaptive sgd for 234 iterations, trained on mtrain
#               gives excellent numbers, took ages. Pk_emu_m_Sims_adaptive.pkl
#                   68% samples within +14% / -10% of true --> 0.24
#                   95% samples within +55% / -30% of true --> 0.85
#                   99.7% samples within +229% / -75% of true --> 3.04



#def model_of_k(z, karr, p, emu=None):
#    par0 = np.array([z, np.NaN, *p])
#    params=np.tile(par0, (len(karr), 1))
#    params[:,1] = karr
#    return emu.predict(params)

#def model_of_z(zarr, k, p, emu=None):
#    par0 = np.array([np.NaN, k, *p])
#    params=np.tile(par0, (len(karr), 1))
#    params[:,0] = zarr
#    return emu.predict(params)

#todo: make some plots of emulators

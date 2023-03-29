path="/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

from codes.emulator_poweremu import *
from codes.loader_21cmSim import *
from codes.tools import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from copy import deepcopy
import seaborn as sns
ccb = sns.color_palette("colorblind")
matplotlib.use('Agg')
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

z_array = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_z_", name="hera", key='z21cm', endings=["mat"])[0]
k_array = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_k_", name="hera", key='ks', endings=["mat"])[0]

Deltak = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_Deltak_", name="hera", key='combined_Deltaks', endings=["mat"])
parameters = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_parameters_", name="hera", key='parameters', endings=["mat"])
Trad = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TradLOS_", name="hera", key='combined_TradLOSs', endings=["mat"])
TK = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TK_", name="hera", key='combined_TKs', endings=["mat"])
Ts = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_Ts_", name="hera", key='combined_Tss', endings=["mat"])

XRB = load_files(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_XRB_", name="hera", key='combined_XRBs', endings=["mat"])
nu_keV = load_files(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_nu_", name="hera", key='nu_keV', endings=["mat"])[0]

SFR = load_files(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_SFR_", name="hera", key='combined_SFRs', endings=["mat"])

#Remove data where temperatures have NaNs and mask zrange and krange
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
nu_mask = (nu_keV >0.4) & (nu_keV <55)#8.85)

Deltak_HERA4 = np.delete(Deltak, nan_samples, axis=0)[:,zmask,:][:,:,kmask]
parameters = np.delete(parameters, nan_samples, axis=0)

Trad_HERA4 = np.delete(Trad, nan_samples, axis=0)[:,zmask]
TK_HERA4 = np.delete(TK, nan_samples, axis=0)[:,zmask]
Ts_HERA4 = np.delete(Ts, nan_samples, axis=0)[:,zmask]

XRB_HERA4 = np.delete(XRB[:,nu_mask], nan_samples, axis=0)
SFR_HERA4 = np.delete(SFR, nan_samples, axis=0)

zarr = z_array[zmask]
karr = k_array[kmask]
nu_keV = nu_keV[nu_mask]

def PT12_to_PL9(PT12):
    # Convert 12 parameters to usually used parameters, with appropriate logs
    features = np.zeros((PT12.shape[0], 9))

    features[:,0] = np.log(PT12[:,0]) #fstarII
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


#Little-h to convert to h / cMpc
h=0.6704

n_over = 400

#Delta power spectrum emulator
if False: #(rank==0) or (size==1):
    parameters_HERA4_train, parameters_HERA4_test, Deltak_HERA4_train, Deltak_HERA4_test = train_test_split(parameters_HERA4, Deltak_HERA4, test_size=0.2, random_state=42)
    #print("Checkpoint 1.0: Gen PS training, rank={0}".format(rank), flush=True)
    print("Checkpoint 1.0: Gen PS training", flush=True)
    train_x, train_y = gen_training(n_over=n_over, params=parameters_HERA4_train, data=Deltak_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), klow=karr.min()/h, khigh=karr.max()/h)
    #Truncate data points below 1mK^2 to 1 for a better emulator.
    train_y[train_y<1] = 1

    # Train & Save
    layers = (100, 100, 100, 100) #(100, 100, 100, 100)
    tol = 1e-6 #1e-6
    offset = 0 #1
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=310, solver="adam", offset=offset,tol=tol) #offset 10-100
    print("Checkpoint 1.1: PS training", flush=True)
    emu.train(train_x, train_y)
    #emu.save("data/trained_emulators_poweremu/Deltasq_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol), str(offset)))
########################################################################

#XRB emulator
if True:
    parameters_HERA4_train, parameters_HERA4_test, XRB_HERA4_train, XRB_HERA4_test = train_test_split(parameters_HERA4, np.log(XRB_HERA4), test_size=0.2, random_state=42)
    print("Checkpoint: XRB generate training set", flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=parameters_HERA4_train, data=XRB_HERA4_train, zlow=np.log( nu_keV ).min(), zhigh=np.log( nu_keV ).max(), zarr=np.log(nu_keV) )
    train_y = np.exp(train_y) #samples drawn in logspace and convert back
    layers = (100, 100, 100, 100)
    tol = 1e-5#1e-5
    offset = 0
    emu_XRB = poweremu(loadfile=None,preprocesss_log_x=False, preprocess_y=True, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=0,tol=tol)
    print("Checkpoint: XRB training", flush=True)
    emu_XRB.train(train_x, train_y)
    emu_XRB.save(path + "data/trained_emulators_poweremu/XRB_emu_PL9_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(500, layers[0], layers[1], layers[2], layers[3], str(tol), str(offset)))

#SFR emulator
if True:
    redshifts = np.array([6,7,8])
    parameters_HERA4_train, parameters_HERA4_test, SFR_HERA4_train, SFR_HERA4_test = train_test_split(parameters_HERA4, SFR_HERA4[:,:redshifts.size], test_size=0.2, random_state=42)
    print("Checkpoint: SFR generate training set", flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=parameters_HERA4_train, data=SFR_HERA4_train, zlow=redshifts.min(), zhigh=redshifts.max(), zarr=redshifts )
    layers = (100, 100, 100, 100)
    tol = 1e-5#1e-5
    offset = 0
    emu_SFR = poweremu(loadfile=None,preprocesss_log_x=False, preprocess_y=True, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=0,tol=tol)
    print("Checkpoint: XRB training", flush=True)
    emu_SFR.train(train_x, train_y)
    emu_SFR.save(path + "data/trained_emulators_poweremu/SFR_emu_PL9_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(500, layers[0], layers[1], layers[2], layers[3], str(tol), str(offset)))

#Trad emulator
if False: #(rank==1) or (size==1):
    Trad_noNaNs = Trad_HERA4#[:,:31]
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, Trad_noNaNs, test_size=0.2, random_state=42)
    #print("Checkpoint 2.0: Gen Trad training, rank={0}".format(rank), flush=True)
    print("Checkpoint 2.0: Gen Trad training", flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    # Train & Save
    layers = (100, 100, 100, 100)#
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    print("Checkpoint 2.1: Trad training", flush=True)
    emu.train(train_x, train_y)
    #emu.save("data/trained_emulators_poweremu/Trad_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T)))

#TK emulator
if False: #(rank==2) or (size==1):
    TK_noNaNs = TK_HERA4
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, TK_noNaNs, test_size=0.2, random_state=42)
    #print("Checkpoint 3.0: Gen TK training, rank={0}".format(rank), flush=True)
    print("Checkpoint 3.0: Gen TK training", flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    layers = (100, 100, 100, 100)
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    print("Checkpoint 3.1: TK training", flush=True)
    emu.train(train_x, train_y)
    #emu.save("data/trained_emulators_poweremu/TK_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T)))

#Ts emulator
if False: #(rank==3) or (size==1):
    Ts_noNaNs = Ts_HERA4#[:,:31]
    PL_HERA4_train, PL_HERA4_test, T_HERA4_train, T_HERA4_test = train_test_split(parameters_HERA4, Ts_noNaNs, test_size=0.2, random_state=42)
    #print("Checkpoint 4.0: Gen Ts training, rank={0}".format(rank), flush=True)
    print("Checkpoint 4.0: Gen Ts training", flush=True)
    train_x, train_y = gen_training_1d(n_over=n_over, params=PL_HERA4_train, data=T_HERA4_train, zlow=zarr.min(), zhigh=zarr.max(), zarr=zarr)
    layers = (100, 100, 100, 100)
    tol_T = 1e-5
    offset_T = 1e-3
    emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam", offset=offset_T,tol=tol_T)
    print("Checkpoint 4.1: Ts training", flush=True)
    emu.train(train_x, train_y)
    #emu.save("data/trained_emulators_poweremu/Ts_emu_n{0}_l{1}{2}{3}{4}_t{5}_o{6}.pkl".format(n_over, layers[0], layers[1], layers[2], layers[3], str(tol_T), str(offset_T) ))


print("Finished training emulator. Evaluating quality...", flush=True)

offset = 0

def calculate_accuracy(emu, test_x, test_y):
    pred_y = emu.predict(test_x)
    deltas = 1-(test_y+offset)/(pred_y+offset)
    return deltas

def score(emu, test_x, test_y):
    deltas = calculate_accuracy(emu, test_x, test_y,)
    limit68 = confidence_level(deltas, level=0.68)
    limit95 = confidence_level(deltas, level=0.95)
    limit997 = confidence_level(deltas, level=0.997)
    print("68% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit68-1)), np.sum(np.abs(limit68-1)))+
    "\n95% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit95-1)), np.sum(np.abs(limit95-1)))+
    "\n99.7% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit997-1)), np.sum(np.abs(limit997-1)))+
    "\n(assuming {0} mK² level threshold; +% means test>pred)".format(offset))
    return limit68, limit95, limit997


def errmap2d(emu, full_x, full_y, x_axis=np.arange(7,26.01), y_axis=np.logspace(-1,0), skip=0, filename=path+"zkmap.png"):
    # Make a colormap showing emulator error as a function of k and z
    def test_emu_2d(emu, x, y, PL=full_x, Pk=full_y):
        test_x, test_y = gen_training(1, PL, Pk, seed=0, fix_k=y, fix_z=x)
        test_y[test_y<1] = 1 #Added by SP
        limit68, limit95, limit997 = score(emu, test_x, test_y)
        return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
    # Compute values
    xarr = x_axis[::skip+1] 
    yarr = y_axis[::skip+1]
    tarr1 = np.ones([len(xarr), len(yarr)])
    tarr2 = np.ones([len(xarr), len(yarr)])
    tarr3 = np.ones([len(xarr), len(yarr)])
    for i in range(len(xarr)):
        for j in range(len(yarr)):
            print("Index ({0},{1}). Progress: {2}/{3} \n z={4}, k [h/cMpc]={5}".format(i,j, np.ravel_multi_index((i,j),(xarr.size,yarr.size)), xarr.size*yarr.size,xarr[i], np.round(yarr[j],4) ))
            tarr1[i,j], tarr2[i,j], tarr3[i,j] = test_emu_2d(emu=emu, x=xarr[i],y=yarr[j])#added /h
    # Make plot
    #zax, kax = make_axes_pcolor(zarr, np.array(karr))
    #print(karr,kax)
    plt.subplot(311)
    plt.suptitle("Emulator average CL sizes (e.g. +15/-5% is 0.1)")
    plt.title("68% CLs")
    plt.pcolormesh(xarr, yarr, tarr1.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr1.T)
    plt.ylabel("Wavenumber k h/cMpc")
    plt.colorbar()
    plt.subplot(312)
    plt.title("95% CLs")
    plt.pcolormesh(xarr, yarr, tarr2.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr2.T)
    plt.ylabel("Wavenumber k h/cMpc")
    plt.colorbar()
    plt.subplot(313)
    plt.title("99.7% CLs")
    plt.pcolormesh(xarr, yarr, tarr3.T)
    #plt.pcolormesh(zax, np.array(kax)/h, tarr3.T)
    plt.xlabel("Redshift z")
    plt.ylabel("Wavenumber k h/cMpc")
    plt.colorbar()
    #plt.show()
    plt.savefig(filename)

emu_dsq = poweremu(loadfile=path + "data/trained_emulators_poweremu/Deltasq_emu_PL9_n1000_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
parameters_HERA4_train, parameters_HERA4_test, Deltak_HERA4_train, Deltak_HERA4_test = train_test_split(parameters_HERA4, Deltak_HERA4, test_size=0.2, random_state=42)
errmap2d(emu=emu_dsq, full_x=parameters_HERA4_test, full_y=Deltak_HERA4_test, x_axis=z_array[zmask], y_axis=k_array[kmask]/h, skip=0)


def errmap1d(emu, full_x, full_y, x_axis=[6,7,8], skip=0, 
    xlabel=r"$Redshift\ z$", ylabel=r"$1 - \rm{SFR_{sim}\ /\ SFR_{emu}}\ [Unitless]$",
    filename="plot.png"):

    # Make a plot showing emulator error as a function of nu
    x_arr = np.array(x_axis[::skip+1]) #np.log(nu_keV[::5])
    def test_emu_1d(emu, x, PL=full_x, Pk=full_y):
        test_x, test_y = gen_training_1d(1, PL, Pk, seed=0, fix_z=x, zarr=x_arr)
        limit68, limit95, limit997 = score(emu, test_x, test_y)
        return limit68, limit95, limit997
    # Compute values
    tarr1 = np.ones(shape=(len(x_arr),2))
    tarr2 = np.ones(shape=(len(x_arr),2))
    tarr3 = np.ones(shape=(len(x_arr),2))
    for i in range(len(x_arr)):
        tarr1[i,:], tarr2[i,:], tarr3[i,:] = test_emu_1d(emu, x=x_arr[i])#added /h
        print("Index ({0}). Progress: {0}/{1} \n log(nu)={2:.2f} {3}".format(i, x_arr.size, x_arr[i], tarr1[i,:]))
    # Make plot
    fig,ax = plt.subplots(1,1,figsize=(8,6),)
    ax.fill_between(x_arr, *tarr3.T , fc=ccb[2], edgecolor="k", linestyle="solid", linewidth=1, label="99.7% CL")
    ax.fill_between(x_arr, *tarr2.T , fc=ccb[1], edgecolor="k", linestyle="solid", linewidth=1, label="95% CL")
    ax.fill_between(x_arr, *tarr1.T , fc=ccb[0], edgecolor="k", linestyle="solid", linewidth=1, label="68% CL")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(loc="upper right")
    
    #plt.show()
    plt.savefig(filename)
    #plt.close(fig)
    return tarr1, tarr2, tarr3


#emu_SFR = poweremu(loadfile=path + "data/trained_emulators_poweremu/SFR_emu_PL9_n60_l40404040_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
parameters_HERA4_train, parameters_HERA4_test, SFR_HERA4_train, SFR_HERA4_test = train_test_split(parameters_HERA4, SFR_HERA4[:,:3], test_size=0.2, random_state=42)
xlabel=r"$Redshift\ z$"
ylabel=r"$1 - \rm{SFR_{sim}\ /\ SFR_{emu}}\ [Unitless]$"
tarr1, tarr2, tarr3 = errmap1d(emu=emu_SFR, full_x=parameters_HERA4_test, full_y=SFR_HERA4_test, x_axis=[6,7,8], skip=0, xlabel=xlabel, ylabel=ylabel, filename=path+"SFR.png")

#emu_XRB = poweremu(loadfile=path + "data/trained_emulators_poweremu/XRB_emu_PL9_n500_l100100100100_t1e-05_o0.pkl", tol=1e-5, n_iter_no_change=99999, preprocesss_log_x=False, offset=0)
parameters_HERA4_train, parameters_HERA4_test, XRB_HERA4_train, XRB_HERA4_test = train_test_split(parameters_HERA4, XRB_HERA4, test_size=0.2, random_state=42)
xlabel = r"$\log \nu$"
ylabel = r"$1 - \rm{XRB_{sim}\ /\ XRB_{emu}}\ [Unitless]$"
tarr1, tarr2, tarr3 = errmap1d(emu=emu_XRB, full_x=parameters_HERA4_test, full_y=XRB_HERA4_test[:,::1], x_axis=np.log(nu_keV[::1]), skip=0, xlabel=xlabel, ylabel=ylabel, filename=path+"XRB.png")


path = '/home/sp2053/rds/hpc-work/powerspectra_analysis/'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import scipy.interpolate as sip
import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
import matplotlib.pyplot as plt
from codes.tools import *

















Trad = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TradLOS_", name="hera", key='combined_TradLOSs', endings=["mat"])
TK = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_TK_", name="hera", key='combined_TKs', endings=["mat"])
Ts = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_Ts_", name="hera", key='combined_Tss', endings=["mat"])
parameters = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_parameters_", name="hera", key='parameters', endings=["mat"])
z_array = load_files(path + 'data/models_21cmSim/HERA_IDR4_Emulator_Data/', middle="_z_", name="hera", key='z21cm', endings=["mat"])[0].astype(float)


nan_samples = np.unique(
    np.concatenate([
        np.where(np.isnan(Trad))[0],
        np.where(np.isnan(TK))[0],
        np.where(np.isnan(Ts))[0],
    ])
)

xHI_ = load_files(path + "data/models_21cmSim/HERA_IDR4_Emulator_Data/", middle="_xHI_", name="hera", key='combined_xHIs', endings=["mat"])
xHI = np.delete(xHI_, nan_samples, axis=0)
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
    #features[:,6] = PT10[:,6] #zeta
    features[:,6] = PT12[:,7] #tau
    features[:,7] = np.log10(PT12[:,8]) #fradio
    features[:,8] = PT12[:,9] #pop
    #features[:,9] = PT12[:,10] #feed
    #features[:,10] = PT12[:,11] #delay
    return features


parameters = PT12_to_PL9(parameters)


data_dir = path+"data/globalemu/xHI_emulator1/" #'downloaded_data/'
base_dir = data_dir+"results/"#'downloaded_data/'


parameters_train, parameters_test, xHI_train, xHI_test = train_test_split(parameters, xHI, test_size=0.34, random_state=42)

from scipy.ndimage import gaussian_filter1d as gf
def interpolate_and_smooth(input_x, input_y, interp_x, width=3, filter_type="gaussian"):
    """
    Interpolate and smoothen input data.

    Parameters:
    - input_x (array-like): Input x-coordinates.
    - input_y (array-like): Input y-coordinates.
    - interp_x (array-like): x-coordinates for interpolation.
    - width (int): Window size or sigma for smoothening.

    Returns:
    - smoothed_x (numpy.ndarray): Interpolated and smoothed x-coordinates.
    - smoothed_y (numpy.ndarray): Interpolated and smoothed y-coordinates.
    """

    
    
    extrapolation_fit = np.poly1d(np.polyfit(input_x[:2], input_y[:2], 1))
    extrapolation_range = np.arange(5, 6., 0.1, dtype=float).round(1)
    extrapolation = extrapolation_fit(extrapolation_range)
    
    input_x = np.array([*extrapolation_range, *input_x] )
    input_y = np.array([*extrapolation, *input_y])
    interp_x = np.array([*extrapolation_range, *interp_x])

    # Linear interpolation and Calculate interpolated y-values
    interpolated_function = sip.interp1d(input_x, input_y, kind='linear')
    interpolated_y = interpolated_function(interp_x)


    # Apply convolution to smoothen the data
    if filter_type == "boxcar":
        smoothing_kernel = np.ones(width) / width
        smoothed_y = np.convolve(interpolated_y, smoothing_kernel, mode='same')
    elif filter_type == "gaussian":
        weights = [np.zeros(width*2),np.ones(len(interpolated_y)-4*width),np.zeros(width*2)]
        smoothed_y = gf(interpolated_y, width, mode="mirror")#, weights=weights)
    elif filter_type == "none":
        smoothed_y = interpolated_y
    else:
        raise ValueError("filter_type must be either 'boxcar' or 'gaussian'.")
    return interp_x[len(extrapolation_range):], smoothed_y[len(extrapolation_range):]

"""
if True:
    max_z = np.where(z_array == 20)[0][0] + 1
    interp_z = np.arange(6, 20.1, 0.1, dtype=float).round(1)
    xHI_train_interp = np.zeros(shape=(len(xHI_train), len(interp_z)))
    xHI_test_interp = np.zeros(shape=(len(xHI_test), len(interp_z)))
    for i,x in enumerate(xHI_train):
        xHI_train_interp[i,:] = interpolate_and_smooth(z_array[:max_z], x[:max_z], interp_z, filter_type="gaussian", width=2)[1]
    for i,x in enumerate(xHI_test):
        xHI_test_interp[i,:] = interpolate_and_smooth(z_array[:max_z], x[:max_z], interp_z, filter_type="gaussian", width=2)[1]      

    np.savetxt(data_dir+"train_data.txt", parameters_train)
    np.savetxt(data_dir+"train_labels.txt", xHI_train_interp)
    np.savetxt(data_dir+"test_data.txt", parameters_test)
    np.savetxt(data_dir+"test_labels.txt", xHI_test_interp)


matching_z = np.any([interp_z==z for z in z_array[:max_z]],axis=0)


fig, ax = plt.subplots(3,1, sharex=True)
for i in np.arange(0, len(xHI_train), 100):
    
    ax[0].plot(z_array[:max_z], xHI_train[i,:max_z], c="k", alpha=0.1)
    ax[0].plot(interp_z, xHI_train_interp[i,:], c="r", alpha=0.1)
    
    ax[1].plot(interp_z[matching_z], 100*(xHI_train[i,:max_z] - xHI_train_interp[i,matching_z])/xHI_train[i,:max_z], c="k", alpha=0.1)
    ax[2].plot(interp_z[matching_z], (xHI_train[i,:max_z] - xHI_train_interp[i,matching_z]), c="k", alpha=0.1)   

quants = np.nanpercentile((xHI_train[:,:max_z] - xHI_train_interp[:,matching_z]), [2.5,97.5],axis=0)
for q in quants:
    ax[2].plot(interp_z[matching_z], q, c="r", alpha=0.5)

ax[1].set_ylim(-20,20)
ax[0].set_ylabel("xHI")
ax[1].set_ylabel("Percentage difference")
ax[2].set_ylabel("Difference")
ax[2].set_xlabel("Redshift z")

#plt.figure()

mask = (xHI_train[:,1] - xHI_train_interp[:,1])/xHI_train[:,1] > 0.1
print(mask.sum(), mask.sum()/len(mask))
plt.figure()
plt.plot(z_array[:max_z], xHI_train[mask,:max_z][:5].T, c="k", alpha=0.5)
plt.plot(interp_z, xHI_train_interp[mask,:][:5].T, c="r", alpha=0.5)

#plt.show()

#import matplotlib.pyplot as plt
#plt.plot(z_array[:15], xHI_train[0,:15])
#plt.show()

#np.savetxt(data_dir+"train_data.txt", parameters_train)
#np.savetxt(data_dir+"train_labels.txt", xHI_train[:, :23])
#np.savetxt(data_dir+"test_data.txt", parameters_test)
#np.savetxt(data_dir+"test_labels.txt", xHI_test[:, :23])
process("full", interp_z, base_dir=base_dir, data_location=data_dir, resampling=True, xHI = True, logs=[])#0,1,2,3,7])
nn(batch_size=128, epochs=2000, base_dir=base_dir, input_shape=9 + 1, layer_sizes=[20,20,20,20], activation="tanh", resume=False, early_stop=True, xHI = True)


"""
####emu test xHI####

from globalemu.eval import evaluate
from tensorflow import keras
import matplotlib.pyplot as plt

test_data = np.loadtxt(data_dir + 'test_data.txt')
test_labels = np.loadtxt(data_dir + 'test_labels.txt')

model = keras.models.load_model(base_dir+'model.h5', compile=False)
predictor = evaluate(base_dir=base_dir, model=model, gc=False, logs=[])#0,1,2,3,7])


#xHI_emu = np.zeros_like(test_labels)
#for i,p in enumerate(test_data):
#    xHI_emu[i,:], zs = predictor(p) 

#    if i % round((len(xHI_emu)/10)) == 0:
#        print("Progress: {0:.2f} %".format(i/len(xHI_emu)*100), flush=True)

#print("zs: ", zs)

interp_z = np.arange(6, 20.1, 0.1, dtype=float).round(1)




xHI_emu = np.load(base_dir + "test_label_emu.npy",)
#np.save(base_dir + "test_label_emu.npy",xHI_emu)
print(xHI_emu.shape, test_labels.shape, flush=True)
"""
diffs = 100*(test_labels - xHI_emu)/test_labels

diff = np.abs(diffs).flatten()
print(np.isnan(diff).sum()/diff.size)
print(np.isinf(diff).sum()/diff.size)
mask = np.stack([np.isinf(diff), np.isnan(diff)]).any(axis=0)



diff = np.log10(
    diff[~mask]
    )

plt.figure()
plt.hist(diff, bins=100, density=True, label="xHI emulator", alpha=0.8)
plt.axvline(np.mean(diff), color="red")
plt.axvline(np.mean(diff) - np.std(diff), color="red")
plt.axvline(np.mean(diff) + np.std(diff), color="red")
quants = np.percentile(diff, [16,84])
plt.vlines(x = quants, ymin=0, ymax=1.5, color="green", linestyles="dashed")
print("Quants: lower {0:.2f}, upper {1:.2f}\nstd: {2:.2f}".format(10**quants[0], 10**quants[1], 10**np.std(diff)), flush=True)
plt.xlabel("log10(Percentage difference)")

fig, ax = plt.subplots(3,1, sharex=True)
alpha = 0.5
for i in np.arange(0, len(test_labels), 100):
    ax[0].plot(interp_z, test_labels[i,:], c="k", alpha=alpha)
    ax[0].plot(interp_z, xHI_emu[i,:], c="r", alpha=alpha, ls="dashed")
    ax[1].plot(interp_z, diffs[i,:], c="k", alpha=alpha)    
    ax[2].plot(interp_z, (test_labels[i,:] - xHI_emu[i,:]), c="k", alpha=alpha)    

quants = np.nanpercentile(diffs, [16,84],axis=0)
quants_ = np.nanpercentile(test_labels - xHI_emu, [2.5,97.5],axis=0)

ax[1].plot(interp_z, quants.T,  c="r", alpha=0.5)
ax[2].plot(interp_z, quants_.T,  c="r", alpha=0.5)

ax[1].axhline(y=20, c="k", alpha=0.5)
ax[1].axhline(y=-20, c="k", alpha=0.5)
ax[1].set_ylim(-100,100)
ax[2].set_ylim(-0.05,0.05)
#ax[2].set_xlim(6,10)

ax[0].set_ylabel("xHI")
ax[1].set_ylabel("Percentage difference")
ax[2].set_ylabel("Difference")
ax[2].set_xlabel("Redshift z")
plt.show()

"""
predictor = evaluate(base_dir=base_dir, model=model, gc=False, logs=[], z=[7.6])#0,1,2,3,7])
plt.figure()
for i in np.arange(0, len(test_labels), 100):
    #ax[0].plot(6., test_labels[i,0], c="k", alpha=alpha)
    plt.plot(6., predictor(test_data[i,:])[0], c="k", marker="o", alpha=0.1, ls="dashed")

plt.show()

"""




#train global signal emulator

data_dir = path+"data/globalemu/emulator14/" #'downloaded_data/'
base_dir = data_dir+"results/"#'downloaded_data/'
####Load simulation data and save in txt format####
T21 = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_T21_mat.mat")["combined_T21s"]#:23
z = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat")["z21cm"][0].astype(float)#:23
parameters = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat")["parameters"][:,:-2] #remove feed delay
parameters = np.delete(parameters, 6, 1) # delete zeta
#parameters_train, parameters_test, T21_train, T21_test = train_test_split(parameters, T21, test_size=0.4998, random_state=42)
parameters_train, parameters_test, T21_train, T21_test = train_test_split(parameters, T21, test_size=0.34, random_state=42)


#np.savetxt(data_dir+"train_data.txt", parameters_train)
#np.savetxt(data_dir+"train_labels.txt", T21_train)
#np.savetxt(data_dir+"test_data.txt", parameters_test)
#np.savetxt(data_dir+"test_labels.txt", T21_test)
#process("full", z, base_dir=base_dir, data_location=data_dir, AFB=True, std_division=True, resampling=True, logs=[0,1,2,3,7])
#nn(batch_size=128, epochs=2000, base_dir=base_dir, input_shape=9 + 1, layer_sizes=[20,20,20,20], activation="tanh", resume=False, early_stop=True)



freq, T_SARAS, weights, fg_fit, fg_fit_T_resid = np.loadtxt(path+"data/SARAS3/SARAS_3_averaged_spectrum.txt").T
redshifts = (1420/freq-1)[::-1][:] #np.linspace(6,40,100)

redshifts = [6,7,8,9,10,11,12,13,14,15,
             *redshifts,
             25,26,27,28]



dens=15/(redshifts.min()-6) #29 #36
print(redshifts.shape)
redshifts = np.unique(
    [
        *np.linspace(start=6,stop=redshifts.min(), 
                    num=np.floor((redshifts.min()-6)*dens).astype(int)),
        *redshifts,
        *np.linspace(start=redshifts.max(),stop=27, 
                    num=np.floor((27-redshifts.max())*dens).astype(int))
    ]
)
print(redshifts.shape, redshifts)


#redshifts = [
#    *np.linspace(6,14,9),
#    *np.linspace(15,25,21),
#    *np.linspace(26,27,2)
#    ]




if False:
    T21_train_interp = np.zeros(shape=(len(T21_train), len(redshifts)))
    T21_test_interp = np.zeros(shape=(len(T21_test), len(redshifts)))

    for i,T in enumerate(T21_train):
        T21_train_interp[i,:] = sip.interp1d(z, T, kind="cubic")(redshifts)
    for i,T in enumerate(T21_test):
        T21_test_interp[i,:] = sip.interp1d(z, T, kind="cubic")(redshifts)

    np.savetxt(data_dir+"train_data.txt", parameters_train)
    np.savetxt(data_dir+"train_labels.txt", T21_train_interp)
    np.savetxt(data_dir+"test_data.txt", parameters_test)
    np.savetxt(data_dir+"test_labels.txt", T21_test_interp)
    ####preprocessing####
    process("full", redshifts, base_dir=base_dir, data_location=data_dir, AFB=True, std_division=True, resampling=False, logs=[0,1,2,3,7])

####train neural network####
print("Starting training", flush=True)
#nn(batch_size=512, epochs=200, base_dir=base_dir, input_shape=9 + 1, layer_sizes=[20,20,20,20], activation="tanh",resume=True)#batch 

nn(batch_size=769, epochs=1000, base_dir=base_dir, input_shape=9 + 1, layer_sizes=[20,20,20,20], activation="tanh", resume=False, early_stop=True)



####emu test data####

from globalemu.eval import evaluate
from tensorflow import keras
import matplotlib.pyplot as plt

test_data = np.loadtxt(data_dir + 'test_data.txt')
test_labels = np.loadtxt(data_dir + 'test_labels.txt')

model = keras.models.load_model(base_dir+'model.h5', compile=False)
predictor = evaluate(base_dir=base_dir, model=model, logs=[0,1,2,3,7], gc=True)

signal_emu = np.zeros_like(test_labels)
for i,p in enumerate(test_data):
    signal_emu[i,:], zs = predictor(p) 

np.save(base_dir + "test_label_emu.npy",signal_emu)
"""
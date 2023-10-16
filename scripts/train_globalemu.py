import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import scipy.interpolate as sip
import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
path = '/home/sp2053/rds/hpc-work/powerspectra_analysis/'
data_dir = path+"data/globalemu/emulator14/" #'downloaded_data/'
base_dir = data_dir+"results/"#'downloaded_data/'
####Load simulation data and save in txt format####
T21 = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_T21_mat.mat")["combined_T21s"]#:23
z = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat")["z21cm"][0].astype(float)#:23
parameters = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat")["parameters"][:,:-2] #remove feed delay
parameters = np.delete(parameters, 6, 1) # delete zeta
#parameters_train, parameters_test, T21_train, T21_test = train_test_split(parameters, T21, test_size=0.4998, random_state=42)
parameters_train, parameters_test, T21_train, T21_test = train_test_split(parameters, T21, test_size=0.34, random_state=42)
#test_size for %128:  0.231
#test_size for %128:  0.328
#test_size for %128:  0.42500000000000004
#test_size for %128:  0.522


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



"""
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

"""


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

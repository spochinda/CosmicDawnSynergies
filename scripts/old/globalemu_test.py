path = "/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np


#import requests
#import os
data_dir = path+"data/global_emu_test1/"#'downloaded_data/'
#if not os.path.exists(data_dir):
#  os.mkdir(data_dir)
#files = ['Par_test_21cmGEM.txt', 'Par_train_21cmGEM.txt', 'T21_test_21cmGEM.txt', 'T21_train_21cmGEM.txt']
#saves = ['test_data.txt', 'train_data.txt', 'test_labels.txt', 'train_labels.txt']
#for i in range(len(files)):
#  url = 'https://zenodo.org/record/4541500/files/' + files[i]
#  with open(data_dir + saves[i], 'wb') as f:
#      f.write(requests.get(url).content)

from globalemu.preprocess import process

base_dir = data_dir+"results/"#'downloaded_data/'
#z = np.linspace(5, 50, 451)
#num = 1000

T21 = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_T21_mat.mat")["combined_T21s"]
z = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat")["z21cm"][0]
parameters = loadmat(path+"data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat")["parameters"][:,:-2]
parameters = np.delete(parameters, 6, 1) # delete zeta

parameters_train, parameters_test, T21_train, T21_test = train_test_split(parameters, T21, test_size=0.2, random_state=42)

np.savetxt(data_dir+"train_data.txt", parameters_train)
np.savetxt(data_dir+"train_labels.txt", T21_train)
np.savetxt(data_dir+"test_data.txt", parameters_test)
np.savetxt(data_dir+"test_labels.txt", T21_test)

print(parameters_train.shape[1])


process("full", z, base_dir=base_dir, data_location=data_dir, AFB=True, std_division=True, resampling=True, logs=[0,1,2,3,7])


from globalemu.network import nn

nn(batch_size=451, epochs=30, base_dir=base_dir, input_shape=parameters_train.shape[1] + 1)#layer_sizes=[8])

"""
test_data = np.loadtxt(data_dir + 'test_data.txt')
test_labels = np.loadtxt(data_dir + 'test_labels.txt')

from globalemu.eval import evaluate

input_params = test_data[0, :]
true_signal = test_labels[0, :]

predictor = evaluate(base_dir=base_dir)
signal, z = predictor(input_params)

import matplotlib.pyplot as plt

plt.plot(z, true_signal, label='True Signal')
plt.plot(z, signal, label='Emulation')
plt.legend()
plt.ylabel(r'$\delta T$ [mK]')
plt.xlabel(r'$z$')
plt.show()
"""
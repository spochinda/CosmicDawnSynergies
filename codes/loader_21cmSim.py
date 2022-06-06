import numpy as np
from copy import deepcopy
from scipy.io import loadmat

def load_files(path, name=None, model_type=None, model_generation=None, endings=None, middle=None, key=None, ArAdjust=False):
    # Function parameters:
    #   path -- path to the file
    #   name -- name of the parameter (relevant for e.g. key)
    #   model_generation and model_type for automatic stuff file names
    #   middle, endings, key for manually setting file names and kets
    # Note: Returned k arrays are _not_ converted, i.e. they should still be in 1/cMpc
    # Automatic settings:
    if key is None:
        key = deepcopy(name) if name[0]=="x" else name.upper()
        if model_generation=="old":
            key += "_Tot"
        else:
            key += "out"
    if endings is not None:
        middle = "_LyACMB_" if middle is None else middle
    elif model_type == 'Fr' and model_generation == 'old':
        middle = "_params_rad_Tot_"
        endings = ["v2", "v3", "v3_p2"]
    elif model_type == 'Ar' and model_generation == 'old':
        middle = "_params_ar_Tot_"
        endings = ['v1']
    elif model_type == 'Fr' and model_generation == 'new':
        middle = "_LyACMB_"
        endings = ['fRad', 'fRad_p2']
    elif model_type == 'Ar' and model_generation == 'new':
        middle = "_LyACMB_"
        endings = ['ARad']
    else:
        assert False

    # Load multiple files and merge, if required
    tmp = []
    for ending in endings:
        try:
            tmp.append(loadmat(path+name+middle+ending+".mat")[key])
            print(path+name+middle+ending+".mat", "-->", len(tmp[-1]))
        except KeyError:
            print(loadmat(path+name+middle+ending+".mat").keys(), key)
            raise KeyError
    params = np.concatenate(tmp)
    
    # In the synchrotron case, the definition of Ar varies between conventions.
    # Code uses 1.42 GHz, correct here to 78 Mhz
    if ArAdjust or (name=="PT" and model_type=="Ar"):
        params[:,-1] *= (0.078/1.42)**(-2.6)
        print("Adjusted Ar by", (0.078/1.42)**(-2.6))
    
    return params

def remove_powerspectra_nans(powerspec, other_arrs):
	# Remove indices which have NaNs in the power spectra
    powerspec = deepcopy(powerspec)
    nan_samples = np.unique(np.where(np.isnan(powerspec))[0])
    print("Dropping samples",nan_samples,"since they contain NaNs at relevant places. This is {:.2f}% of the data, number of samples dropped =".format(len(nan_samples)/len(powerspec)*100), len(nan_samples))
    powerspec = np.delete(powerspec, nan_samples, axis=0)
    for i in range(len(other_arrs)):
        other_arrs[i] = np.delete(other_arrs[i], nan_samples, axis=0)
        other_arrs[i] = np.real(other_arrs[i])
    # Convert to real
    assert np.allclose(np.imag(powerspec), 0)
    powerspec = np.real(powerspec)
    return powerspec, other_arrs

def PT9_to_PL8(PT9):
    # Convert 9 parameters to usually used parameters, with appropriate logs
    # PT = [Rmfp, fStar, Vc, FX, powerInd, numin, zeta, tau, Fr];
    features = np.zeros((PT9.shape[0], 8))
    features[:,0] = PT9[:,0] #Rmfp
    features[:,1] = np.log10(PT9[:,1]) #fStar
    features[:,2] = np.log10(PT9[:,2]) #Vc
    features[:,3] = np.log10(PT9[:,3]) #FX
    features[:,4] = PT9[:,4] #powerInd
    features[:,5] = PT9[:,5] #numin
    #PT9[:,6] #zeta
    features[:,6] = PT9[:,7] #tau
    features[:,7] = np.log10(PT9[:,8]) #Fr/Ar
    return features

def PT9_to_PL5(PT9):
    # Convert 9 parameters to usually used parameters, with appropriate logs
    # PT = [Rmfp, fStar, Vc, FX, powerInd, numin, zeta, tau, Fr];
    features = np.zeros((PT9.shape[0], 5))
    #features[:,0] = PT9[:,0] #Rmfp
    features[:,0] = np.log10(PT9[:,1]) #fStar
    features[:,1] = np.log10(PT9[:,2]) #Vc
    features[:,2] = np.log10(PT9[:,3]) #FX
    #features[:,4] = PT9[:,4] #powerInd
    #features[:,5] = PT9[:,5] #numin
    #PT9[:,6] #zeta
    features[:,3] = PT9[:,7] #tau
    features[:,4] = np.log10(PT9[:,8]) #Fr/Ar
    return features

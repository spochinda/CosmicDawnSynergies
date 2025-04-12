import os
import sys
import time
import torch
import glob
import numpy as np
import CosmicDawnSynergies.likelihoodnew as like
#import CosmicDawnSynergies.likelihood_hera as like_hera
from CosmicDawnSynergies.inference import prepare_prior_dict, add_SARAS3_foreground_parameters, initialize_emulators, get_prior, get_loglikelihood, dumper
from CosmicDawnSynergies.plotting import triangle_plot
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.output import PolyChordOutput

if __name__ == "__main__":
    #get file absolute path
    path = os.path.dirname(os.path.abspath(__file__)).split("/scripts")[0]
    inference_dict = {
        "inference_id": "_Arad_h1cidr3",
        "polychord_settings": {
            "nlive": 10000,
            "read_resume": False,
            "precision_criterion": 0.001,
        },
        "LikelihoodModules": {
            "LikelihoodHERA": {
                "likelihood_kwargs": {
                    "files": glob.glob(path+"/data/observations_H1C_IDR3/*.h5"), #[*glob.glob(path+"/data/observations_H6C_IDR2/all_baselines*.h5"), *glob.glob(path+"/data/observations_H1C_IDR3/*.h5")],
                    "emulator": path+"/data/trained_emulators_poweremu/dsq_Arad_emu.pth",
                    "decimate_data": False,
                    },
                    },
            #"LikelihoodRadioBackground": {
            #    "likelihood_kwargs": {
            #        "datapath": path+"/src/CosmicDawnSynergies/itamar/LWA1_with_err.npy",
            #        "emulator": path+"/data/trained_emulators_poweremu/T_today_emu.pth"
            #    },
            #    },
            #"LikelihoodXRB": {
            #    "likelihood_kwargs": {
            #        "emulator": path+"/data/trained_emulators_poweremu/xrb_emu.pth",
            #        "data_dims": ["log10E_min",],
            #    },
            #    },
            #"LikelihoodSARAS3": {
            #    "likelihood_kwargs": {
            #        "emulator": path+"/data/trained_emulators_poweremu/T21_emu.pth",
            #        "file": path+"/data/SARAS3/SARAS_3_averaged_spectrum.txt",
            #        "data_dims": ["z",],
            #        "poly_coeff": {"fg_a0": [-10., 10.], "fg_a1": [-10., 10.], "fg_a2": [-10., 10.], "fg_a3": [-10., 10.], "fg_a4": [-10., 10.], "fg_a5": [-10., 10.], "fg_a6": [-10., 10.]},
            #        #"poly_coeff": {"fg_a0": [3.54425, 3.54430], "fg_a1": [-0.2195, -0.2194], "fg_a2": [0.001, 0.00125], "fg_a3": [-0.000225, -0.002], "fg_a4": [0.0015, 0.002], "fg_a5": [-0.00035, 0.0], "fg_a6": [-0.001, 0.0005]},
            #        "noise": {"fg_std21": [0.01, 1.]},
            #        #"noise": {"fg_std21": [0.01, 0.02]},
            #        }, 
            #    },
            #"LikelihoodPowerSpectrum": {
            #    "likelihood_kwargs": {
            #        "files": [path+"/data/observations_LOFAR/lofar_limits_Acharya_hpc.npy"],
            #        "emulator": path+"/data/trained_emulators_poweremu/dsq_emu.pth",
            #        "data_dims": ["z", "log10k",]
            #        }, 
            #    },
            }
    }

    inference_dict = initialize_emulators(inference_dict)

    prior_dict = prepare_prior_dict(inference_dict, use_scaler_in_prior=False) #use scaler later in predict method of likelihood classes
    prior_dict = add_SARAS3_foreground_parameters(prior_dict, inference_dict) #add SARAS3 foreground parameters if SARAS3 likelihood is present otherwise does nothing
    prior = get_prior(inference_dict, prior_dict)
    
    LikelihoodModules = [getattr(like, key)(prior_dict, **inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]) for key in inference_dict["LikelihoodModules"].keys()]
    loglikelihood = get_loglikelihood(LikelihoodModules)

    #run polychord inference
    nDims = len(prior_dict.keys())
    nDerived = sum([likelihood.nDerived for likelihood in LikelihoodModules])
    settings = PolyChordSettings(nDims=nDims, nDerived=nDerived, **inference_dict["polychord_settings"])
    settings.base_dir = path+"/scripts/non-public/"+"_".join( list(inference_dict["LikelihoodModules"].keys()) ) + inference_dict["inference_id"]
    settings.file_root = 'run'
    
    polychordnames = list(prior_dict.keys())
    derivednames = []
    for likelihood in LikelihoodModules:
        derivednames += likelihood.output_names
    polychordnames = polychordnames + derivednames
    output = run_polychord(loglikelihood, nDims, nDerived, settings, prior=prior, dumper=dumper)
    
    output = PolyChordOutput(settings.base_dir, settings.file_root)
    polychordnames = [(name, name) for name in polychordnames]
    output.make_paramnames_files(polychordnames)

    #triangle plot
    files = [
        path + "/scripts/non-public/LikelihoodHERA_Arad_h1cidr3_h6cidr2/run",
        path + "/scripts/non-public/LikelihoodHERA_Arad_h6cidr2/run",
        settings.base_dir+"/run",
        ]
    paramNames = ["log10fstarII", "log10fstarIII", "log10Vc", "log10fX", "log10Arad"]
    basename = os.path.basename(settings.base_dir)
    image_dir = path+"/images/"
    plot_path = os.path.join(image_dir, basename) + f"_nlive_{settings.nlive}.png"
    triangle_plot(files, paramNames, plot_path=plot_path)




    

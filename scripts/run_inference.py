import os
import sys
import torch
import glob
import numpy as np
import CosmicDawnSynergies.likelihoodnew as like
#import CosmicDawnSynergies.likelihood_hera as like_hera
from CosmicDawnSynergies.inference import UniformDiscretePrior, prepare_prior_dict
from CosmicDawnSynergies.train_tools import poweremu_torch, Scaler
from CosmicDawnSynergies.plotting import triangle_plot
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior
from pypolychord.output import PolyChordOutput

if __name__ == "__main__":
    path = "/Users/simonpochinda/venvs/cosmicdawn/lib/python3.12/site-packages/CosmicDawnSynergies"
    files = glob.glob(path+"/data/observations_H1C*/*.h5")
    inference_dict = {
        "inference_id": "0",
        "polychord_settings": {
            "nlive": 500,
            "read_resume": False,
        },
        "LikelihoodModules": {
            #"LikelihoodHERA": {
            #    "likelihood_kwargs": {
            #        "files": files,
            #        "emulator": path+"/data/trained_emulators_poweremu/dsq_emu.pth",
            #        }, 
            #    },
            #"LikelihoodRadioBackground": {
            #    "likelihood_kwargs": {
            #        "datapath": path+"/src/CosmicDawnSynergies/itamar/LWA1_with_err.npy",
            #        "emulator": path+"/data/trained_emulators_poweremu/T_today_emu_tmp.pth"
            #    },
            #},
            "LikelihoodXRB": {
                "likelihood_kwargs": {
                    "emulator": path+"/data/trained_emulators_poweremu/XRB_emu_tmp.pth",
                },
                },
            }
        }
    
    for key in inference_dict["LikelihoodModules"].keys():
        emulator = poweremu_torch()
        emulator.load_network(inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"])
        inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"] = emulator

    #define prior
    prior_dict = prepare_prior_dict(inference_dict, use_scaler_in_prior=False) #decide to use scaler later

    def prior(cube):
        theta = np.zeros_like(cube)
        for i,param in enumerate(prior_dict.keys()):
            prior = prior_dict[param]
            is_discrete_param = param in emulator.data_opt["discrete_params"].keys()
            if is_discrete_param:
                a, b = 0., len(prior)
                theta[i] = UniformPrior(a=a, b=b)(cube[i])
                index = np.floor(theta[i]).astype(int)
                theta[i] = prior[index]
            else:
                a, b = prior 
                theta[i] = UniformPrior(a=a, b=b)(cube[i])    
                #theta[i] = LogUniformPrior(*prior)(cube[i])
        return theta

    
    def dumper(live, dead, logweights, logZ, logZerr):
        # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
        print("Last dead point:", dead[-1], flush=True)

    def loglikelihood(params):
        """
        This function computes the log-likelihood of the model and derived
        parameters (phi) from the physical coordinates (theta).
        """
        
        try:
            logL = 0.
            logL_nDerived = []
            for i,likelihood in enumerate(LikelihoodModules):
                logL_, logL_nDerived_ = likelihood.computeLikelihood(params)
                logL += logL_
                logL_nDerived += logL_nDerived_
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(i, exc_type, fname, exc_tb.tb_lineno)
            assert False

        return logL, logL_nDerived

    
    LikelihoodModules = [getattr(like, key)(prior_dict, **inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]) for key in inference_dict["LikelihoodModules"].keys()]
    
    nDims = len(prior_dict.keys())
    nDerived = sum([likelihood.nDerived for likelihood in LikelihoodModules])
    settings = PolyChordSettings(nDims=nDims, nDerived=nDerived)
    settings.nlive = 500
    settings.base_dir = path+"/scripts/non-public/"+"_".join( list(inference_dict["LikelihoodModules"].keys()) ) + inference_dict["inference_id"]
    settings.file_root = 'run'
    settings.read_resume = False
    
    polychordnames = list(prior_dict.keys())
    derivednames = []
    for likelihood in LikelihoodModules:
        derivednames += likelihood.output_names
    polychordnames = polychordnames + derivednames
    output = run_polychord(loglikelihood, nDims, nDerived, settings, prior=prior, dumper=dumper)
    
    output = PolyChordOutput(settings.base_dir, settings.file_root)
    polychordnames = [(name, name) for name in polychordnames]
    output.make_paramnames_files(polychordnames)

    #triangle
    files = [settings.base_dir+"/run",]
    paramNames = ["log10fstarII", "log10fstarIII", "log10Vc", "log10fX", "log10fradio"]
    basename = os.path.basename(settings.base_dir)
    image_dir = path+"/images/"
    plot_path = os.path.join(image_dir, basename) + f"_nlive_{settings.nlive}.png"
    triangle_plot(files, paramNames, plot_path=plot_path)




    

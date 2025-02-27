import torch
import glob
import numpy as np
import CosmicDawnSynergies.likelihoodnew as like
#import CosmicDawnSynergies.likelihood_hera as like_hera
from CosmicDawnSynergies.inference import UniformDiscretePrior, prepare_prior_dict
from CosmicDawnSynergies.train_tools import poweremu_torch, Scaler
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior

if __name__ == "__main__":
    path = "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/"
    files = glob.glob("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/observations_H*/*.h5")
    inference_dict = {
        #"likelihood":
        #    {"likelihood_kwargs": {
        #        "a": 1,
        #        "data_dims": ["z", "log10k"],
        #        },
        #     "emulator": path+"data/trained_emulators_poweremu/MLP_0.pth",
        #     "discrete_params": {"alpha": np.array([1., 1.3, 1.5]), "nu_0": np.array([*range(100,1500,100), 1500, 2000, 3000]), "pop": np.array([231, 232, 233])},
        #     },
        #
        "LikelihoodXRB":
            {"likelihood_kwargs": {
                "a": 1,
                "data_dims": ["log10E_min",],
                }, 
             "emulator": path+"data/trained_emulators_poweremu/XRB_emu.pth",
             "discrete_params": {"alpha": np.array([1., 1.3, 1.5]), "nu_0": np.array([*range(100,1500,100), 1500, 2000, 3000]), "pop": np.array([231, 232, 233])},
             },
        
        "LikelihoodRadioBackground":
            {"likelihood_kwargs": {
                "datapath": "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/src/CosmicDawnSynergies/itamar/LWA1_with_err.npy",
                "data_dims": ["log10nu_today",],
                },
             "emulator": path+"data/trained_emulators_poweremu/T_today_emu.pth",     
             "discrete_params": {"alpha": np.array([1., 1.3, 1.5]), "nu_0": np.array([*range(100,1500,100), 1500, 2000, 3000]), "pop": np.array([231, 232, 233])},
             },
        "LikelihoodHERA":
            {"likelihood_kwargs": {
                "files": files,
                "data_dims": ["z","log10k"],
                }, 
                "emulator": "/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/trained_emulators_poweremu/MLP_0.pth",
                "discrete_params": {"alpha": np.array([1., 1.3, 1.5]), "nu_0": np.array([*range(100,1500,100), 1500, 2000, 3000]), "pop": np.array([231, 232, 233])},
                },
                }    
    
    
    #define prior
    
    prior_dict = prepare_prior_dict(inference_dict)


    def prior(cube):
        theta = np.zeros_like(cube)
        for i,param in enumerate(prior_dict.keys()):
            prior = prior_dict[param]["prior"]
            is_discrete_param = prior_dict[param]["is_discrete"]
            if is_discrete_param:
                theta[i] = UniformDiscretePrior(prior)(cube[i])
            else:
                if param!="std21":
                    theta[i] = UniformPrior(*prior)(cube[i])    
                else:
                    theta[i] = LogUniformPrior(*prior)(cube[i])
        return theta
    
    def dumper(live, dead, logweights, logZ, logZerr):
        # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
        print("Last dead point:", dead[-1], flush=True)

    def loglikelihood(params):
        """
        This function computes the log-likelihood of the model and derived
        parameters (phi) from the physical coordinates (theta).
        """
        
        logL = 0.
        logL_nDerived = []
        for likelihood in LikelihoodModules:
            logL_, logL_nDerived_ = likelihood.computeLikelihood(params)
            logL += logL_
            logL_nDerived += logL_nDerived_

        return logL, logL_nDerived

    
    emulators = [poweremu_torch() for key in inference_dict.keys()]
    for key,emu in zip(inference_dict.keys(), emulators):
        emu.load_network(inference_dict[key]["emulator"])
    LikelihoodModules = [getattr(like, key)(emu, prior_dict, **inference_dict[key]["likelihood_kwargs"]) for key,emu in zip(inference_dict.keys(), emulators)]


    nDims = len(prior_dict.keys())
    nDerived = sum([likelihood.nDerived for likelihood in LikelihoodModules])
    settings = PolyChordSettings(nDims=nDims, nDerived=nDerived)
    settings.nlive = 1000
    settings.base_dir = path+"/scripts/non-public/"+"_".join( list(inference_dict.keys()) )
    settings.file_root = 'run'
    settings.read_resume = False
    
    polychordnames = list(prior_dict.keys())
    derivednames = []
    for likelihood in LikelihoodModules:
        derivednames += likelihood.output_names
    polychordnames = polychordnames + derivednames
    output = run_polychord(loglikelihood, nDims, nDerived, settings, prior=prior, dumper=dumper)
    output.make_paramnames_files(polychordnames)

    

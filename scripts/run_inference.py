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
            "LikelihoodXRB": {
                "likelihood_kwargs": {
                    "emulator": path+"/data/trained_emulators_poweremu/XRB_emu.pth",
                },
                },
            "LikelihoodRadioBackground": {
                "likelihood_kwargs": {
                    "datapath": path+"/src/CosmicDawnSynergies/itamar/LWA1_with_err.npy",
                    "emulator": path+"/data/trained_emulators_poweremu/T_today_emu.pth"
                },
            },
            "LikelihoodHERA": {
                "likelihood_kwargs": {
                    "files": files,
                    "emulator": path+"/data/trained_emulators_poweremu/MLP_0.pth",
                    }, 
                },
            }
        }
    
    ######################## temporary solution: manually add data_dims and discrete_params to emulator.data_opt ########################
    for key in inference_dict["LikelihoodModules"].keys():
        emulator = poweremu_torch()
        emulator.load_network(inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"])
        scaler = Scaler(emulator.scale_opt)
        emulator.data_opt["discrete_params"] = {
        "alpha": scaler.standardize(np.array([1., 1.3, 1.5]), **emulator.scale_opt["alpha"]["stats"]),
        "nu_0": scaler.standardize(np.array([*range(100,1500,100), 1500, 2000, 3000]), **emulator.scale_opt["nu_0"]["stats"]),
        "pop": scaler.standardize(np.array([231, 232, 233]), **emulator.scale_opt["pop"]["stats"])
        }
        inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"] = emulator

    inference_dict["LikelihoodModules"]["LikelihoodXRB"]["likelihood_kwargs"]["emulator"].data_opt["data_dims"] = ["log10E_min",]
    inference_dict["LikelihoodModules"]["LikelihoodRadioBackground"]["likelihood_kwargs"]["emulator"].data_opt["data_dims"] = ["log10nu_today",]
    inference_dict["LikelihoodModules"]["LikelihoodHERA"]["likelihood_kwargs"]["emulator"].data_opt["data_dims"] = ["z","log10k"]
    ######################## end of temporary solution ########################

    assert False
    
    #define prior
    prior_dict = prepare_prior_dict(inference_dict)

    def prior(cube):
        try:
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
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            assert False
    
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
            print("Exception in loglikelihood:", i, e, flush=True)


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
    paramNames = ["log10fstarII", "log10fstarIII", "log10Vc", "log10fX", "tau", "log10fradio"]
    basename = os.path.basename(settings.base_dir)
    image_dir = path+"/images/"
    plot_path = os.path.join(image_dir, basename) + f"_nlive_{settings.nlive}.png"
    triangle_plot(files, paramNames, plot_path=plot_path)




    

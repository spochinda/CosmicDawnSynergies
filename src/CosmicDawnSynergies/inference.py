import numpy as np
from CosmicDawnSynergies.train_tools import poweremu_torch, Scaler
from pypolychord.priors import UniformPrior, LogUniformPrior

class UniformDiscretePrior:
    def __init__(self, a):
        self.a = a
        self.b = len(a)
        #self.b = b

    def __call__(self, x):
        index = np.floor(self.b * x).astype(int)
        return self.a[ index ] #+ (self.b-self.a) * x
    
def initialize_emulators(inference_dict):
    for key in inference_dict["LikelihoodModules"].keys():
        emulator = poweremu_torch()
        emulator.load_network(inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"])
        inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"] = emulator
    return inference_dict
    
def prepare_prior_dict(inference_dict, use_scaler_in_prior=False):
    prior_dict = {}
    for key in inference_dict["LikelihoodModules"].keys():
        emulator = inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"]
        scaler = Scaler(emulator.scale_opt)
        for param in emulator.scale_opt.keys():
            if use_scaler_in_prior:
                method = emulator.scale_opt[param]["method"]
            else:
                method = "identity"
            scale_fn = getattr(scaler, method)
            
            data_dims_keys = [list(dim.keys())[0] for dim in emulator.data_opt["data_dims"]]
            log_data_dims_keys = [f"log10{dim}" for dim in data_dims_keys]
            if not ((param in data_dims_keys) or (param in log_data_dims_keys)):
                if param not in prior_dict.keys():
                    prior_dict[param] = {}
                
                is_discrete_param = param in emulator.data_opt["discrete_params"].keys()
                if is_discrete_param:
                    prior_dict[param] = emulator.data_opt["discrete_params"][param]
                else:
                    minimum = emulator.scale_opt[param]["stats"]["minimum"]
                    maximum = emulator.scale_opt[param]["stats"]["maximum"]
                    minimum, maximum = scale_fn(data=np.array([minimum, maximum]), **emulator.scale_opt[param]["stats"])
                    
                    if len(prior_dict[param]) == 0:
                        prior_dict[param] = np.array([minimum, maximum])
                    else:
                        prior_dict[param][0] = max(prior_dict[param][0], minimum)
                        prior_dict[param][1] = min(prior_dict[param][1], maximum)
    return prior_dict

def add_SARAS3_foreground_parameters(prior_dict, inference_dict):
    if "LikelihoodSARAS3" in inference_dict["LikelihoodModules"].keys():
        likelihood_kwargs = inference_dict["LikelihoodModules"]["LikelihoodSARAS3"]["likelihood_kwargs"]
        has_poly_coeff = "poly_coeff" in likelihood_kwargs.keys()
        has_noise = "noise" in likelihood_kwargs.keys()
        if has_poly_coeff:
            for a_i, v in likelihood_kwargs["poly_coeff"].items():
                prior_dict[a_i] = v
        else:
            assert has_poly_coeff, "poly_coeff not found in SARAS3 likelihood_kwargs"
        if has_noise:
            for noise, v in likelihood_kwargs["noise"].items():
                prior_dict[noise] = v
        else:
            assert has_noise, "noise not found in SARAS3 likelihood_kwargs"
    else:
        pass
    return prior_dict

def get_prior(inference_dict, prior_dict):
    LikelihoodModules = inference_dict["LikelihoodModules"]
    discrete_params = np.array([])
    for key in LikelihoodModules.keys():
        emulator_discrete_params = list(LikelihoodModules[key]["likelihood_kwargs"]["emulator"].data_opt["discrete_params"].keys())
        discrete_params = np.append(discrete_params, emulator_discrete_params)
        
    discrete_params = np.unique(discrete_params)

    def prior(cube):
        theta = np.zeros_like(cube)
        for i,(param,prior) in enumerate(prior_dict.items()):
            is_discrete_param = param in discrete_params
            if is_discrete_param:
                a, b = 0., len(prior)
                theta[i] = UniformPrior(a=a, b=b)(cube[i])
                index = np.floor(theta[i]).astype(int)
                theta[i] = prior[index]
            elif "fg_" in param:
                a,b = prior
                if "std" in param:
                    theta[i] = LogUniformPrior(a=a, b=b)(cube[i])
                else:
                    theta[i] = UniformPrior(a=a, b=b)(cube[i])
            else:
                a,b = prior 
                theta[i] = UniformPrior(a=a, b=b)(cube[i])    
        return theta
    return prior

def get_loglikelihood(LikelihoodModules):
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
    return loglikelihood

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)
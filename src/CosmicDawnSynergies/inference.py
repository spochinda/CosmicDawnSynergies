import numpy as np
from CosmicDawnSynergies.train_tools import poweremu_torch, Scaler

class UniformDiscretePrior:
    def __init__(self, a):
        self.a = a
        self.b = len(a)
        #self.b = b

    def __call__(self, x):
        index = np.floor(self.b * x).astype(int)
        return self.a[ index ] #+ (self.b-self.a) * x
    
def prepare_prior_dict(inference_dict):
    prior_dict = {}
    for key in inference_dict["LikelihoodModules"].keys():
        emulator = inference_dict["LikelihoodModules"][key]["likelihood_kwargs"]["emulator"]
        scaler = Scaler(emulator.scale_opt)
        for param in emulator.scale_opt.keys():
            scale_fn = getattr(scaler, emulator.scale_opt[param]["method"])
            if param not in emulator.data_opt["data_dims"]:
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

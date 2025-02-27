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
    for key in inference_dict.keys():
        emulator = poweremu_torch()
        emulator.load_network(inference_dict[key]["emulator"])
        scaler = Scaler(emulator.scale_opt)
        for param in emulator.scale_opt.keys():
            scale_fn = getattr(scaler,emulator.scale_opt[param]["method"])
            if param not in inference_dict[key]["likelihood_kwargs"]["data_dims"]:
                if param not in prior_dict.keys():
                    prior_dict[param] = {}

                
                is_discrete_param = param in inference_dict[key]["discrete_params"].keys()
                prior_dict[param]["is_discrete"] = is_discrete_param
                if is_discrete_param:
                    discrete_values = scale_fn(data=inference_dict[key]["discrete_params"][param])
                    prior_dict[param]["prior"] = discrete_values
                else:
                    minimum = emulator.scale_opt[param]["stats"]["minimum"]
                    maximum = emulator.scale_opt[param]["stats"]["maximum"]
                    minimum, maximum = scale_fn(data=np.array([minimum, maximum]), **emulator.scale_opt[param]["stats"])
                    
                    if "prior" not in prior_dict[param].keys():
                        prior_dict[param]["prior"] = [minimum, maximum]
                    else:
                        prior_dict[param]["prior"][0] = max(prior_dict[param]["prior"][0], minimum)
                        prior_dict[param]["prior"][1] = min(prior_dict[param]["prior"][1], maximum)
    return prior_dict

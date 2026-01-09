import glob
from os import path as osp
import sys
import shutil

import CosmicDawnSynergies.likelihood as like
from CosmicDawnSynergies.plotting import triangle_plot
from CosmicDawnSynergies.utils import parse_inference_options, copy_file, copy_directory, mkdir_and_rename, yaml_load

from collections import OrderedDict
import numpy as np
from pypolychord.priors import UniformPrior
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.output import PolyChordOutput


def get_prior(prior_dict):
    def prior_fn(cube):
        theta = np.zeros_like(cube)
        for i,(param,prior) in enumerate(prior_dict.items()):
            a,b = prior 
            theta[i] = UniformPrior(a=a, b=b)(cube[i])    
        return theta
    return prior_fn

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
            fname = osp.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(i, exc_type, fname, exc_tb.tb_lineno, flush=True) 
            assert False

        return logL, logL_nDerived
    return loglikelihood

def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)

def norm_minmax(params, param_stats, keys, invert=False):
    minimum = np.array([])
    maximum = np.array([])
    for key in keys:
        maximum = np.append(maximum, param_stats[key]['max'])
        minimum = np.append(minimum, param_stats[key]['min'])
    if invert:
        return params * (maximum - minimum) + minimum
    else:
        return (params - minimum) / (maximum - minimum)

def norm_standard(params, param_stats, keys, invert=False):
    mean = np.array([])
    std = np.array([])
    for key in keys:
        mean = np.append(mean, param_stats[key]['mean'])
        std = np.append(std, param_stats[key]['std'])
    if invert:
        return params * std + mean
    else:
        return (params - mean) / std

def inference_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_inference_options(root_path)
    opt['root_path'] = root_path
    
    path = osp.join(opt['root_path'], 'inferences', opt['inference_id'])
    if opt['polychord_settings']['read_resume']:
        if not osp.exists(path):
            raise ValueError(f'No existing inference found at {path} to resume from.')
    else:    
        mkdir_and_rename(path)

    copy_file(args.opt, path)

    
    LikelihoodModules = []
    for key in opt['LikelihoodModules'].keys():
        likelihood_kwargs = opt['LikelihoodModules'][key]['likelihood_kwargs']
        emulator = likelihood_kwargs['emulator']
        src = osp.abspath(osp.join(emulator, osp.pardir, osp.pardir))
        dst = path
        full_dst = copy_directory(src, dst, emulator=emulator)
        model_opt_path = glob.glob(osp.join(full_dst, '*.yml'))[0]
        model_opt = yaml_load(model_opt_path)
        model_opt['is_train'] = False
        model_opt['dist'] = False
        model_opt['num_gpu'] = 0
        model_opt['network_opt']['in_dim'] = likelihood_kwargs['in_dim']
        like_module = getattr(like, key)(likelihood_kwargs, model_opt)
        LikelihoodModules.append(like_module)


    prior_dict = OrderedDict()
    for module in LikelihoodModules:
        n_data_dims = len(module.model_opt['dataset']['data_dims'])
        param_keys = list(module.model.param_stats.keys())
        for param in param_keys[n_data_dims:]:
            if param in prior_dict.keys():
                continue
            else:
                prior_dict[param] = [module.model.param_stats[param]["min"], module.model.param_stats[param]["max"]]
        if module.__class__.__name__ == 'LikelihoodSARAS3':
            for fg_param in likelihood_kwargs['poly_coeff'].keys():
                prior_dict[fg_param] = likelihood_kwargs['poly_coeff'][fg_param]
            prior_dict['lognoise'] = likelihood_kwargs['lognoise']
        if module.__class__.__name__ == 'LikelihoodSDC3b':
            prior_dict['lognoise'] = likelihood_kwargs['lognoise']
    
    for module in LikelihoodModules:
        if hasattr(module, 'get_prior_indices'):
            module.get_prior_indices(prior_dict)

    loglikelihood = get_loglikelihood(LikelihoodModules)
    LikelihoodModules_str = "_".join( list(opt["LikelihoodModules"].keys()) )

    #run polychord inference
    nDims = len(prior_dict.keys())
    nDerived = sum([likelihood.nDerived for likelihood in LikelihoodModules])
    settings = PolyChordSettings(nDims=nDims, nDerived=nDerived, **opt["polychord_settings"])
    settings.base_dir = osp.join(path, LikelihoodModules_str)
    settings.file_root = 'run'

    paramNames = list(prior_dict.keys())
    derivednames = []
    for likelihood in LikelihoodModules:
        derivednames += likelihood.output_names
    polychordnames = paramNames + derivednames
    prior_fn = get_prior(prior_dict)
    output = run_polychord(loglikelihood, nDims, nDerived, settings, prior=prior_fn, dumper=dumper)

    output = PolyChordOutput(settings.base_dir, settings.file_root)
    polychordnames = [(name, name) for name in polychordnames]
    output.make_paramnames_files(polychordnames)

    # Create denormalized dead-birth and phys_live-birth files for denormalized chains
    original_file_root = osp.join(settings.base_dir,settings.file_root)
    param_stats = LikelihoodModules[0].model.param_stats
    emuNames = list(param_stats.keys())
    astroNames = []
    paramNames_indices = []
    for i,key in enumerate(paramNames):
        if key in emuNames:
            astroNames.append(key)
            paramNames_indices.append(i)
    files = [original_file_root]
    plot_path = osp.join(path, f"triangle_orig_nlive_{settings.nlive}.png")
    triangle_plot(files, astroNames, plot_path=plot_path)    



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    inference_pipeline(root_path)
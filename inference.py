import jax

# inference runs in double precision (evidence sums, k-bin values); training is fp32
jax.config.update('jax_enable_x64', True)

import shutil
from collections import OrderedDict
from os import path as osp

import numpy as np

from src.CosmicDawnSynergies.likelihoods import build_likelihood
from src.CosmicDawnSynergies.plotting import triangle_plot
from src.CosmicDawnSynergies.sampling import run_nested_sampling, save_chains, to_nested_samples
from src.CosmicDawnSynergies.utils import (copy_file, get_root_logger, get_time_str, mkdir_and_rename,
                                           parse_inference_options)


def copy_emulator(src, dst):
    """Copy an emulator dir into the inference run (yml, param_stats, best ckpt)."""
    src = osp.normpath(src)
    full_dst = osp.join(dst, osp.basename(src))
    if osp.exists(full_dst):
        print(f'Destination directory {full_dst} already exists. Skipping copy.')
    else:
        shutil.copytree(src, full_dst,
                        ignore=shutil.ignore_patterns('steps', 'tb_logger', '*.log', 'visualization'))
    return full_dst


def inference_pipeline(root_path):
    opt, args = parse_inference_options(root_path)
    opt['root_path'] = root_path

    path = osp.join(root_path, 'inferences', opt['inference_id'])
    mkdir_and_rename(path)
    copy_file(args.opt, path)

    logger = get_root_logger(log_file=osp.join(path, f"inference_{opt['inference_id']}_{get_time_str()}.log"))
    logger.info(f'JAX {jax.__version__}, x64={jax.config.jax_enable_x64}, devices: {jax.devices()}')

    # build likelihood modules from emulators copied into the run directory
    modules = []
    for name, module_opt in opt['LikelihoodModules'].items():
        kwargs = dict(module_opt['likelihood_kwargs'])
        for key in ('emulator', 'emulator_xHI'):
            if kwargs.get(key):
                kwargs[key] = copy_emulator(kwargs[key], path)
        modules.append(build_likelihood(name, kwargs))

    # merged prior (insertion-ordered, first module wins on duplicates)
    prior_bounds = OrderedDict()
    for module in modules:
        for key, bounds in module.prior_bounds.items():
            prior_bounds.setdefault(key, bounds)
    param_names = list(prior_bounds.keys())
    logger.info('Priors:\n' + '\n'.join(f'  {k}: {v}' for k, v in prior_bounds.items()))

    def loglikelihood(particle):
        logL = 0.0
        for module in modules:
            logL += module.loglikelihood(particle)
        return logL

    sampler_opt = dict(opt.get('sampler', {}))
    seed = sampler_opt.pop('seed', 0)
    dead, state = run_nested_sampling(jax.random.key(seed), loglikelihood, prior_bounds, **sampler_opt)

    # derived parameters, vmapped over all samples
    derived = {}
    positions = dead.particles.position
    for module in modules:
        if module.derived({k: v[0] for k, v in positions.items()}) is not None:
            out = jax.vmap(module.derived)(positions)
            derived.update({k: np.asarray(v) for k, v in out.items()})

    samples = to_nested_samples(dead, param_names, derived=derived)
    csv_path = save_chains(samples, osp.join(path, 'run.csv'))
    logger.info(f'Chains written to {csv_path}')
    logger.info(f'logZ = {float(samples.logZ()):.3f} (anesthetic) / '
                f'{float(state.integrator.logZ):.3f} (integrator)')

    plot_noise = opt.get('plot_noise', True)
    astro_names = param_names if plot_noise else [n for n in param_names if n != 'lognoise']
    plot_path = osp.join(path, f"triangle_{opt['inference_id']}.png")
    triangle_plot([csv_path], astro_names, plot_path=plot_path)
    logger.info(f'Triangle plot written to {plot_path}')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    inference_pipeline(root_path)

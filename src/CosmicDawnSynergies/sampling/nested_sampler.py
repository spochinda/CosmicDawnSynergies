import time

import blackjax
import jax
from blackjax.ns.utils import finalise, uniform_prior

from CosmicDawnSynergies.utils import get_root_logger


def run_nested_sampling(rng_key, loglikelihood_fn, prior_bounds, num_live=500, num_delete=None,
                        num_inner_steps=None, stop_dlogZ=3.0, max_steps=100000, log_every=50):
    """Run blackjax nested slice sampling to convergence.

    Args:
        rng_key: JAX PRNG key.
        loglikelihood_fn: pure function of one particle dict -> scalar logL.
        prior_bounds: OrderedDict name -> (min, max) uniform prior bounds.
        num_live: number of live points.
        num_delete: particles deleted/replaced per step (default num_live // 2).
        num_inner_steps: slice-sampling steps per new particle (default 5 * ndims).
        stop_dlogZ: stop when logZ_live - logZ < -stop_dlogZ.

    Returns:
        (dead, state): finalised NSInfo (dead + final live points) and final state.
    """
    logger = get_root_logger()
    ndims = len(prior_bounds)
    num_delete = num_delete or max(1, num_live // 2)
    num_inner_steps = num_inner_steps or 5 * ndims

    rng_key, init_key = jax.random.split(rng_key)
    particles, logprior_fn = uniform_prior(init_key, num_live, dict(prior_bounds))

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )
    state = algo.init(particles)
    step = jax.jit(algo.step)

    logger.info(f'Nested sampling: ndims={ndims}, num_live={num_live}, num_delete={num_delete}, '
                f'num_inner_steps={num_inner_steps}, stop at dlogZ<{-stop_dlogZ}')

    dead = []
    start = time.time()
    for i in range(1, max_steps + 1):
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step(step_key, state)
        dead.append(info)
        dlogZ = float(state.integrator.logZ_live - state.integrator.logZ)
        if i % log_every == 0:
            logger.info(f'step {i:6d} | logZ={float(state.integrator.logZ):.3f} '
                        f'| logZ_live-logZ={dlogZ:.3f} | n_dead={i * num_delete}')
        if dlogZ < -stop_dlogZ:
            break
    else:
        logger.warning(f'Nested sampling hit max_steps={max_steps} before convergence')

    logger.info(f'Nested sampling finished: {i} steps, {i * num_delete + num_live} samples, '
                f'logZ={float(state.integrator.logZ):.3f}, took {time.time() - start:.1f}s')
    return finalise(state, dead), state

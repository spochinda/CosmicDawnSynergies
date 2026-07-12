"""Input-normalization functions and helpers shared by Models.

Normalization is part of the emulator contract: the Model applies it inside
make_predict_fn using its own param_stats; likelihood code never does.
"""
import jax.numpy as jnp


def norm_minmax(params, stats, invert=False):
    minimum, maximum = stats['min'], stats['max']
    if invert:
        return params * (maximum - minimum) + minimum
    return (params - minimum) / (maximum - minimum)


def norm_standard(params, stats, invert=False):
    mean, std = stats['mean'], stats['std']
    if invert:
        return params * std + mean
    return (params - mean) / std


def norm_minmax_extended(params, stats, invert=False):
    minimum, maximum = stats['min'], stats['max']
    if invert:
        return (params + 1) / 2 * (maximum - minimum) + minimum
    return (params - minimum) / (maximum - minimum) * 2 - 1


NORMALIZATIONS = {
    'norm_minmax': norm_minmax,
    'norm_standard': norm_standard,
    'norm_minmax_extended': norm_minmax_extended,
}


def stats_arrays(param_stats, keys=None):
    """param_stats (per-name dicts) -> column vectors of min/max/mean/std."""
    keys = list(param_stats.keys()) if keys is None else list(keys)
    return {
        'min': jnp.array([param_stats[k]['min'] for k in keys]),
        'max': jnp.array([param_stats[k]['max'] for k in keys]),
        'mean': jnp.array([param_stats[k]['mean'] for k in keys]),
        'std': jnp.array([param_stats[k]['std'] for k in keys]),
    }

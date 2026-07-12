import numpy as np
from anesthetic import NestedSamples


def to_nested_samples(dead, param_names, derived=None):
    """blackjax NSInfo (finalised) -> anesthetic NestedSamples.

    Initial live points are born at loglikelihood_birth = nan in blackjax;
    anesthetic expects -inf (born from the whole prior).

    Args:
        dead: finalised NSInfo from run_nested_sampling.
        param_names: sampled parameter names, defines column order.
        derived: optional dict name -> (n_samples,) array of derived columns.
    """
    sw = dead.particles  # StateWithLogLikelihood
    columns = list(param_names)
    data = [np.asarray(sw.position[name]) for name in columns]
    for name, values in (derived or {}).items():
        columns.append(name)
        data.append(np.asarray(values))

    logL = np.asarray(sw.loglikelihood)
    logL_birth = np.asarray(sw.loglikelihood_birth)
    logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)

    # labels match the legacy PolyChord paramnames convention (label = name),
    # so PolyChord and BlackJAX chains overlay in one triangle_plot call
    samples = NestedSamples(data=np.stack(data, axis=1), columns=columns,
                            logL=logL, logL_birth=logL_birth)
    return samples.set_labels([f'${c}$' for c in columns] + [None] * (samples.shape[1] - len(columns)))


def save_chains(samples, csv_path):
    """Persist a NestedSamples as anesthetic CSV (read back via read_chains)."""
    samples.to_csv(csv_path)
    return csv_path

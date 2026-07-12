# Legacy (PyTorch / PolyChord) implementation

Frozen reference copy of the pre-JAX stack: `train.py`, `inference.py`, `tune.py`,
`model.py`, `likelihood.py`, `dataset.py`, `utils.py`.

Not maintained and not importable as part of the JAX package (imports here still
reference `basicsr`, `torch`, `pypolychord` and the old module paths). To run it,
use the `main` branch. Kept on this branch only so behaviour can be diffed against
the JAX rewrite during the migration.

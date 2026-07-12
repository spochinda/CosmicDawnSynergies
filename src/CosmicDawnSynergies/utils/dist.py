import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def get_mesh():
    """Return a 1D 'data' mesh over all local devices, or None on one device."""
    devices = jax.devices()
    if len(devices) > 1:
        return Mesh(devices, ('data',))
    return None


def batch_sharding(mesh):
    """Batch-axis sharding for data-parallel training (None on single device)."""
    if mesh is None:
        return None
    return NamedSharding(mesh, P('data'))


def replicate_sharding(mesh):
    """Fully-replicated sharding (for model parameters)."""
    if mesh is None:
        return None
    return NamedSharding(mesh, P())

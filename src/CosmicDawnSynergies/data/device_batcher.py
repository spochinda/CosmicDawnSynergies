import math

import jax
import jax.numpy as jnp
import numpy as np


class DeviceBatcher:
    """In-memory replacement for the DataLoader: arrays live on device once,
    each epoch is a fresh ``jax.random.permutation`` sliced into fixed batches.

    On multi-device hosts pass a ``sharding`` (batch-axis NamedSharding) so the
    arrays are laid out data-parallel from the start; jit then keeps batches
    sharded without per-step host transfers.
    """

    def __init__(self, params, targets, batch_size, shuffle=True, drop_last=True,
                 dtype=jnp.float32, sharding=None):
        params = jnp.asarray(np.asarray(params), dtype=dtype)
        targets = jnp.asarray(np.asarray(targets), dtype=dtype)
        if sharding is not None:
            params = jax.device_put(params, sharding)
            targets = jax.device_put(targets, sharding)
        self.params = params
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(params)
        if drop_last:
            self.num_batches = self.n // batch_size
        else:
            self.num_batches = math.ceil(self.n / batch_size)

    def __len__(self):
        return self.num_batches

    def epoch(self, key=None):
        """Yield (params, targets) batches; pass a fresh PRNG key per epoch."""
        if self.shuffle:
            if key is None:
                raise ValueError('shuffle=True requires a PRNG key per epoch')
            perm = jax.random.permutation(key, self.n)
        else:
            perm = jnp.arange(self.n)
        for i in range(self.num_batches):
            idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.params[idx], self.targets[idx]

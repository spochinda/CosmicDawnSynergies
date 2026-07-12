import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from CosmicDawnSynergies.utils.registry import ARCH_REGISTRY


def _torch_default_kernel_init(key, shape, dtype=jnp.float32):
    # torch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)); kernel shape is (in, out)
    bound = 1.0 / np.sqrt(shape[0])
    return jax.random.uniform(key, shape, dtype, -bound, bound)


def _make_torch_default_bias_init(fan_in):
    def init(key, shape, dtype=jnp.float32):
        bound = 1.0 / np.sqrt(fan_in)
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def _get_initializers(init, activation, fan_in):
    """Return (kernel_init, bias_init) for a Linear layer with given fan_in."""
    if init == 'torch_default':
        return _torch_default_kernel_init, _make_torch_default_bias_init(fan_in)
    if init == 'kaiming_normal':
        return nnx.initializers.he_normal(), nnx.initializers.constant(0.01)
    if init == 'xavier_normal':
        return nnx.initializers.glorot_normal(), nnx.initializers.constant(0.01)
    raise ValueError(f"Unknown init '{init}'. Use torch_default | kaiming_normal | xavier_normal.")


@ARCH_REGISTRY.register()
class MLP(nnx.Module):
    """
    Multi-Layer Perceptron (MLP) for emulator training.

    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        n_hidden (int): Number of hidden layers
        out_dim (int): Output dimension
        activation (str): jax.nn activation name ('relu', 'gelu', 'tanh', ...)
        out_activation (str): Output activation ('softplus' for non-negative
            outputs such as power spectra; avoids dying-ReLU while
            guaranteeing positivity)
        init (str): 'torch_default' (matches torch nn.Linear default init),
            'kaiming_normal', or 'xavier_normal'
    """

    def __init__(self, in_dim: int, hidden_dim: int = 100, n_hidden: int = 6, out_dim: int = 1,
                 activation: str = 'relu', out_activation: str = None,
                 init: str = 'torch_default', *, rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.activation = getattr(jax.nn, activation)
        self.out_activation = getattr(jax.nn, out_activation) if out_activation else None

        dims = [in_dim] + [hidden_dim] * (n_hidden + 1) + [out_dim]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            kernel_init, bias_init = _get_initializers(init, activation, d_in)
            layers.append(nnx.Linear(d_in, d_out, kernel_init=kernel_init,
                                     bias_init=bias_init, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x.squeeze(-1) if self.out_dim == 1 else x

    def get_num_parameters(self) -> int:
        return sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(self, nnx.Param)))

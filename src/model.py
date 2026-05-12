import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr

from typing import Any
from src.constants import LayerSize


class XLR8(eqx.Module):
    mlp: eqx.nn.Sequential 

    def __init__(self, key: jax.Array) -> None:
        keys = jr.split(key, 4)

        # Wrap tanh in a lambda that accepts and ignores **kwargs (like the 'key', cuz I got that error)
        safe_tanh = lambda x, **kwargs: jnn.tanh(x)

        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(in_features=3, out_features=LayerSize.L1, key=keys[0]),
            safe_tanh,
            eqx.nn.Linear(LayerSize.L1, LayerSize.L2, key=keys[1]),
            safe_tanh,
            eqx.nn.Linear(LayerSize.L2, LayerSize.L3, key=keys[2]),
            safe_tanh,
            eqx.nn.Linear(LayerSize.L3, 3, key=keys[3])
        ])


    def __call__(self, t: jax.Array, y: jax.Array, args: Any) -> jax.Array:
        # Even though the Lorenz system is autonomous (it doesn't depend on t),
        # Diffrax will pass t, y, and args to this function, so I needed to include them in the signature.
        return self.mlp(y)

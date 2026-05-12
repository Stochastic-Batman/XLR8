import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr
from typing import Any, Final


class LayerSize:
    L1: Final = 16
    L2: Final = 64
    L3: Final = 16


class XLR8(eqx.Module):
    mlp: eqx.nn.Sequential 

    def __init__(self, key: jax.Array) -> None:
        keys = jr.split(key, 4)

        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(in_features=3, out_features=LayerSize.L1, key=keys[0]),
            jnn.tanh,
            eqx.nn.Linear(LayerSize.L1, LayerSize.L2, key=keys[1]),
            jnn.tanh,
            eqx.nn.Linear(LayerSize.L2, LayerSize.L3, key=keys[2]),
            jnn.tanh,
            eqx.nn.Linear(LayerSize.L3, 3, key=keys[3])
        ])


    def __call__(self, t: jax.Array, y: jax.Array, args: Any) -> jax.Array:
        # Even though the Lorenz system is autonomous (it doesn't depend on t),
        # Diffrax will pass t, y, and args to this function, so I needed to include them in the signature.
        return self.mlp(y)

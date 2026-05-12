import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr

from typing import Any


class XLR8(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key: jax.Array) -> None:
        self.mlp = eqx.nn.MLP(in_size=3, out_size=3, width_size=128, depth=3, key=key)


    def __call__(self, t: jax.Array, y: jax.Array, args: Any) -> jax.Array:
        # Even though the Lorenz system is autonomous (it doesn't depend on t),
        # Diffrax will pass t, y, and args to this function, so I needed to include them in the signature.
        return self.mlp(y)

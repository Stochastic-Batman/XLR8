import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from src.constants import SimulationConstants
from src.model import XLR8



key = jr.key(SimulationConstants.SEED)
model_key, train_key = jr.split(key)

data = jnp.load("data/lorenz_trajectories.npy")
times = jnp.linspace(0, SimulationConstants.T_MAX, SimulationConstants.NUM_STEPS)

xlr8 = XLR8(key=model_key)
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(xlr8, eqx.is_array))


def solve_trajectory(model: XLR8, times: jax.Array, y0: jax.Array) -> jax.Array:
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=times)
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    return diffrax.diffeqsolve(
        terms=term, 
        solver=solver, 
        t0=times[0], 
        t1=times[-1], 
        dt0=times[1] - times[0],
        y0=y0, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller
    ).ys


@eqx.filter_jit
def make_step(model: XLR8, opt_state: optax.OptState, times: jax.Array, y_true: jax.Array) -> tuple[jax.Array, XLR8, optax.OptState]:
    def loss_fn(model: XLR8) -> jax.Array:
        preds = solve_trajectory(model, times, y_true[0])
        return jnp.mean((preds - y_true) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


if __name__ == "__main__":
    for epoch in range(SimulationConstants.NUM_EPOCHS):
        batch_index = epoch % len(data)
        y_true = data[batch_index]

        loss, xlr8, opt_state = make_step(xlr8, opt_state, times, y_true)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}) Loss: {loss:.5f}")

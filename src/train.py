import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import os

from src.constants import SimulationConstants, TrainingConstants
from src.model import XLR8
from src.visualize import plot_loss_curve, plot_rollouts



key = jr.key(SimulationConstants.SEED)
model_key, train_key = jr.split(key)

data = jnp.load(TrainingConstants.DATA_PATH)  # Shape: (num_particles, num_steps, 3)
all_times = jnp.linspace(0, SimulationConstants.T_MAX, SimulationConstants.NUM_STEPS)

# Normalize once, globally, so curriculum changes do not change scaling.
data_mean, data_std = jnp.mean(data, axis=(0, 1), keepdims=True), jnp.std(data, axis=(0, 1), keepdims=True)
data_norm = (data - data_mean) / data_std

xlr8 = XLR8(key=model_key)
optimizer = optax.adam(learning_rate=TrainingConstants.LEARNING_RATE)
opt_state = optimizer.init(eqx.filter(xlr8, eqx.is_array))


def get_shard_steps(epoch: int) -> int:
    steps = TrainingConstants.CURRICULUM[0][1]
    for start_epoch, horizon in TrainingConstants.CURRICULUM:
        if epoch >= start_epoch:
            steps = horizon
    return steps


def solve_trajectory(model: XLR8, times: jax.Array, y0: jax.Array) -> jax.Array:
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=times)
    stepsize_controller = diffrax.PIDController(rtol=TrainingConstants.RTOL, atol=TrainingConstants.ATOL)

    return diffrax.diffeqsolve(
        terms=term, 
        solver=solver, 
        t0=times[0], 
        t1=times[-1], 
        dt0=times[1] - times[0],
        y0=y0, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        max_steps=TrainingConstants.MAX_STEPS,
        throw=False  # Returns partial solution instead of crashing if max_steps is hit
    ).ys


# Batch over multiple trajectories using vmap
def batch_loss(model: XLR8, times: jax.Array, y_true_batch: jax.Array) -> jax.Array:
    solve_vmap = jax.vmap(lambda y0: solve_trajectory(model, times, y0))  # vmap over the first axis of y_true_batch (the batch size)
    preds = solve_vmap(y_true_batch[:, 0])
    return jnp.mean((preds - y_true_batch) ** 2)


@eqx.filter_jit
def make_step(model: XLR8, opt_state: optax.OptState, times: jax.Array, y_true: jax.Array) -> tuple[jax.Array, XLR8, optax.OptState]:
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, times, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("XLR8 training started!")


    loss_history = []

    for epoch in range(TrainingConstants.NUM_EPOCHS):
        shard_steps = get_shard_steps(epoch)
        times = all_times[:shard_steps]
        data_sharded = data_norm[:, :shard_steps, :]

        sample_key = jr.fold_in(train_key, epoch)
        batch_indices = jr.randint(sample_key, (TrainingConstants.BATCH_SIZE,), 0, len(data_sharded))
        y_true_batch = data_sharded[batch_indices]

        loss, xlr8, opt_state = make_step(xlr8, opt_state, times, y_true_batch)

        loss_history.append(float(loss))

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | horizon={shard_steps} | Loss: {loss:.5f}")


    eqx.tree_serialise_leaves(TrainingConstants.MODEL_PATH, xlr8)
    jnp.savez(TrainingConstants.NORM_PATH, mean=jnp.asarray(data_mean), std=jnp.asarray(data_std))
    print(f"Training Complete! The model is saved at {TrainingConstants.MODEL_PATH}")

    plot_loss_curve(loss_history, TrainingConstants.LOSS_PLOT_PATH)
    plot_rollouts(xlr8, data, all_times, data_mean, data_std, TrainingConstants.ROLLOUT_PLOT_PATH, num_to_plot=5)
    print(f"You can view the actual vs predicted plot at {TrainingConstants.ROLLOUT_PLOT_PATH}")

import os
import jax
import jax.numpy as jnp
from typing import Final
import matplotlib.pyplot as plt


class LorenzConstants:
    SIGMA: Final = 10.0
    RHO: Final = 28.0
    BETA: Final = 8/3


class SimulationConstants:
    NUM_PARTICLES: Final = 10000
    NUM_STEPS: Final = 1000
    T_MAX: Final = 20.0
    SEED: Final = 95


def lorenz_deriv(state: jax.Array, t: float) -> jax.Array:
    x, y, z = state
    dx = LorenzConstants.SIGMA * (y - x)
    dy = x * (LorenzConstants.RHO - z) - y
    dz = x * y - LorenzConstants.BETA * z
    return jnp.array([dx, dy, dz])


def rk4_step(state: jax.Array, t: float, dt: float) -> jax.Array:
    k1 = lorenz_deriv(state, t)
    k2 = lorenz_deriv(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = lorenz_deriv(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = lorenz_deriv(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_trajectory(initial_state: jax.Array, times: jax.Array) -> jax.Array:
    dt = times[1] - times[0]

    def aux(carry: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_state = rk4_step(carry, t, dt)
        return next_state, next_state

    _, trajectory = jax.lax.scan(aux, initial_state, times)
    return trajectory


def visualize_trajectories(dataset: jax.Array, output_path: str, num_to_plot: int = 5) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    colors = ["#00d2ff", "#00f2fe", "#4facfe", "#00f2fe", "#2af598"]
    for i in range(min(num_to_plot, len(dataset))):
        traj = dataset[i]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=colors[i % len(colors)],
            alpha=0.7,
            linewidth=1.0,
        )

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(colors="white")
    ax.set_title("Lorenz Attractor Trajectories", color="white", fontsize=14)
    
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    key = jax.random.key(SimulationConstants.SEED)
    initial_states = jax.random.normal(key, (SimulationConstants.NUM_PARTICLES, 3)) * 10
    times = jnp.linspace(0, SimulationConstants.T_MAX, SimulationConstants.NUM_STEPS)

    vmapped = jax.jit(jax.vmap(simulate_trajectory, in_axes=(0, None)))

    print(f"XLR8: Generating {SimulationConstants.NUM_PARTICLES} trajectories...")
    dataset = vmapped(initial_states, times)
    print(f"Done! Dataset shape: {dataset.shape}")

    os.makedirs("data", exist_ok=True)
    jnp.save("data/lorenz_trajectories.npy", dataset)
    print("Saved trajectories to data/lorenz_trajectories.npy")

    visualize_trajectories(dataset, "data/lorenz_visualization.png")

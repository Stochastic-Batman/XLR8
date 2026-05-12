from typing import Final


class LorenzConstants:
    SIGMA: Final = 10.0
    RHO: Final = 28.0
    BETA: Final = 8/3


class SimulationConstants:
    NUM_PARTICLES: Final = 10000
    NUM_STEPS: Final = 1000
    T_MAX: Final = 20.0
    SEED: Final = 95


class TrainingConstants:
    DATA_PATH: Final = "data/lorenz_trajectories.npy"
    SHARD_STEPS: Final = 50
    LEARNING_RATE: Final = 3e-4
    RTOL: Final = 1e-2
    ATOL: Final = 1e-4
    MAX_STEPS: Final = 2000
    BATCH_SIZE: Final = 32
    NUM_EPOCHS: Final = 3001

    MODEL_PATH: Final = "checkpoints/xlr8.eqx"
    NORM_PATH: Final = "checkpoints/norm_stats.npz"
    LOSS_PLOT_PATH: Final = "data/loss_curve.png"
    ROLLOUT_PLOT_PATH: Final = "data/rollout.png"

    # (start_epoch, horizon_steps)
    CURRICULUM: Final = (
        (0, 10),
        (2000, 20),
        (5000, 50),
        (8000, 100),
    )

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
    NUM_EPOCHS: Final = 10000

class LayerSize:
    L1: Final = 16
    L2: Final = 64
    L3: Final = 16

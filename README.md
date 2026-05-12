# XLR8
Learning the Chaotic Lorenz Attractor via Neural ODEs

The entire goal of this project is to take a neural network that knows absolutely nothing about physics, the model only observes trajectories, not the governing equations, and let it learn a continuous neural approximation of the Lorenz vector field from raw trajectories. Once trained, the neural network behaves as a learned dynamical system that approximates the Lorenz attractor and can generate continuous-time chaotic trajectories without explicitly encoding the governing equations.

The name XLR8 is a double tribute to speed:
1. In Ben 10, XLR8 is a sleek alien that accelerates to insane speeds instantly, controls friction, and sees the world in slow motion.
2. In our code, `JAX` allows us to JIT-compile the Neural ODE into optimized XLA machine code while vectorizing trajectory generation and training across thousands of simulations in parallel. 


## Stack
This is my first JAX project, so the stack might be clear for experienced users, but not for me:

* **Core Engine:** `JAX`
* **Neural Network Layering:** `Equinox`
* **ODE Solver:** `Diffrax`
* **Optimization:** `Optax`

## Local Setup Instructions
This project is configured to run fully on CPU environments, no GPU/TPU is required. Ensure you have Python 3.14 installed on your system.

```bash
python3 --version  # Should output: Python 3.14.x
```

We will create and isolate our project using Python's built-in virtual environment called `xlr8_env`:
```bash
python3 -m venv xlr8_env

source xlr8_env/bin/activate   # Linux/macOS
.\xlr8_env\Scripts\activate    # Windows
```

`JAX` (for CPU) and its dependencies:
```bash
pip install --upgrade "jax[cpu]"
pip install equinox diffrax optax matplotlib
```

you can optionally verify the installation:
```bash
python3 -c "import jax; print('JAX devices:', jax.devices())"
```

## Generating the Data & Training the `XLR8` Model

To generate the data, run:
```bash
python -m src.generate_data
```

Train the network with:
```bash
python -m src.train
```

To change the hyperparameters, modify `src/constants.py`.

Training automatically saves:
- serialized model checkpoints
- normalization statistics
- rollout visualizations
- training loss curves


## Project Structure
```bash
XLR8/
├── checkpoints/           # (Gitignored) Saved models and norm stats 
├── data/                  # (Gitignored) Generated synthetic trajectories (Lorenz System) + some visualization
├── LICENSE
├── README.md
├── src/
│   ├── constants.py       # Project-wide constants
│   ├── generate_data.py   # Runge-Kutta data generation
│   ├── model.py           # Equinox Neural ODE model configuration
│   ├── train.py           # Training loop utilizing Diffrax and Optax
│   └── visualize.py       # I was too lazy to code this, so I generated it with Gemini 3.1 Pro
└── xlr8_env/
```

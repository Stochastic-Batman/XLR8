# XLR8
Learning the Chaotic Lorenz Attractor via Neural ODEs

The entire goal of this project is to take a neural network that knows absolutely nothing about physics, show it a bunch of coordinates over time, and let it learn the underlying math of the Lorenz system from scratch. Once trained, the neural network acts as a functional clone of the chaotic Lorenz equations, allowing us to predict complex, continuous-time trajectories without hardcoded physical laws.

The name XLR8 is a double tribute to speed:
1. In Ben 10, XLR8 is a sleek alien that accelerates to insane speeds instantly, controls friction, and sees the world in slow motion.
2. In our code, traditional physics solvers get bogged down in slow Python loops. By using JAX, we compile our Neural ODE directly to machine code using XLA and use vectorized mapping to run thousands of chaotic simulations in parallel. It is pure speed directly on your CPU.


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

We will create and isolate our project using a native virtual environment called `xlr8_env`:
```bash
python3 -m venv xlr8_env

source xlr8_env/bin/activate   # Linux/macOS
.\xlr8_env\Scripts\activate    # Windows
```

JAX (for CPU) and its dependencies:
```bash
pip install --upgrade "jax[cpu]"
pip install equinox diffrax optax matplotlib
```

you can optionally verify the installation:
```bash
python3 -c "import jax; print('JAX devices:', jax.devices())"
```

## Project Structure
```bash
XLR8/
├── data/                  # Generated synthetic trajectories (Lorenz System) + some visualization
├── LICENSE
├── README.md
├── src/
│   ├── benchmark.py       # Speed comparison scripts
│   ├── constants.py       # Project-wide constants
│   ├── generate_data.py   # Runge-Kutta data generation
│   ├── model.py           # Equinox Neural ODE model configuration
│   └── train.py           # Training loop utilizing Diffrax and Optax
└── xlr8_env/
```

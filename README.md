# XLR8
Neural ODEs with JAX for chaotic dynamics

Traditional numerical integration of complex, chaotic physical systems can be a massive computational bottleneck. XLR8 leverages Neural ODEs to learn the continuous-time vector field directly from raw observational trajectories. By modeling the system's dynamics using a neural network and solving the continuous trajectory forward in time via modern adaptive ODE solvers, XLR8 bypasses slow, step-by-step classical loops.

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
├── data/                  # Generated synthetic trajectories (Lorenz System)
├── README.md
├── src/
│   ├── generate_data.py   # Runge-Kutta data generation
│   ├── model.py           # Equinox Neural ODE model configuration
│   ├── train.py           # Training loop utilizing Diffrax and Optax
│   └── benchmark.py       # Speed comparison scripts
└── xlr8_env/
```

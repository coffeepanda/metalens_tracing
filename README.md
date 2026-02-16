# Metalens Simulation

This repository provides an example of simulating metalens behavior using two different representations: **SIREN** and **bicubic interpolation**. The example demonstrates the training process and highlights key considerations for each representation.

## Getting Started

To run the example, execute the following command:

```bash
python example.py
```

The training loss history will be saved as `loss_history.png` in the current directory.

## Key Notes

1. **Representations**:
    - **SIREN**: A neural network-based representation.
    - **Bicubic interpolation**: A grid-based representation. Note that second-order gradients are not supported in PyTorch for this method, so finite differences are used to compute phase gradients.

2. **Surrogate Model**:
    - Currently, a **fake network** is used as a placeholder for the surrogate model.
    - In actual training, this should be replaced with a **real surrogate model** with **fixed weights**.

3. **Gradient Behavior**:
    - The gradient behavior can vary significantly between the SIREN and bicubic representations.
    - Experiment with different hyperparameters to optimize performance.

4. **Metric Units**:
    - The metric unit used across the scripts is **meters**. This may not be ideal for all applications, so consider switching to **millimeters** in the future.

## Requirements

The following Python packages are required to run the code:

- `torch` (PyTorch)
- `matplotlib`
- `numpy`

Install them using pip:

```bash
pip install torch matplotlib numpy
```

## Future Improvements

- Replace the fake network with a real surrogate model.
- Refactor the code to use millimeters as the default unit.


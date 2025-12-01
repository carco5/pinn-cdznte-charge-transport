# Physics-Informed Neural Networks for Charge Transport in CdZnTe Detectors

This repository contains the main Python scripts used in the thesis

> *“Physics-Informed Neural Networks for Charge Transport in Semiconductor Radiation Detectors”*.

The code implements:
- A **drift-only finite-difference baseline simulator** on a 2D grid.
- A **5D Physics-Informed Neural Network (PINN)** that learns the time–space evolution of an electron packet in a planar CdZnTe-like detector,
  \[
    f_\theta : (t,x,y,x_0,y_0) \mapsto n_\theta(t,x,y; x_0,y_0).
  \]
  
The PINN combines drift, diffusion and trapping with several loss components (IC, PDE, mass, boundary and outer ring terms) and generates the diagnostics analysed in the thesis.

---

## Repository contents

The core files are:

- `baseline_simulator.py`  
  Drift-only finite-difference simulator on a 2D Cartesian grid.  
  It uses a prescribed electrostatic potential to compute the electric field and evolves a Gaussian electron packet under pure drift.  
  This script is included mainly as **reference** for the non-differentiable baseline described in the thesis.

- `pinn_5d_full_suite.py`  
  Full 5D PINN implementation with:
  - Geometry and physics parameters defined in memory.
  - Curriculum training for the composite loss (IC + PDE + MASS + BC + RING, plus an optional anchor term).
  - Four training **phases** (A–D) corresponding to:
    - Phase A: Ablation study (16 loss combinations) + Welch *t*-tests on PDE residual RMS.
    - Phase B: Long training for the FULL model and a fixed ablation model.
    - Phase C: Hyperparameter tuning for the best ablation model.
    - Phase D: Long training of the MASS+BC+RING configuration used in the final results.
  - A full suite of **diagnostics**: loss curves, PDE residual RMS, initial-condition checks, spatio-temporal maps, mass decay vs. analytic law, COM trajectory and diffusion-driven broadening.

- `geometry.yaml`  
  YAML file with the planar detector geometry (domain size, mesh resolution) and electrode configuration used by the baseline simulator.

- `utils.py`  
  Small helper module providing utilities to load the geometry from `geometry.yaml` and to compute / cache the electrostatic potential used by `baseline_simulator.py`.

- `requirements.txt`  
  List of Python packages needed to run the PINN and the baseline code (JAX/JAXlib with CUDA support, Equinox, Optax, NumPy, Matplotlib, SciPy, etc.).

- `README.md`  
  This file.

---

## Python and dependencies

The code was developed and tested with **Python 3.11** on a GPU cluster.

Create a virtual environment if desired and install the dependencies with:

```bash
pip install -r requirements.txt

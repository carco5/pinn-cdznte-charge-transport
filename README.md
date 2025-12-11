# Physics-Informed Neural Networks for Charge Transport in CdZnTe Detectors

This repository contains the main Python scripts used in the thesis

> *“Physics-Informed Neural Networks for Charge Transport in Semiconductor Radiation Detectors”*.

The code implements:
- A **drift-only finite-difference baseline simulator** on a 2D grid.
- A **5D Physics-Informed Neural Network (PINN)** that learns the time–space evolution of an electron packet in a planar CdZnTe-like detector. The network approximates the electron density  
  `f_θ : (t, x, y, x0, y0) → n_θ(t, x, y; x0, y0)`.

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
  Helper module providing utilities to load the geometry from `geometry.yaml` and to compute / cache the electrostatic potential used by `baseline_simulator.py`.

- `requirements.txt`  
  List of Python packages needed to run the PINN and the baseline code (JAX/JAXlib with CUDA support, Equinox, Optax, NumPy, Matplotlib, SciPy, etc.).

- `README.md`  
  This file.

---

## Python and dependencies

The code was developed and tested with **Python 3.11** on a CUDA-enabled GPU cluster.

It is recommended to use a virtual environment. To install the dependencies:

```bash
pip install -r requirements.txt
Running the baseline simulator
The drift-only baseline is CPU-friendly and can be run directly:


python baseline_simulator.py
This script:

Reads the detector geometry from geometry.yaml via utils.py.

Computes (or loads from disk) the electrostatic potential.

Evolves a Gaussian electron packet under pure drift.

Produces a set of 2D plots illustrating the evolution and boundary artefacts.

The plots are displayed with Matplotlib and are intended as a non-differentiable reference for the PINN results.

Running the 5D PINN
The 5D PINN script is designed to run on a CUDA-enabled GPU.

The training / evaluation workflow is selected via the PINN_PHASE environment variable:

PINN_PHASE=A – Ablation study (16 loss combinations) + Welch t-tests on PDE residual RMS.

PINN_PHASE=B – Long training for the FULL model and for a fixed ablation model (MASS+BC+RING).

PINN_PHASE=C – Hyperparameter tuning for the best ablation model.

PINN_PHASE=D – Long training of the MASS+BC+RING configuration used in the final results.

Example commands:


# Phase A: ablation + pi-test
PINN_PHASE=A python pinn_5d_full_suite.py

# Phase D: long MASS+BC+RING run (1200 epochs, ramp 800)
PINN_PHASE=D python pinn_5d_full_suite.py
Output and logs
By default, figures and numerical summaries are saved under:


ablation/figs/<RUN_TAG>/
where <RUN_TAG> is generated automatically from the current date/time, and can be overridden via the PINN_RUN_TAG environment variable. The root directory for figures can be changed with PINN_FIGS_DIR if needed.

Each run produces:

PNG and PDF figures (training curves, residuals, spatio-temporal maps, diagnostics).

Text and JSON files with the main metrics used in the thesis (e.g. PDE residual RMS, ablation statistics and pi-test results).
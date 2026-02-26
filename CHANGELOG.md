# Changelog

All notable changes to turbomodal are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-02-26

Initial production release.

### Added

**Subsystem A -- FEA & Geometry (C++)**

- Cyclic symmetry modal solver (`CyclicSymmetrySolver`) with shift-invert
  Lanczos via the Spectra library.
- Quadratic tetrahedral (TET10) elements with Keast Rule 6 mass quadrature.
- CAD import from STEP, IGES, and BREP via gmsh OpenCASCADE kernel
  (`load_cad`, `inspect_cad`, `plot_cad`).
- Mesh import from gmsh MSH, NASTRAN, Abaqus, VTK, CGNS, MED, and XDMF
  (`load_mesh`).
- Automatic cyclic boundary detection and periodic node matching.
- Interactive 3D boundary condition editor (`bc_editor`).
- Centrifugal stiffening, spin softening, and Coriolis/gyroscopic effects.
- Kwak AVMI added mass model for fluid-coupled vibration (`AddedMassModel`).
- Potential flow BEM for submerged disk vibration.
- Damping models: Rayleigh, modal, and aerodynamic (`DampingConfig`).
- Harmonic forced response solver with modal superposition
  (`ForcedResponseSolver`).
- Fundamental Mistuning Model (FMM) solver (`FMMSolver`).
- C++ mode identification: nodal circle counting, family classification,
  and whirl direction (`identify_modes`).

**

- Virtual sensor array with BTT probes, strain gauges, and accelerometers
  (`_RemovedClass`, `_RemovedClass`).
- Signal synthesis pipeline (`_removed`).
- Noise models: Gaussian, harmonic interference, signal drift, bandwidth
  limiting, ADC quantisation, and signal dropout (`_RemovedClass`, `_removed`).
- HDF5 dataset management with parametric sweeps driven by Latin Hypercube
  Sampling (`_removed`).

**

- 6-tier complexity ladder: Linear, Tree (Internal model/Internal model/RF), SVM,
  Shallow NN, 1-D CNN (with ResNet variant), Conv+BiLSTM (with Transformer
  variant).
- Physics-informed feature extraction: spectrogram, mel, order tracking,
  traveling wave decomposition, cross-spectral density.
- Automatic hyperparameter optimization via Internal TPE sampling.
- Uncertainty quantification: MC Dropout, Deep Ensembles, heteroscedastic
  output heads.
- Internal tracker experiment tracking with automatic no-op fallback.

**

- Sensor placement optimization: Fisher Information pre-screening, greedy
  forward selection, Bayesian refinement, minimize-sensors mode.
- SHAP_REMOVEDvalues and Grad-CAM attribution for model explainability.
- Physics consistency validation (6 rule-based checks).
- Confidence calibration: Platt, isotonic, temperature scaling, conformal
  prediction.
- Per-prediction explanation cards and model selection reports.

**Visualization**

- Mesh and full-annulus mesh plotting (`plot_mesh`, `plot_full_mesh`).
- Mode shape visualization with full annulus reconstruction and animation
  (`plot_mode`).
- Campbell diagram with MAC-based mode tracking (`plot_campbell`).
- ZZENF interference diagram (`plot_zzenf`).
- Contribution analysis heatmap (`_removed_func`).

**Testing**

- 15 C++ test suites covering material, element, mesh, assembly, solver,
  cyclic symmetry, added mass, potential flow, rotating effects, damping,
  forced response, mistuning, mode identification, validation, and
  integration.
- 13 Python test files covering bindings, I/O, visualization, solver,
  datasets, parametric sweeps, sensors, noise, signal generation, ML
  pipeline, optimization, validation, and end-to-end integration.
- Validation against Leissa plate theory, Kwak added mass analytical
  solutions, Coriolis splitting, FMM tuned-system identity, and SDOF FRF.

**Platform Support**

- Linux (x86_64), macOS (x86_64, arm64), Windows (AMD64).
- Python 3.9, 3.10, 3.11, 3.12.
- BLAS/LAPACK acceleration on Linux and macOS.

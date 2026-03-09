# Changelog

All notable changes to turbomodal are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Full-annulus virtual probe signal generation model with correct
  circumferential phase factor (`-w·k·θ_s`) for stationary-frame
  sensors (`generate_signals_for_condition`).
- Rotating vs stationary sensor distinction: strain gauges observe
  rotating-frame frequency; BTT probes and casing accelerometers
  observe Doppler-shifted stationary-frame frequency.
- Blade passage gating via ray tracing against the full-annulus
  surface mesh — works for any stationary sensor and orientation,
  no tolerance heuristics required.
- BTT discrete output: `btt_arrival_times`, `btt_deflections`, and
  `btt_blade_indices` in signal generation result dict.
- `is_stationary` field on `SensorLocation` for explicit rotating/
  stationary classification (inferred from `sensor_type` by default).
- `mesh` property on `VirtualSensorArray`.
- `sensor_circumferential_angles()` method on `VirtualSensorArray`.
- `blade_tip_profile()` method on `VirtualSensorArray`.
- Time vector control on `SignalGenerationConfig`: `time` (custom
  array), `t_start`, `t_end`, and `damping_ratio` fields.

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

**Subsystem B -- Signals & Noise (Python)**

- Virtual sensor array with BTT probes, strain gauges, accelerometers, and
  displacement sensors (`VirtualSensorArray`, `SensorArrayConfig`).
- Signal synthesis pipeline (`generate_signals_for_condition`).
- Noise models: Gaussian, harmonic interference, sensor drift, bandwidth
  limiting, ADC quantisation, and signal dropout (`NoiseConfig`, `apply_noise`).
- HDF5 dataset management with parametric sweeps driven by Latin Hypercube
  Sampling (`run_parametric_sweep`).

**Subsystem C -- ML Pipeline (Python)**

- 6-tier complexity ladder: Linear, Tree (XGBoost/LightGBM/RF), SVM,
  Shallow NN, 1-D CNN (with ResNet variant), Conv+BiLSTM (with Transformer
  variant).
- Physics-informed feature extraction: spectrogram, mel, order tracking,
  traveling wave decomposition, cross-spectral density.
- Automatic hyperparameter optimization via Optuna TPE sampling.
- Uncertainty quantification: MC Dropout, Deep Ensembles, heteroscedastic
  output heads.
- MLflow experiment tracking with automatic no-op fallback.

**Subsystem D -- Optimisation & Explainability (Python)**

- Sensor placement optimization: Fisher Information pre-screening, greedy
  forward selection, Bayesian refinement, minimize-sensors mode.
- SHAP values and Grad-CAM attribution for model explainability.
- Physics consistency validation (6 rule-based checks).
- Confidence calibration: Platt, isotonic, temperature scaling, conformal
  prediction.
- Per-prediction explanation cards and model selection reports.

**Visualization**

- Mesh and full-annulus mesh plotting (`plot_mesh`, `plot_full_mesh`).
- Mode shape visualization with full annulus reconstruction and animation
  (`plot_mode`).
- Campbell diagram with MAC-based mode tracking, stator vane NPF lines,
  and `DiagramStyle` configuration (`plot_campbell`).
- ZZENF interference diagram with EO zig-zag, stator vane horizontal
  lines, crossing markers, confidence bands, and `DiagramStyle`
  configuration (`plot_zzenf`).
- Frequency diagnostic comparison against ground truth with error
  heatmaps, per-ND bar charts, and parity scatter plots
  (`diagnose_frequencies`).
- Sensor contribution heatmap (`plot_sensor_contribution`).

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

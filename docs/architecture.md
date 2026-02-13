# Architecture

## Overview

turbomodal is structured around four subsystems that form an end-to-end
pipeline for turbomachinery modal identification. Data flows from finite
element analysis through signal synthesis and feature extraction into
machine learning models, with optimisation and explainability layered on
top.

The four subsystems are:

| Subsystem | Concern                      | Language | Key Modules              |
|-----------|------------------------------|----------|--------------------------|
| A         | FEA and Geometry             | C++      | `_core`, `io`, `solver`  |
| B         | Signals and Noise            | Python   | `sensors`, `noise`, `signal_gen` |
| C         | ML Pipeline                  | Python   | `ml.features`, `ml.models`, `ml.pipeline` |
| D         | Optimisation and Explainability | Python | `optimization.sensor_placement`, `optimization.explainability` |

## System Architecture

```
                     +---------------------+
                     |  CAD / Mesh File    |
                     | (.step, .msh, .bdf) |
                     +---------+-----------+
                               |
                               v
                  +------------+----------------+
                  |       Subsystem A           |
                  |    FEA & Geometry (C++)     |
                  |                             |
                  |  load_cad() / load_mesh()   |
                  |             |               |
                  |             v               |
                  |    CyclicSymmetrySolver     |
                  |      .solve_at_rpm()        |
                  |             |               |
                  |             v               |
                  |  ModalResult                |
                  |    .frequencies             |
                  |    .mode_shapes             |
                  |    .harmonic_index          |
                  |    .whirl_direction         |
                  +--------------+--------------+
                                 |               
                  mode shapes,   |         HDF5 dataset
                  frequencies    |     (export_modal_results)
                                 v               |
                  +-------------------------+    |
                  |       Subsystem B       |    |
                  |  Signals & Noise (Py)   |    |
                  |                         |    |
                  |  VirtualSensorArray     |    |
                  |    .sample_mode_shape() |    |
                  |    .generate_time_      |    |
                  |        signal()         |    |
                  |                         |    |
                  |  NoiseConfig            |    |
                  |    + apply_noise()      |    |
                  |                         |    |
                  |  SignalGenerationConfig |    |
                  |    + generate_signals_  |    |
                  |       for_condition()   |    |
                  +------------+------------+    |
                               |                 |
                    synthetic  |                 |
                    signals    |                 |
                               v                 v
                  +------------+-------------------+
                  |       Subsystem C              |
                  |    ML Pipeline (Python)        |
                  |                                |
                  |  extract_features()            |
                  |    (STFT, mel, order tracking, |
                  |     TWD, cross-spectral)       |
                  |          |                     |
                  |          v                     |
                  |  build_feature_matrix()        |
                  |          |                     |
                  |          v                     |
                  |  train_mode_id_model()         |
                  |    Complexity ladder           |
                  |    Tiers 1-6                   |
                  |          |                     |
                  |          v                     |
                  |  ModeIDModel.predict()         |
                  |    -> nodal_diameter           |
                  |    -> whirl_direction          |
                  |    -> amplitude                |
                  |    -> wave_velocity            |
                  |    -> confidence               |
                  +--------------+-----------------+
                                 |
                     predictions |
                                 v
                  +--------------+----------------+
                  |       Subsystem D             |
                  |  Optimisation &               |
                  |  Explainability (Python)      |
                  |                               |
                  |  optimize_sensor_placement()  |
                  |  compute_shap_values()        |
                  |  compute_grad_cam()           |
                  |  physics_consistency_check()  |
                  |  calibrate_confidence()       |
                  +-------------------------------+
```

## Subsystem A: FEA and Geometry

### C++ Backend

The core finite element solver is written in C++17 and compiled as a static
library `turbomodal_core`. It is exposed to Python through pybind11 bindings
compiled as the `turbomodal._core` extension module.

Source files in `src/`:

| File                    | Responsibility                                     |
|-------------------------|----------------------------------------------------|
| `material.cpp`          | Isotropic material with temperature dependence      |
| `element.cpp`           | TET10 element stiffness and mass matrices           |
| `mesh.cpp`              | Mesh I/O (ZZENM format, gmsh), boundary detection   |
| `assembler.cpp`         | Global sparse matrix assembly                      |
| `rotating_effects.cpp`  | Spin-softening and centrifugal stiffening           |
| `modal_solver.cpp`      | Standard eigenvalue solver (Spectra)                |
| `cyclic_solver.cpp`     | Cyclic symmetry reduction and harmonic-index loop   |
| `added_mass.cpp`        | Fluid-structure added mass (Kwak model)             |
| `damping.cpp`           | Proportional and modal damping                      |
| `forced_response.cpp`   | Engine-order excitation forced response             |
| `mistuning.cpp`         | Fundamental Mistuning Model (FMM)                   |
| `mode_identification.cpp` | Nodal circle counting, mode family classification |

### CyclicSymmetrySolver

The central solver class. It exploits the rotational periodicity of a bladed
disk with `N` sectors by solving only a single-sector mesh. For each harmonic
index `k` in `[0, N/2]`, cyclic boundary conditions couple left and right
boundary DOFs through the complex phase factor `exp(j * 2*pi*k/N)`.

```
CyclicSymmetrySolver(mesh, material, fluid)
  .solve_at_rpm(rpm, num_modes_per_harmonic)
      -> list[ModalResult]   (one per harmonic index)
```

Each `ModalResult` contains:

- `frequencies` -- natural frequencies in Hz
- `mode_shapes` -- complex eigenvectors, shape `(n_dof, n_modes)`
- `harmonic_index` -- the nodal diameter count
- `whirl_direction` -- `+1` forward, `-1` backward, `0` standing wave
- `rpm` -- the rotational speed at which the solve was performed

### TET10 Element

All meshes use 10-node quadratic tetrahedra. The `TET10Element` class
computes 30x30 element stiffness and consistent mass matrices via 4-point
Gauss quadrature. Quadratic shape functions provide second-order convergence
for displacement and first-order convergence for stress.

### Mesh I/O

Two Python-side entry points handle geometry import:

- `load_cad(filepath, num_sectors, ...)` -- imports STEP/IGES/BREP/STL
  geometry via gmsh, automatically detects cyclic boundary surfaces, and
  generates a TET10 mesh.
- `load_mesh(filepath, num_sectors, ...)` -- imports pre-meshed files via
  meshio (NASTRAN, Abaqus, VTK, CGNS, Salome MED, XDMF) or gmsh MSH format
  via the native C++ loader.

### External Dependencies (C++)

- **Eigen 3.4.0** -- dense and sparse linear algebra. Downloaded
  automatically by CMake via `FetchContent` to avoid version conflicts with
  system Eigen.
- **Spectra** -- header-only eigenvalue solver for sparse matrices. Included
  as a git submodule at `external/spectra/`.

### Mode Identification (C++)

The `mode_identification.cpp` module provides:

- `identify_nodal_circles(mode_shape, mesh)` -- counts radial zero-crossings
  of the dominant displacement component.
- `classify_mode_family(mode_shape, mesh)` -- returns `"B"` (bending),
  `"T"` (torsion), or `"A"` (axial) based on the dominant DOF component.
- `identify_modes(result, mesh)` -- produces a vector of `ModeIdentification`
  structs with nodal diameter, nodal circle, whirl direction, frequency,
  wave velocity, participation factor, and family label.

## Subsystem B: Signals and Noise

### VirtualSensorArray

Defined in `sensors.py`. Maps FE mode shapes to discrete sensor locations
using a sparse interpolation matrix. Key operations:

- `build_interpolation_matrix()` -- constructs a `(n_sensors, n_dof)` sparse
  CSR matrix by finding the nearest mesh node to each sensor and projecting
  the three displacement DOFs onto the sensor measurement direction.
- `sample_mode_shape(mode_shape)` -- returns a complex `(n_sensors,)` vector
  of projected sensor readings for one mode.
- `generate_time_signal(modal_results, rpm, ...)` -- synthesises a
  `(n_sensors, n_samples)` time-domain signal by superposing all modal
  contributions.

Three sensor types are supported via the `SensorType` enum:

| Type                   | Typical Use                         |
|------------------------|-------------------------------------|
| `BTT_PROBE`            | Blade tip timing (radial direction) |
| `STRAIN_GAUGE`         | On-blade strain (tangential)        |
| `CASING_ACCELEROMETER` | Casing vibration monitoring         |

Convenience constructors `SensorArrayConfig.default_btt_array()` and
`SensorArrayConfig.default_strain_gauge_array()` generate uniformly spaced
circumferential sensor rings.

### Noise Model

Defined in `noise.py`. The `NoiseConfig` dataclass controls six noise effects
applied in sequence by `apply_noise()`:

1. Harmonic interference -- sinusoidal tones at configurable frequencies.
2. Additive Gaussian white noise at a specified SNR (dB).
3. Bandwidth limiting -- Butterworth low-pass filter.
4. Sensor drift -- linear ramp or random walk.
5. ADC quantisation -- bit-depth simulation.
6. Signal dropout -- random per-sample zeroing.

### Signal Generation Pipeline

Defined in `signal_gen.py`. The `SignalGenerationConfig` dataclass controls
sample rate, duration, amplitude mode (`"unit"`, `"forced_response"`,
`"random"`), and mode filtering. Two orchestrator functions:

- `generate_signals_for_condition()` -- generates signals for a single
  operating condition from a list of `ModalResult` objects.
- `generate_dataset_signals()` -- generates signals for an entire parametric
  dataset across multiple operating conditions.

## Subsystem C: ML Pipeline

### Design Rationale -- Complexity Ladder

The ML pipeline implements an iterative complexity ladder. The idea is to
start with the simplest possible model and escalate only when simpler
approaches fail to meet performance targets. This design provides:

- Interpretability by default (Tier 1-2 models are fully inspectable).
- Faster iteration during early development.
- A principled stopping criterion: the ladder halts when all four performance
  targets are met or when the marginal improvement between tiers drops below
  `performance_gap_threshold`.

### Tier Definitions

| Tier | Class                   | Architecture                            | Interpretability |
|------|-------------------------|-----------------------------------------|------------------|
| 1    | `LinearModeIDModel`     | LogisticRegression + Ridge              | Full             |
| 2    | `TreeModeIDModel`       | XGBoost (RandomForest fallback)         | High             |
| 3    | `SVMModeIDModel`        | SVC + SVR with RBF kernel, StandardScaler | Medium         |
| 4    | `ShallowNNModeIDModel`  | 2-hidden-layer MLP (128 -> 64), PyTorch | Medium           |
| 5    | `CNNModeIDModel`        | 1-D CNN (Conv-BN-ReLU x2 + pool), PyTorch | Low            |
| 6    | `TemporalModeIDModel`   | Conv front-end + BiLSTM, PyTorch        | Low              |

All tiers implement the `ModeIDModel` protocol, which defines four methods:

```python
class ModeIDModel(Protocol):
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
```

Every model predicts a dict with keys: `nodal_diameter`, `nodal_circle`,
`frequency`, `whirl_direction`, `amplitude`, `wave_velocity`, `confidence`.

### Feature Extraction

Defined in `ml/features.py`. The `FeatureConfig` dataclass selects one of
four feature types:

| `feature_type`    | Description                                    |
|-------------------|------------------------------------------------|
| `"spectrogram"`   | Time-averaged STFT magnitude per sensor        |
| `"mel"`           | Mel-scale filterbank applied to STFT magnitude |
| `"order_tracking"`| Complex amplitude at integer engine orders     |
| `"twd"`           | Traveling Wave Decomposition via spatial DFT   |

An optional cross-spectral density overlay (`include_cross_spectra=True`)
appends coherence-thresholded CSD magnitudes for all sensor pairs.

The `extract_features()` function takes `(n_sensors, n_samples)` signals and
returns a 1-D feature vector. The `build_feature_matrix()` function
automates the full loop: load HDF5 dataset, generate signals per condition,
extract features, and collect ground-truth labels into `(X, y)` arrays.

### Training Pipeline

Defined in `ml/pipeline.py`. The `train_mode_id_model()` function drives the
complexity ladder:

1. **Data loading** -- accepts pre-built `(X, y)` arrays or a `.npz` file.
2. **Data splitting** -- condition-based splitting via `GroupShuffleSplit` to
   prevent data leakage between operating conditions that share physical
   parameters. Falls back to random splitting when `split_by_condition=False`.
3. **Complexity ladder** -- iterates tiers 1 through `max_tier`, trains each,
   evaluates on the validation set, and stops when targets are met or
   diminishing returns are detected.
4. **Final evaluation** -- the best model is evaluated on a held-out test set.
5. **MLflow logging** -- all metrics, parameters, and timing are logged
   automatically. The `_MLflowProxy` class silently no-ops when mlflow is
   not installed.

### Performance Targets

Configured via `TrainingConfig`:

| Metric              | Default Target | Description                              |
|---------------------|----------------|------------------------------------------|
| `mode_detection_f1` | >= 0.92        | Macro F1 on encoded (ND, NC) labels      |
| `whirl_accuracy`    | >= 0.95        | Balanced accuracy on whirl direction      |
| `amplitude_mape`    | <= 0.08        | Mean Absolute Percentage Error on amplitude |
| `velocity_r2`       | >= 0.93        | R-squared on wave velocity               |

### Prediction Targets

Every model produces predictions for five quantities:

1. `nodal_diameter` -- integer, the circumferential wave number.
2. `whirl_direction` -- `+1` (forward), `-1` (backward), `0` (standing).
3. `amplitude` -- peak vibration amplitude.
4. `wave_velocity` -- circumferential wave propagation speed in m/s.
5. `confidence` -- model confidence score in `[0, 1]`.

(Frequency is currently passed through from the feature extraction stage
rather than predicted by the ML model.)

## Subsystem D: Optimisation and Explainability

### Sensor Placement Optimisation

Defined in `optimization/sensor_placement.py`. The optimisation proceeds in
four stages:

**Stage 1 -- FIM Pre-screening.** `compute_fisher_information()` evaluates
the trace of the Fisher Information Matrix at each candidate sensor position.
Candidates are ranked by information gain and filtered by a minimum angular
spacing constraint.

**Stage 2 -- Greedy Forward Selection.** Sensors are added one at a time,
choosing the candidate that maximises `log det(FIM)` (D-optimal design). The
greedy loop runs until `max_sensors` is reached.

**Stage 3 -- Bayesian Refinement (optional).** When
`optimization_method="bayesian"`, Optuna's TPE sampler fine-tunes the angular
positions of the sensors selected in Stage 2 within the minimum-spacing
tolerance band.

**Stage 4 -- Robustness Validation.** Monte Carlo trials simulate sensor
dropout and position perturbation. The output `SensorOptimizationResult`
reports the fraction of trials that maintained acceptable conditioning and the
mean degradation from single-sensor dropout.

The full pipeline is exposed as:

```python
optimize_sensor_placement(mesh, modal_results, config) -> SensorOptimizationResult
```

The `SensorOptimizationResult` dataclass provides `sensor_positions`,
`num_sensors`, `objective_value`, `observability_matrix`,
`condition_number`, `robustness_score`, and `dropout_degradation`.

### Observability Analysis

`compute_observability()` computes the condition number and singular values
of the sensor-space mode shape matrix, plus the Modal Assurance Criterion
(MAC) matrix for all mode pairs. A high condition number indicates that some
modes are poorly distinguishable with the current sensor configuration.

### SHAP Values

`compute_shap_values(model, signals)` uses TreeSHAP for tree-based models
(Tier 2) and KernelSHAP for all other model types. Returns a
`(n_samples, n_features, n_outputs)` array showing the contribution of each
feature to each of the four prediction targets (mode class, whirl, amplitude,
velocity).

### Grad-CAM

`compute_grad_cam(model, signals, target_class)` is applicable to Tiers 5-6
only. It registers forward and backward hooks on the last convolutional layer,
computes gradient-weighted activation maps, and returns a `(batch, length)`
heatmap normalised to `[0, 1]` showing which spectral regions drove the
classification decision.

### Physics Consistency Check

`physics_consistency_check(predictions, num_sectors, rpm, blade_radius)`
applies five rule-based constraints:

1. **Positive frequency** -- `f > 0`.
2. **Valid nodal diameter** -- `0 <= ND <= N/2` where `N` is the blade count.
3. **Valid whirl direction** -- whirl in `{-1, 0, +1}`.
4. **Whirl frequency ordering** -- for each ND group, the minimum forward-whirl
   frequency must be >= the maximum backward-whirl frequency (gyroscopic
   splitting).
5. **Velocity consistency** -- `v_predicted` agrees with
   `2 * pi * f * R / ND` within 50% (for `ND > 0`).

Returns `is_consistent`, `violations`, `consistency_score`, and
`anomaly_flag` arrays.

### Confidence Calibration

`calibrate_confidence(model, X_val, y_val, method)` wraps a trained model in
a `CalibratedModel` that transforms raw confidence scores. Four methods are
available:

| Method        | Technique                                              |
|---------------|--------------------------------------------------------|
| `"platt"`     | Logistic regression on raw confidence vs. correctness  |
| `"isotonic"`  | Isotonic regression on raw confidence vs. correctness  |
| `"temperature"` | Temperature scaling that minimises binary cross-entropy |
| `"conformal"` | Conformal prediction intervals for regression targets  |

The conformal method additionally adds `prediction_interval_lower` and
`prediction_interval_upper` keys to the prediction output, providing 90%
coverage intervals for amplitude and wave velocity.

## C++ / Python Boundary

The boundary between C++ and Python is the pybind11 module `turbomodal._core`,
built from `src/python_bindings.cpp`. It exposes the following C++ types and
functions to Python:

**Classes:**
`Material`, `NodeSet`, `Mesh`, `GlobalAssembler`, `SolverConfig`,
`SolverStatus`, `ModalResult`, `FluidConfig`, `FluidType`,
`CyclicSymmetrySolver`, `AddedMassModel`, `ModalSolver`, `DampingConfig`,
`DampingType`, `ForcedResponseConfig`, `ForcedResponseResult`,
`ForcedResponseSolver`, `ExcitationType`, `MistuningConfig`,
`MistuningResult`, `FMMSolver`, `ModeIdentification`, `GroundTruthLabel`.

**Free functions:**
`identify_nodal_circles()`, `classify_mode_family()`, `identify_modes()`.

Data exchange between C++ and Python uses Eigen-to-NumPy automatic conversion
provided by pybind11. Sparse matrices are passed as `scipy.sparse.csr_matrix`
where appropriate. Complex mode shapes are handled as `numpy.complex128`
arrays.

The Python layer adds:

- Geometry import (`io.py`) via gmsh and meshio.
- High-level solver wrappers (`solver.py`) with progress bars and
  Campbell diagram extraction.
- Visualisation (`viz.py`) via PyVista and Matplotlib.
- Signal synthesis (`sensors.py`, `noise.py`, `signal_gen.py`).
- HDF5 dataset management (`dataset.py`, `parametric.py`).
- The entire ML and optimisation pipeline (`ml/`, `optimization/`).

## Design Decisions

### Why Cyclic Symmetry

A bladed disk with `N` identical sectors has `N`-fold rotational symmetry.
Exploiting this reduces the eigenvalue problem from the full annulus
(`N * n_dof_sector` DOFs) to a single sector (`n_dof_sector` DOFs) solved
`N/2 + 1` times -- one per harmonic index. For a typical 36-blade disk this
is an 18x reduction in problem size, making parametric sweeps over thousands
of operating conditions tractable.

### Why the Complexity Ladder

Classical approaches to turbomachinery modal identification use
physics-based signal processing (order tracking, traveling wave
decomposition). These work well under clean conditions but degrade with noise,
sensor failures, or closely spaced modes. Pure deep learning approaches
require large labelled datasets that are expensive to obtain from physical
test rigs.

The complexity ladder bridges these extremes: Tier 1-3 models are fast,
interpretable, and need only moderate data. If they fail to meet targets, the
pipeline automatically escalates to neural network tiers that can learn
nonlinear patterns from the synthetic datasets generated by Subsystems A-B.

### Why Condition-Based Splitting

Operating conditions (RPM, temperature, pressure ratio) define the physics of
a measurement. Randomly splitting individual samples would allow the model to
see signals from the same operating condition in both training and test sets,
creating data leakage. The `GroupShuffleSplit` strategy ensures entire
conditions are held out, producing realistic generalisation estimates.

### Deferred Imports Pattern

Optional dependencies (PyTorch, XGBoost, SHAP, Optuna, MLflow) are imported
inside function bodies rather than at module level. This allows users to
`import turbomodal` and use the core FEA functionality without installing
ML-specific packages. An `ImportError` is raised only when a specific
function that requires the missing dependency is called.

### HDF5 Dataset Layout

Modal results are stored in a hierarchical HDF5 layout with groups for mesh
data, operating conditions (as a structured NumPy array), and per-condition
modal results indexed by condition ID. Mode shapes are stored as
`(n_harmonics, n_modes, n_dof)` complex128 arrays with optional gzip
compression. This layout supports random access to individual conditions
without loading the entire file.

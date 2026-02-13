# Quick Start Tutorial

## Overview

End-to-end walkthrough: load a mesh, solve cyclic symmetry FEA, synthesize
sensor signals, build a parametric dataset, extract features, train an ML
model, optimize sensor placement, and explain predictions.

---

## Step 1: Load a Mesh

```python
import turbomodal as tm

# From a mesh file (.msh, .bdf/.nas, .inp, .vtk/.vtu, .cgns, .med, .xdmf)
mesh = tm.load_mesh("blade_sector.msh", num_sectors=36)

# Or from CAD geometry (.step, .iges, .brep, .stl)
mesh = tm.load_cad(
    "blade_sector.step",
    num_sectors=36,
    mesh_size=0.005,
    order=2,
    auto_detect_boundaries=True,
)

print(f"Nodes: {mesh.num_nodes()}, Elements: {mesh.num_elements()}")
tm.plot_mesh(mesh, show_boundaries=True).show()
```

Turbomodal requires quadratic tetrahedra (TET10). The `load_cad` function
uses gmsh with the OpenCASCADE kernel and automatically detects cyclic
boundary surfaces (left/right cuts and hub).

---

## Step 2: Solve Cyclic Symmetry FEA

```python
mat = tm.Material(E=200e9, nu=0.3, rho=7800)
results = tm.solve(mesh, mat, rpm=3000, num_modes=10, verbose=1)

for r in results:
    freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:3])
    print(f"  ND={r.harmonic_index}: [{freqs}] Hz")
```

`solve` returns a list of `ModalResult` objects (one per harmonic index,
ND 0 through N/2). Each provides `frequencies`, `mode_shapes` (complex,
shape n_dof x n_modes), `whirl_direction` (+1 FW, -1 BW, 0 standing),
and `harmonic_index`.

For fluid coupling, pass a `FluidConfig`:

```python
fluid = tm.FluidConfig()
fluid.fluid_type = tm.FluidType.LIQUID
fluid.density = 1000.0
results = tm.solve(mesh, mat, rpm=3000, num_modes=10, fluid=fluid)
```

Visualize mode shapes:

```python
tm.plot_mode(mesh, results[2], mode_index=0, scale=0.001).show()
tm.plot_full_annulus(mesh, results[2], mode_index=0, scale=0.001).show()
```

---

## Step 3: RPM Sweep and Campbell Diagram

```python
import numpy as np

rpm_values = np.linspace(0, 15000, 20)
sweep_results = tm.rpm_sweep(mesh, mat, rpm_values, num_modes=10, verbose=1)

campbell = tm.campbell_data(sweep_results)
# campbell['frequencies'] shape: (N_rpm, N_harmonics, N_modes)

fig = tm.plot_campbell(sweep_results, engine_orders=[1, 2, 36])
fig = tm.plot_zzenf(sweep_results[-1], num_sectors=36)
```

---

## Step 4: Generate Sensor Signals

```python
from turbomodal import SensorArrayConfig, VirtualSensorArray, NoiseConfig, apply_noise

sensor_config = SensorArrayConfig.default_btt_array(
    num_probes=8, casing_radius=0.5,
    axial_positions=[0.0, 0.02],
    sample_rate=500_000.0, duration=1.0,
)
sensor_array = VirtualSensorArray(mesh, sensor_config)

clean_signal = sensor_array.generate_time_signal(
    modal_results=results, rpm=3000.0,
)  # shape: (n_sensors, n_samples)

noise_config = NoiseConfig(
    gaussian_snr_db=30.0,
    bandwidth_hz=200_000.0,
    drift_type="random_walk", drift_rate=0.001,
    adc_bits=16, adc_range=10.0,
    dropout_probability=0.001,
    harmonic_interference=[
        {"frequency_hz": 50.0, "amplitude_ratio": 0.05, "phase_deg": 0},
    ],
)
noisy_signal = apply_noise(clean_signal, noise_config, sample_rate=500_000.0)
```

Or use the end-to-end pipeline with `SignalGenerationConfig`:

```python
from turbomodal import SignalGenerationConfig, generate_signals_for_condition

signal_config = SignalGenerationConfig(
    sample_rate=500_000.0, duration=1.0,
    amplitude_mode="unit",       # "unit", "random", "forced_response"
    amplitude_scale=1e-6, seed=42,
)
sig_result = generate_signals_for_condition(
    sensor_array, results, rpm=3000.0,
    config=signal_config, noise_config=noise_config,
)
# sig_result keys: 'signals', 'clean_signals', 'time'
```

---

## Step 5: Build a Parametric Dataset

```python
from turbomodal import (
    ParametricRange, ParametricSweepConfig, DatasetConfig, run_parametric_sweep,
)

sweep_config = ParametricSweepConfig(
    ranges=[
        ParametricRange(name="rpm", low=1000, high=15000),
        ParametricRange(name="temperature", low=293.15, high=1073.15),
    ],
    num_samples=200, seed=42, num_modes=10,
    include_mistuning=True, mistuning_sigma=0.02,
)

dataset_config = DatasetConfig(
    output_path="turbomodal_dataset.h5",
    num_modes_per_harmonic=10,
    include_mode_shapes=True,
    compression="gzip", compression_level=4,
)

output_path = run_parametric_sweep(
    mesh, base_material=mat,
    config=sweep_config, dataset_config=dataset_config, verbose=1,
)
```

The sweep uses Latin Hypercube Sampling, adjusts material for temperature,
solves cyclic symmetry, and optionally applies the Fundamental Mistuning
Model (FMM). Results are exported to a structured HDF5 file.

Load an existing dataset:

```python
from turbomodal import load_modal_results
mesh_data, conditions, results_dict = load_modal_results("turbomodal_dataset.h5")
```

---

## Step 6: Extract Features

```python
from turbomodal.ml import FeatureConfig, extract_features, build_feature_matrix

feat_config = FeatureConfig(
    feature_type="spectrogram",  # "spectrogram", "mel", "order_tracking", "twd"
    fft_size=2048, hop_size=512, window="hann",
)
features = extract_features(sig_result['signals'], sample_rate=500_000.0, config=feat_config)
```

Feature types: **spectrogram** (time-averaged FFT magnitude), **mel**
(mel-scale filterbank, set `n_mels`, `f_min`, `f_max`), **order_tracking**
(integer engine order amplitudes, requires `rpm`), **twd** (traveling wave
decomposition via spatial DFT, requires `sensor_angles`). Add
cross-spectral density features with `include_cross_spectra=True`.

Build a full feature matrix from HDF5:

```python
X, y = build_feature_matrix(
    "turbomodal_dataset.h5", sensor_array, signal_config, feat_config,
)
# X: (n_samples, n_features)
# y keys: 'nodal_diameter', 'nodal_circle', 'whirl_direction', 'frequency', 'wave_velocity'
```

---

## Step 7: Train an ML Model

The complexity ladder trains tiers in order, stopping when performance
targets are met or improvement diminishes:

| Tier | Model                  | Description                           |
|----- |----------------------- |-------------------------------------- |
| 1    | LinearModeIDModel      | Logistic / Ridge regression           |
| 2    | TreeModeIDModel        | XGBoost / Random Forest               |
| 3    | SVMModeIDModel         | SVM with RBF kernel                   |
| 4    | ShallowNNModeIDModel   | 2-layer multi-task PyTorch net        |
| 5    | CNNModeIDModel         | 1-D CNN on spectral features          |
| 6    | TemporalModeIDModel    | Conv + BiLSTM on sequences            |

```python
from turbomodal.ml import TrainingConfig, train_mode_id_model

config = TrainingConfig(
    max_tier=6,
    performance_gap_threshold=0.02,
    validation_split=0.15, test_split=0.15,
    split_by_condition=True,
    batch_size=32, epochs=100, learning_rate=1e-3,
    use_optuna=True, optuna_trials=50,
    device="auto",
    mode_detection_f1_min=0.92,
    whirl_accuracy_min=0.95,
    amplitude_mape_max=0.08,
    velocity_r2_min=0.93,
    experiment_name="turbomodal_mode_id",
)

best_model, report = train_mode_id_model(X=X, y=y, config=config)
print(f"Best tier: {report['best_tier']}")
print(f"Test metrics: {report['test_metrics']}")
```

---

## Step 8: Evaluate and Predict

```python
from turbomodal.ml import evaluate_model, predict_mode_id

metrics = evaluate_model(best_model, X_test, y_test)
# Keys: 'mode_detection_f1', 'whirl_accuracy', 'amplitude_mape',
#        'amplitude_r2', 'velocity_rmse', 'velocity_r2'

predictions = predict_mode_id(
    model=best_model, signals=new_signals,
    sample_rate=500_000.0, rpm=5000.0,
)
# Keys: 'nodal_diameter', 'nodal_circle', 'frequency',
#        'whirl_direction', 'amplitude', 'wave_velocity', 'confidence'
```

---

## Step 9: Optimize Sensor Placement

```python
from turbomodal.optimization import (
    SensorOptimizationConfig, optimize_sensor_placement, compute_fisher_information,
)

opt_config = SensorOptimizationConfig(
    max_sensors=16, min_sensors=4,
    sensor_type="btt_probe",
    optimization_method="greedy",    # "greedy", "bayesian", "exhaustive"
    objective="fisher_info",         # "fisher_info", "mac_conditioning", "mutual_info"
    min_angular_spacing=5.0,
    feasible_radii=(0.4, 0.6),
    feasible_axial=(0.0, 0.05),
    robustness_trials=100,
    dropout_probability=0.05,
)

opt_result = optimize_sensor_placement(mesh, results, config=opt_config)
print(f"Sensors: {opt_result.num_sensors}, Objective: {opt_result.objective_value:.4f}")
print(f"Condition number: {opt_result.condition_number:.2f}")
print(f"Robustness: {opt_result.robustness_score:.2%}")
```

---

## Step 10: Explain Predictions

### SHAP values

```python
from turbomodal.optimization import compute_shap_values

shap_values = compute_shap_values(model=best_model, signals=X_test)
# Tree models: (n_samples, n_features, 4) -- one slice per sub-task
importance = np.abs(shap_values).mean(axis=(0, 2))
```

### Physics consistency check

```python
from turbomodal.optimization import physics_consistency_check

consistency = physics_consistency_check(
    predictions, num_sectors=36, rpm=5000.0, blade_radius=0.3,
)
# Keys: 'is_consistent', 'violations', 'consistency_score', 'anomaly_flag'
```

Checks: positive frequency, valid ND range, valid whirl direction,
forward/backward frequency ordering, and wave velocity consistency.

### Confidence calibration

```python
from turbomodal.optimization import calibrate_confidence

calibrated_model = calibrate_confidence(
    best_model, X_val, y_val, method="platt",
)  # Methods: "platt", "isotonic", "temperature", "conformal"

cal_preds = calibrated_model.predict(X_test)
```

The `conformal` method also provides `prediction_interval_lower` and
`prediction_interval_upper` keys (90% coverage).

---

## Next Steps

- **API Reference** -- See `docs/api/` for detailed class and function
  documentation.
- **Examples** -- `examples/python_example.py` demonstrates mesh loading,
  solving, and visualization. `examples/generate_visualizations.py` shows
  plotting workflows.
- **C++ Tests** -- Build with `BUILD_TESTS=ON` and run `ctest` to verify
  the solver against analytical benchmarks.
- **MLflow** -- Training metrics are logged automatically when mlflow is
  installed. Launch the dashboard with `mlflow ui`.

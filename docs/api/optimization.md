# turbomodal Optimization API Reference

This document covers the sensor placement optimization and explainability
module (`turbomodal.optimization`). It provides:

- Sensor placement optimization using Fisher Information Matrix and Bayesian
  optimization
- Explainability tools (SHAP, Grad-CAM) for model predictions
- Physics consistency checks for mode identification results
- Confidence calibration (Platt, isotonic, temperature scaling, conformal)

---

## turbomodal.optimization.sensor_placement

### SensorOptimizationConfig

Configuration for sensor placement optimization.

```python
@dataclass
class SensorOptimizationConfig:
    max_sensors: int = 16
    min_sensors: int = 4
    sensor_type: str = "btt_probe"
    optimization_method: str = "greedy"
    objective: str = "fisher_info"
    bayesian_iterations: int = 100
    bayesian_init_points: int = 10
    min_angular_spacing: float = 5.0
    feasible_radii: Optional[tuple[float, float]] = None
    feasible_axial: Optional[tuple[float, float]] = None
    robustness_trials: int = 100
    dropout_probability: float = 0.0
    position_tolerance: float = 0.0
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_sensors` | `int` | `16` | Maximum number of sensors |
| `min_sensors` | `int` | `4` | Minimum number of sensors |
| `sensor_type` | `str` | `"btt_probe"` | Sensor type: `"btt_probe"`, `"strain_gauge"`, `"casing_accel"` |
| `optimization_method` | `str` | `"greedy"` | Method: `"greedy"`, `"bayesian"`, `"exhaustive"` |
| `objective` | `str` | `"fisher_info"` | Objective function: `"fisher_info"`, `"mac_conditioning"`, `"mutual_info"` |
| `bayesian_iterations` | `int` | `100` | Bayesian optimization iterations (Optuna TPE sampler) |
| `bayesian_init_points` | `int` | `10` | Initial random points for Bayesian optimization |
| `min_angular_spacing` | `float` | `5.0` | Minimum angular spacing between probes in degrees |
| `feasible_radii` | `tuple \| None` | `None` | `(r_min, r_max)` feasible radial range |
| `feasible_axial` | `tuple \| None` | `None` | `(z_min, z_max)` feasible axial range |
| `robustness_trials` | `int` | `100` | Number of Monte Carlo robustness validation trials |
| `dropout_probability` | `float` | `0.0` | Probability of single sensor failure in robustness test |
| `position_tolerance` | `float` | `0.0` | Angular position tolerance in degrees for robustness test |
| `mode` | `str` | `"maximize_performance"` | Optimization mode: `"maximize_performance"` or `"minimize_sensors"` |
| `target_f1_min` | `float` | `0.92` | Target mode detection F1 (minimize_sensors mode) |
| `target_whirl_acc_min` | `float` | `0.95` | Target whirl accuracy (minimize_sensors mode) |
| `target_amp_mape_max` | `float` | `0.08` | Target amplitude MAPE (minimize_sensors mode) |
| `target_vel_r2_min` | `float` | `0.93` | Target velocity R2 (minimize_sensors mode) |
| `observability_penalty_weight` | `float` | `0.1` | Weight for condition-number penalty in Bayesian refinement |

### SensorOptimizationResult

Result container for sensor placement optimization.

```python
@dataclass
class SensorOptimizationResult:
    sensor_positions: np.ndarray       # (n_sensors, 3)
    num_sensors: int                   # Number of selected sensors
    objective_value: float             # Final objective function value
    objective_history: np.ndarray      # (n_steps,) objective per greedy step
    sensor_count_curve: np.ndarray     # (n_steps,) sensor count at each step
    observability_matrix: np.ndarray   # (n_modes, n_modes) MAC matrix
    condition_number: float            # Condition number of observability
    worst_observable_mode: str         # Label of worst-observable mode
    robustness_score: float            # Fraction of trials meeting targets
    dropout_degradation: float         # Performance loss with 1 sensor dropout
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `sensor_positions` | `ndarray (n_sensors, 3)` | Optimal sensor Cartesian coordinates |
| `num_sensors` | `int` | Number of selected sensors |
| `objective_value` | `float` | Final log-det(FIM) value |
| `objective_history` | `ndarray (n_steps,)` | Objective at each greedy selection step |
| `sensor_count_curve` | `ndarray (n_steps,)` | 1, 2, ..., n_sensors |
| `observability_matrix` | `ndarray (n_modes, n_modes)` | MAC matrix in sensor space |
| `condition_number` | `float` | Condition number of sensor-space mode shapes |
| `worst_observable_mode` | `str` | Label of worst-observed mode |
| `robustness_score` | `float` | Fraction of Monte Carlo trials passing threshold |
| `dropout_degradation` | `float` | Mean condition number degradation with dropout |

### compute_fisher_information

Compute the Fisher Information Matrix for a given sensor configuration.

```python
def compute_fisher_information(
    mode_shapes: np.ndarray,
    sensor_positions: np.ndarray,
    noise_covariance: Optional[np.ndarray] = None,
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode_shapes` | `ndarray (n_modes, n_dof)` | required | Complex mode shape matrix |
| `sensor_positions` | `ndarray` | required | `(n_sensors, n_dof)` interpolation matrix or `(n_sensors, 3)` positions |
| `noise_covariance` | `ndarray \| None` | `None` | `(n_sensors, n_sensors)` noise covariance (None = identity) |

**Returns:** `ndarray (n_modes, n_modes)` -- Real Fisher Information Matrix.

The FIM is computed as: `FIM = J^H * Sigma_inv * J` where `J = H @ Phi^T`
is the sensitivity matrix, `H` is the observation matrix, and `Phi` is
the mode shape matrix. When `sensor_positions` has 3 columns, a nearest-DOF
mapping is used to build `H`.

### compute_observability

Compute observability metrics for a sensor configuration.

```python
def compute_observability(
    mode_shapes: np.ndarray,
    interpolation_matrix: np.ndarray,
) -> dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode_shapes` | `ndarray (n_modes, n_dof)` | Complex mode shape matrix |
| `interpolation_matrix` | `ndarray (n_sensors, n_dof)` | Sensor interpolation matrix |

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"condition_number"` | `float` | Condition number of sensor-space modes |
| `"min_singular_value"` | `float` | Smallest singular value |
| `"mac_matrix"` | `ndarray (n_modes, n_modes)` | MAC matrix in sensor space |
| `"singular_values"` | `ndarray` | All singular values |

### optimize_sensor_placement

Optimize sensor placement for mode identification using a multi-stage strategy.

```python
def optimize_sensor_placement(
    mesh,
    modal_results: list,
    config: SensorOptimizationConfig = SensorOptimizationConfig(),
) -> SensorOptimizationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh \| None` | required | Mesh with `.nodes` (n_nodes, 3) or None |
| `modal_results` | `list` | required | List of ModalResult with `.mode_shapes` and `.frequencies` |
| `config` | `SensorOptimizationConfig` | `SensorOptimizationConfig()` | Optimization configuration |
| `ml_model_factory` | `callable \| None` | `None` | Optional callable `(positions) -> float` for ML-based objective in greedy selection |

**Returns:** `SensorOptimizationResult`

**Multi-stage strategy:**

1. **Stage 1 -- FIM Pre-screening:** Evaluate the trace of the single-sensor FIM
   for all candidate positions. Keep the top `4 * max_sensors` candidates,
   filtering by minimum angular spacing.
2. **Stage 2 -- Greedy Forward Selection:** Iteratively add the sensor that
   maximizes `log(det(FIM))` until `max_sensors` is reached.
3. **Stage 3 -- Bayesian Refinement (optional):** If
   `optimization_method="bayesian"`, refine angular positions of greedy-selected
   sensors using Optuna TPE sampler within `+/- min_angular_spacing/2`.
4. **Stage 4 -- Robustness Validation:** Monte Carlo assessment with sensor
   dropout and position perturbation.

**Example:**

```python
import turbomodal as tm
from turbomodal.optimization import (
    SensorOptimizationConfig, optimize_sensor_placement
)

config = SensorOptimizationConfig(
    max_sensors=8,
    min_angular_spacing=10.0,
    optimization_method="bayesian",
    bayesian_iterations=50,
    robustness_trials=200,
    dropout_probability=0.1,
    feasible_radii=(0.20, 0.30),
)

results_at_rpm = tm.solve(mesh, mat, rpm=10000, num_modes=15)
result = optimize_sensor_placement(mesh, results_at_rpm, config)

print(f"Sensors: {result.num_sensors}")
print(f"Objective: {result.objective_value:.2f}")
print(f"Condition number: {result.condition_number:.1f}")
print(f"Robustness: {result.robustness_score:.1%}")
print(f"Worst mode: {result.worst_observable_mode}")
```

---

## turbomodal.optimization.explainability

### compute_shap_values

Compute SHAP values for sensor channel importance.

```python
def compute_shap_values(
    model: Any,
    signals: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModeIDModel` | required | Trained model (must have `_is_fitted = True`) |
| `signals` | `ndarray (n_samples, n_features)` | required | Input features |
| `feature_names` | `list[str] \| None` | `None` | Optional feature names for labeling |

**Returns:**
- For tree models (`TreeModeIDModel`): `ndarray (n_samples, n_features, 4)` --
  One slice per estimator (mode_clf, whirl_clf, amp_reg, vel_reg). Uses
  TreeExplainer (exact, fast).
- For other models: `ndarray (n_samples, n_features, n_outputs)` -- Uses
  KernelExplainer with 100 background samples.

**Raises:** `ImportError` if the `shap` package is not installed;
`RuntimeError` if model is not fitted.

### compute_grad_cam

Compute Grad-CAM attribution for mode identification decisions.

```python
def compute_grad_cam(
    model: Any,
    signals: np.ndarray,
    target_class: int,
    layer_name: Optional[str] = None,
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `CNNModeIDModel \| TemporalModeIDModel` | required | Trained CNN/Temporal model (Tier 5-6 only) |
| `signals` | `ndarray (batch, n_features)` | required | Input features |
| `target_class` | `int` | required | Class index to explain (for mode detection head) |
| `layer_name` | `str \| None` | `None` | CNN layer for attribution (default: last conv layer) |

**Returns:** `ndarray (batch, length)` -- Grad-CAM heatmap normalized to [0, 1].

The Grad-CAM computation:
1. Registers forward/backward hooks on the target convolutional layer.
2. Performs a forward pass and backward pass for the target class.
3. Computes per-channel weights via global average pooling of gradients.
4. Produces a weighted combination of activations, passed through ReLU and
   normalized per sample.

**Raises:** `TypeError` if model is not CNN/Temporal; `ImportError` if PyTorch
is not installed; `RuntimeError` if model is not fitted.

### physics_consistency_check

Check predictions against known physical constraints.

```python
def physics_consistency_check(
    predictions: dict[str, np.ndarray],
    num_sectors: int,
    rpm: float = 0.0,
    blade_radius: float = 0.3,
) -> dict[str, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `dict` | required | Output from `model.predict()` |
| `num_sectors` | `int` | required | Number of blades/sectors |
| `rpm` | `float` | `0.0` | Rotational speed (0 = skip whirl ordering check) |
| `blade_radius` | `float` | `0.3` | Nominal blade tip radius in metres |
| `epistemic_uncertainty` | `ndarray \| None` | `None` | Per-sample epistemic uncertainty values |
| `epistemic_threshold` | `float` | `0.1` | Flag predictions with uncertainty above this threshold |

**Constraints checked:**

1. Frequency must be positive.
2. ND must be in range `[0, N/2]` for N sectors.
3. Whirl direction must be in `{-1, 0, 1}`.
4. Forward whirl frequency >= backward whirl frequency per ND group (when
   RPM > 0).
5. Wave velocity consistent with `v = 2*pi*f*R / ND` within 50% tolerance
   (when ND > 0).
6. Epistemic uncertainty below threshold (when `epistemic_uncertainty` is provided).

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"is_consistent"` | `bool (N,)` | True if all checks pass |
| `"violations"` | `list[list[str]]` | Per-sample violation descriptions |
| `"consistency_score"` | `float (N,)` | Fraction of checks passed [0, 1] |
| `"anomaly_flag"` | `bool (N,)` | True if consistency_score < 0.8 |

**Example:**

```python
from turbomodal.optimization import physics_consistency_check

preds = model.predict(X_test)
checks = physics_consistency_check(preds, num_sectors=24, rpm=10000)

n_anomalies = checks["anomaly_flag"].sum()
print(f"Anomalies: {n_anomalies}/{len(preds['frequency'])}")
for i, v in enumerate(checks["violations"]):
    if v:
        print(f"  Sample {i}: {v}")
```

### calibrate_confidence

Calibrate model confidence scores on validation data.

```python
def calibrate_confidence(
    model: Any,
    X_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    method: str = "platt",
) -> CalibratedModel
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModeIDModel` | required | Trained model |
| `X_val` | `ndarray (n_val, n_features)` | required | Validation features |
| `y_val` | `dict` | required | Validation labels (must have `"nodal_diameter"`) |
| `method` | `str` | `"platt"` | Calibration method |

**Supported methods:**

| Method | Description |
|--------|-------------|
| `"platt"` | Platt scaling -- logistic regression on raw confidence (requires scikit-learn) |
| `"isotonic"` | Isotonic regression on raw confidence (requires scikit-learn) |
| `"temperature"` | Temperature scaling -- minimizes binary cross-entropy to find optimal T (requires scipy) |
| `"conformal"` | Conformal prediction intervals for regression tasks with 90% coverage |

**Returns:** `CalibratedModel` wrapping the original model.

**Raises:** `RuntimeError` if model is not fitted; `ImportError` for missing
dependencies; `ValueError` for unknown method.

### CalibratedModel

Wrapper that applies confidence calibration to a base model.

```python
class CalibratedModel:
    def __init__(
        self,
        base_model: Any,
        calibration_transform: Callable[[np.ndarray], np.ndarray],
        method: str,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `base_model` | `ModeIDModel` | The underlying trained model |

The `predict` method delegates to `base_model.predict()` then applies
the fitted `calibration_transform` to the `"confidence"` key. All other
methods (`save`, `load`, `train`) delegate directly to `base_model`.

For the `"conformal"` method, an extended subclass adds
`"prediction_interval_lower"` and `"prediction_interval_upper"` keys
(each `ndarray (N, 2)` for amplitude and velocity) to the prediction dict.

**Example:**

```python
from turbomodal.optimization import calibrate_confidence

calibrated = calibrate_confidence(model, X_val, y_val, method="temperature")
preds = calibrated.predict(X_test)
print(preds["confidence"][:5])  # Calibrated confidence scores
```

### generate_model_selection_report

Generate a structured model selection report from training results.

```python
def generate_model_selection_report(
    training_report: dict,
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_report` | `dict` | Report dict from `train_mode_id_model()` |

**Returns:** dict with keys:
- `"summary"` -- Human-readable summary of selected tier and metrics.
- `"per_tier_metrics"` -- Dict mapping tier number to validation metrics.
- `"gap_analysis"` -- List of score deltas between consecutive tiers.
- `"selected_tier"` -- Tier number of the best model.

### generate_explanation_card

Generate a per-prediction explanation card.

```python
def generate_explanation_card(
    model: Any,
    X_single: np.ndarray,
    predictions: dict[str, np.ndarray],
    sample_idx: int = 0,
    num_sectors: int = 36,
    rpm: float = 0.0,
    feature_names: Optional[list[str]] = None,
    uncertainty: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModeIDModel` | required | Trained model |
| `X_single` | `ndarray` | required | `(1, n_features)` or `(n_features,)` single sample |
| `predictions` | `dict` | required | Full predictions dict from `model.predict()` |
| `sample_idx` | `int` | `0` | Index into predictions arrays |
| `num_sectors` | `int` | `36` | Number of sectors for physics check |
| `rpm` | `float` | `0.0` | RPM for physics check |
| `feature_names` | `list[str] \| None` | `None` | Feature names for SHAP labeling |
| `uncertainty` | `dict \| None` | `None` | Uncertainty dict from `predict_with_uncertainty()` |

**Returns:** dict with keys:
- `"predicted_values"` -- Predicted ND, whirl, amplitude, velocity.
- `"confidence"` -- Confidence score.
- `"physics_check"` -- Physics consistency results.
- `"shap_values"` -- SHAP attributions (None if unavailable).
- `"confidence_interval"` -- 95% CI from uncertainty.
- `"anomaly_flag"` -- True if flagged by physics or uncertainty.
- `"explanation_text"` -- Human-readable summary.

---

## See also

- [ML API](ml.md) -- Model training and the complexity ladder
- [Signals API](signals.md) -- Sensor arrays and signal generation
- [Core API](core.md) -- Mesh, Material, Solver
- [Data API](data.md) -- Dataset export
- [Analysis API](analysis.md) -- Campbell and ZZENF diagrams

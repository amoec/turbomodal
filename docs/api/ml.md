# turbomodal ML API Reference

This document covers the machine learning pipeline for modal identification
(`turbomodal.ml`). The package implements an iterative complexity ladder
(Tiers 1-6) for four sub-tasks:

1. **Mode Detection** -- Multi-label classification of (ND, NC) pairs
2. **Whirl Classification** -- Per-mode binary classification (FW/BW)
3. **Amplitude Estimation** -- Regression for peak amplitude
4. **Propagation Velocity** -- Regression for wave velocity (m/s)

---

## turbomodal.ml.features -- Feature Extraction

### FeatureConfig

Configuration for feature extraction from sensor signals.

```python
@dataclass
class FeatureConfig:
    fft_size: int = 2048
    hop_size: int = 512
    window: str = "hann"
    feature_type: str = "spectrogram"
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 0.0
    max_engine_order: int = 48
    rpm: float = 0.0
    sensor_angles: Optional[np.ndarray] = None
    include_cross_spectra: bool = False
    coherence_threshold: float = 0.5
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fft_size` | `int` | `2048` | FFT window size |
| `hop_size` | `int` | `512` | STFT hop size |
| `window` | `str` | `"hann"` | Window function name |
| `feature_type` | `str` | `"spectrogram"` | Feature extraction type: `"spectrogram"`, `"mel"`, `"order_tracking"`, `"twd"`, `"physics"` |
| `n_mels` | `int` | `128` | Number of mel bands (for `"mel"` type) |
| `f_min` | `float` | `0.0` | Minimum frequency for mel filterbank |
| `f_max` | `float` | `0.0` | Maximum frequency for mel filterbank (0 = Nyquist) |
| `max_engine_order` | `int` | `48` | Maximum engine order (for `"order_tracking"` type) |
| `rpm` | `float` | `0.0` | Rotational speed in RPM (required for order tracking) |
| `sensor_angles` | `ndarray \| None` | `None` | Circumferential sensor positions in radians (required for `"twd"` type) |
| `include_cross_spectra` | `bool` | `False` | Append cross-spectral density features |
| `coherence_threshold` | `float` | `0.5` | Coherence threshold for cross-spectral features |
| `blade_alone_frequencies` | `ndarray \| None` | `None` | Blade-alone natural frequencies in Hz (for `"physics"` type) |
| `centrifugal_alpha` | `float` | `0.0` | Centrifugal stiffening coefficient |
| `reference_temperature` | `float` | `293.0` | Reference temperature in K |
| `temperature` | `float` | `293.0` | Current temperature in K |
| `youngs_modulus_ratio_fn` | `callable \| None` | `None` | Callable `T -> E(T)/E_ref` for temperature correction |

### extract_features

Extract features from time-domain sensor signals.

```python
def extract_features(
    signals: np.ndarray,
    sample_rate: float,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signals` | `ndarray` | required | `(n_sensors, n_samples)` or `(n_samples,)` |
| `sample_rate` | `float` | required | Sampling rate in Hz |
| `config` | `FeatureConfig` | `FeatureConfig()` | Feature configuration |

**Returns:** `ndarray (n_features,)` -- 1-D feature vector.

Feature vector length depends on `feature_type`:
- `"spectrogram"` : `n_sensors * n_freq_bins`
- `"mel"` : `n_sensors * n_mels`
- `"order_tracking"` : `n_sensors * max_engine_order * 2` (real + imaginary)
- `"twd"` : `2 * n_freq * max_nd` (forward + backward magnitudes)

If `include_cross_spectra=True`, cross-spectral density features for all
sensor pairs are appended.

**Raises:** `ValueError` for unknown `feature_type` or missing parameters.

### compute_order_spectrum

Compute order spectrum from a single-channel time-domain signal.

```python
def compute_order_spectrum(
    signal: np.ndarray,
    sample_rate: float,
    rpm: float,
    max_order: int = 48,
) -> tuple[np.ndarray, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `ndarray (n_samples,)` | required | Single-channel signal |
| `sample_rate` | `float` | required | Sampling rate in Hz |
| `rpm` | `float` | required | Rotational speed in RPM (must be positive) |
| `max_order` | `int` | `48` | Maximum engine order to extract |

**Returns:**

- `orders` : `ndarray (max_order,)` -- Integer engine orders 1 through `max_order`.
- `amplitudes` : `ndarray (max_order,)` complex128 -- Complex amplitude at each
  order, scaled by `2/N`.

Extracts the complex amplitude at each integer engine order by locating the
nearest FFT bin to `f_n = n * rpm / 60`.

**Raises:** `ValueError` if `rpm <= 0` or signal is empty.

### traveling_wave_decomposition

Traveling wave decomposition via spatial DFT.

```python
def traveling_wave_decomposition(
    signals: np.ndarray,
    sensor_angles: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `signals` | `ndarray (n_sensors, n_samples)` | Circumferentially distributed sensor signals |
| `sensor_angles` | `ndarray (n_sensors,)` | Angular positions in radians |
| `frequencies` | `ndarray (n_freq,)` | Target frequencies (Hz) |
| `sample_rate` | `float` | Sampling rate in Hz |

**Returns:**

- `forward` : `ndarray (n_freq, max_nd)` complex128 -- Forward (co-rotating)
  traveling wave amplitudes.
- `backward` : `ndarray (n_freq, max_nd)` complex128 -- Backward
  (counter-rotating) traveling wave amplitudes.

Where `max_nd = n_sensors // 2 + 1`.

### build_feature_matrix

Build a feature matrix and label arrays from an HDF5 modal dataset.

```python
def build_feature_matrix(
    dataset_path: str,
    sensor_array,
    signal_config: SignalGenerationConfig,
    feature_config: FeatureConfig = FeatureConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | `str` | required | Path to HDF5 file from `export_modal_results` |
| `sensor_array` | `VirtualSensorArray` | required | Sensor array for signal synthesis |
| `signal_config` | `SignalGenerationConfig` | required | Signal synthesis configuration |
| `feature_config` | `FeatureConfig` | `FeatureConfig()` | Feature extraction configuration |

**Returns:**

- `X` : `ndarray (n_total_samples, n_features)` -- Feature matrix.
- `y` : dict of label arrays with keys:
  - `"nodal_diameter"` : int array
  - `"nodal_circle"` : int array (always 0)
  - `"whirl_direction"` : int array
  - `"frequency"` : float array
  - `"wave_velocity"` : float array (computed as `2*pi*f/ND` for ND > 0)

---

## turbomodal.ml.pipeline -- Training Pipeline

### TrainingConfig

Configuration for ML model training and the complexity ladder.

```python
@dataclass
class TrainingConfig:
    max_tier: int = 6
    performance_gap_threshold: float = 0.02
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.15
    test_split: float = 0.15
    cv_folds: int = 5
    split_by_condition: bool = True
    use_optuna: bool = True
    optuna_trials: int = 50
    output_dir: str = "ml_output"
    experiment_name: str = "turbomodal_mode_id"
    device: str = "auto"
    mode_detection_f1_min: float = 0.92
    whirl_accuracy_min: float = 0.95
    amplitude_mape_max: float = 0.08
    velocity_r2_min: float = 0.93
    independent_subtasks: bool = False
    ood_fraction: float = 0.0
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tier` | `int` | `6` | Maximum complexity tier to try |
| `performance_gap_threshold` | `float` | `0.02` | Min improvement to justify next tier |
| `batch_size` | `int` | `32` | Training batch size |
| `epochs` | `int` | `100` | Maximum training epochs |
| `learning_rate` | `float` | `1e-3` | Learning rate (Adam optimizer) |
| `weight_decay` | `float` | `1e-4` | L2 regularization weight |
| `validation_split` | `float` | `0.15` | Fraction held for validation |
| `test_split` | `float` | `0.15` | Fraction held for testing |
| `cv_folds` | `int` | `5` | Cross-validation folds |
| `split_by_condition` | `bool` | `True` | Split by operating condition, not by sample |
| `use_optuna` | `bool` | `True` | Enable Optuna hyperparameter optimization |
| `optuna_trials` | `int` | `50` | Number of Optuna trials |
| `output_dir` | `str` | `"ml_output"` | Output directory |
| `experiment_name` | `str` | `"turbomodal_mode_id"` | MLflow experiment name |
| `device` | `str` | `"auto"` | Compute device: `"cpu"`, `"cuda"`, `"mps"`, `"auto"` |
| `mode_detection_f1_min` | `float` | `0.92` | Target: mode detection F1 (macro) |
| `whirl_accuracy_min` | `float` | `0.95` | Target: whirl classification accuracy |
| `amplitude_mape_max` | `float` | `0.08` | Target: amplitude MAPE upper bound |
| `velocity_r2_min` | `float` | `0.93` | Target: velocity R-squared lower bound |
| `independent_subtasks` | `bool` | `False` | Run complexity ladder independently per sub-task |
| `ood_fraction` | `float` | `0.0` | Fraction of extreme conditions held out as OOD test set |

### ModeIDModel (Protocol)

Protocol defining the interface for all mode identification models.

```python
@runtime_checkable
class ModeIDModel(Protocol):
    def train(self, X: np.ndarray, y: dict[str, np.ndarray],
              config: TrainingConfig) -> dict[str, float]: ...

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]: ...

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**`train()` returns:** dict of training metrics (e.g. `mode_accuracy`,
`whirl_accuracy`, `amplitude_r2`, `velocity_r2`).

**`predict()` returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"nodal_diameter"` | `int64 (N,)` | Predicted nodal diameters |
| `"nodal_circle"` | `int64 (N,)` | Predicted nodal circles |
| `"frequency"` | `float64 (N,)` | Predicted frequencies |
| `"whirl_direction"` | `int64 (N,)` | -1 (BW), 0 (standing), +1 (FW) |
| `"amplitude"` | `float64 (N,)` | Predicted amplitudes |
| `"wave_velocity"` | `float64 (N,)` | Predicted wave velocities |
| `"confidence"` | `float64 (N,)` | Prediction confidence [0, 1] |

### train_mode_id_model

Train a mode identification model via the complexity ladder.

```python
def train_mode_id_model(
    dataset_path: str | None = None,
    config: TrainingConfig = TrainingConfig(),
    *,
    X: np.ndarray | None = None,
    y: dict[str, np.ndarray] | None = None,
    condition_ids: np.ndarray | None = None,
) -> tuple[object, dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | `str \| None` | `None` | Path to `.npz` feature file |
| `config` | `TrainingConfig` | `TrainingConfig()` | Training configuration |
| `X` | `ndarray \| None` | `None` | Pre-built feature matrix `(N, n_features)` |
| `y` | `dict \| None` | `None` | Pre-built label dict |
| `condition_ids` | `ndarray \| None` | `None` | Condition IDs for grouped splitting |

Either `(X, y)` or `dataset_path` must be provided.

**Returns:** `(best_model, report)` where report contains:
- `"best_tier"` : int -- Tier number of the best model
- `"val_metrics"` : dict -- Validation metrics
- `"test_metrics"` : dict -- Test-set metrics
- `"tier_history"` : list of dicts with per-tier training details

The ladder trains tiers 1 through `config.max_tier` in order, stopping when:
- All four performance targets are met, or
- Improvement over the previous tier falls below `performance_gap_threshold`.

### predict_mode_id

Run mode identification on raw sensor signals.

```python
def predict_mode_id(
    model: ModeIDModel,
    signals: np.ndarray,
    sample_rate: float,
    rpm: float = 0.0,
) -> dict[str, np.ndarray]
```

Extracts features via `extract_features` then calls `model.predict`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModeIDModel` | required | Trained model |
| `signals` | `ndarray` | required | Raw sensor signals |
| `sample_rate` | `float` | required | Sampling rate in Hz |
| `rpm` | `float` | `0.0` | Rotational speed |

**Returns:** dict from `model.predict()`.

### evaluate_model

Evaluate a model on all six sub-task metrics.

```python
def evaluate_model(
    model: ModeIDModel,
    X_test: np.ndarray,
    y_test: dict[str, np.ndarray],
) -> dict[str, float]
```

**Returns:** dict with keys:
- `"mode_detection_f1"` -- Macro F1 score for mode detection
- `"whirl_accuracy"` -- Balanced accuracy for whirl classification
- `"amplitude_mape"` -- Mean Absolute Percentage Error for amplitude
- `"amplitude_r2"` -- R-squared for amplitude
- `"velocity_rmse"` -- RMSE for wave velocity
- `"velocity_r2"` -- R-squared for wave velocity
- `"ece"` -- Expected Calibration Error (15-bin confidence histogram)
- `"inference_latency_mean_ms"` -- Mean single-sample inference latency in ms
- `"inference_latency_p95_ms"` -- 95th-percentile inference latency in ms

---

## turbomodal.ml.models -- Model Implementations

### TIER_MODELS Registry

```python
TIER_MODELS: dict[int, type] = {
    1: LinearModeIDModel,
    2: TreeModeIDModel,
    3: SVMModeIDModel,
    4: ShallowNNModeIDModel,
    5: CNNModeIDModel,
    6: TemporalModeIDModel,
}
```

### Tier 1: LinearModeIDModel

Logistic Regression + Ridge regression.

```python
class LinearModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Uses four independent scikit-learn estimators:
- `LogisticRegression` for mode classification (encoded ND*100+NC labels, OVR)
- `Ridge` or `Lasso` regression for amplitude and velocity

**Parameter:**
- `variant` : `str` -- `"ridge"` (default) or `"lasso"`. Lasso uses L1
  regularization for sparser coefficients.

- `LogisticRegression` for whirl direction classification
- `Ridge` regression for amplitude
- `Ridge` regression for wave velocity

**When to use:** Baseline model with full interpretability via coefficient
weights. Best for small datasets or when interpretability is paramount.

**Persistence:** `joblib` format.

### Tier 2: TreeModeIDModel

Random Forest / XGBoost ensemble.

```python
class TreeModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Three-level fallback: tries `LGBMClassifier`/`LGBMRegressor` (LightGBM)
first, falls back to `XGBClassifier`/`XGBRegressor` (XGBoost), then to
scikit-learn `RandomForestClassifier`/`RandomForestRegressor`.

**Property:**
- `feature_importances_` : `ndarray` -- Average feature importances across
  all four estimators (available after training).

**When to use:** Good balance of accuracy and interpretability. Handles
non-linear relationships and provides feature importance rankings.

**Persistence:** `joblib` format.

### Tier 3: SVMModeIDModel

SVM with RBF kernel.

```python
class SVMModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Uses `SVC(kernel="rbf", probability=True)` for classification and
`SVR(kernel="rbf")` for regression. Features are internally scaled via
`StandardScaler`.

**When to use:** Effective with moderate-sized datasets. Good for problems
with clear margins between mode families in feature space.

**Persistence:** `joblib` format.

### Tier 4: ShallowNNModeIDModel

Shallow multi-task neural network (PyTorch).

```python
class ShallowNNModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Architecture: Two hidden layers (128 -> 64, ReLU) with four task-specific
linear heads. Features are `StandardScaler`-normalized. Training uses Adam
optimizer with early stopping (patience = 10).

**When to use:** When linear/tree models plateau. Medium interpretability
via gradient-based attribution.

**Persistence:** PyTorch `state_dict` + scaler parameters.

### Tier 5: CNNModeIDModel

1-D CNN on spectral inputs (PyTorch).

**Parameter:**
- `variant` : `str` -- `"cnn"` (default) or `"resnet"`. The ResNet variant
  uses two residual blocks with skip connections for deeper feature extraction.

```python
class CNNModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Architecture:
- Conv1d(n_channels, 32, kernel=7) -> BN -> ReLU
- Conv1d(32, 64, kernel=5) -> BN -> ReLU
- AdaptiveAvgPool1d(1)
- Four task-specific linear heads

Input features are reshaped from `(batch, n_features)` to
`(batch, n_channels, n_freq_bins)`.

**Attribute:**
- `last_conv_activations` -- Activations from the last convolutional layer
  (for Grad-CAM attribution after a forward pass).

**When to use:** When spectral patterns contain discriminative spatial
information not captured by flattened features. Supports Grad-CAM
explainability.

**Persistence:** PyTorch `state_dict` + scaler/shape parameters.

### Tier 6: TemporalModeIDModel

Temporal CNN + Bidirectional LSTM (PyTorch).

**Parameter:**
- `variant` : `str` -- `"lstm"` (default) or `"transformer"`. The
  Transformer variant uses a 2-layer encoder with 4 attention heads and
  sinusoidal positional encoding.

```python
class TemporalModeIDModel:
    def __init__(self) -> None: ...
    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Architecture:
- Conv1d(n_channels, 32, kernel=15) -> BN -> ReLU
- Conv1d(32, 64, kernel=7) -> BN -> ReLU
- Bidirectional LSTM(64 -> 32, last timestep)
- Linear(64, 64) -> ReLU
- Four task-specific linear heads

**When to use:** For temporal sequence modeling of transient events or
run-up/run-down data. Highest complexity, lowest interpretability.
Use SHAP or attention-based methods for explanation.

**Persistence:** PyTorch `state_dict` + scaler/shape parameters.

### CompositeModel

Wraps four independent sub-task models into a single predict interface.

```python
class CompositeModel:
    def __init__(
        self,
        mode_model: ModeIDModel,
        whirl_model: ModeIDModel,
        amp_model: ModeIDModel,
        vel_model: ModeIDModel,
        subtask_tiers: dict[str, int] | None = None,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

Used when `config.independent_subtasks=True` causes the complexity ladder to
run independently per sub-task, potentially selecting different tiers for
each. The `subtask_tiers` dict records which tier was selected for each task.

### DeepEnsemble

Ensemble of independently trained models for uncertainty estimation.

```python
class DeepEnsemble:
    def __init__(self, n_members: int = 5, tier: int = 4) -> None: ...

    def train(self, X, y, config) -> dict[str, float]: ...
    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**Parameters:**
- `n_members` : `int` -- Number of ensemble members (default 5).
- `tier` : `int` -- Complexity tier for each member (4, 5, or 6).

`predict()` returns standard keys plus `amplitude_epistemic_var` and
`velocity_epistemic_var` (variance across ensemble members).

### mc_dropout_predict

```python
def mc_dropout_predict(
    model: Any,
    X: np.ndarray,
    n_forward_passes: int = 30,
) -> dict[str, np.ndarray]
```

Run MC Dropout inference for epistemic uncertainty estimation. Enables
dropout at inference time, runs `n_forward_passes` stochastic forward
passes, and returns mean predictions plus `amplitude_epistemic_var`,
`velocity_epistemic_var`, and `mode_entropy`.

Requires a PyTorch-based model (Tier 4-6).

### predict_with_uncertainty

```python
def predict_with_uncertainty(
    model: Any,
    X: np.ndarray,
    method: str = "mc_dropout",
    n_forward_passes: int = 30,
) -> dict[str, np.ndarray]
```

Unified entry point for uncertainty-aware prediction. Returns standard
prediction keys plus aleatoric/epistemic/total variance decomposition:

| Key | Description |
|-----|-------------|
| `amplitude_aleatoric_var` | Data noise variance (heteroscedastic) |
| `amplitude_epistemic_var` | Model uncertainty variance |
| `amplitude_total_var` | Aleatoric + epistemic |
| `velocity_aleatoric_var` | Data noise variance |
| `velocity_epistemic_var` | Model uncertainty variance |
| `velocity_total_var` | Aleatoric + epistemic |
| `mode_entropy` | Classification entropy (MC Dropout only) |

**Parameters:**
- `method` : `str` -- `"mc_dropout"` or `"deep_ensemble"`.
- `n_forward_passes` : `int` -- Number of stochastic passes (MC Dropout only).

---

## MLflow Integration

The training pipeline integrates with MLflow for experiment tracking.
If `mlflow` is installed, `train_mode_id_model` will:

1. Create an experiment named `config.experiment_name` (default:
   `"turbomodal_mode_id"`).
2. Start a parent run `"complexity_ladder"` containing nested runs for each tier.
3. Log parameters (`tier`, `n_train`) and validation metrics per tier.
4. Log training time and test metrics for the best model.

If `mlflow` is not installed, the pipeline runs without error using a silent
no-op proxy -- no tracking data is recorded.

---

## Example Usage

```python
import numpy as np
from turbomodal.ml import (
    FeatureConfig, TrainingConfig,
    build_feature_matrix, train_mode_id_model, predict_mode_id,
)
from turbomodal.signal_gen import SignalGenerationConfig

# 1. Build features from an HDF5 dataset
signal_config = SignalGenerationConfig(sample_rate=100000, duration=0.5)
feature_config = FeatureConfig(feature_type="spectrogram", fft_size=1024)

X, y = build_feature_matrix(
    "dataset.h5", sensor_array, signal_config, feature_config
)

# 2. Train via complexity ladder
config = TrainingConfig(max_tier=4, epochs=50, device="auto")
best_model, report = train_mode_id_model(X=X, y=y, config=config)

print(f"Best tier: {report['best_tier']}")
print(f"Test F1:   {report['test_metrics']['mode_detection_f1']:.3f}")

# 3. Inference on new signals
new_signals = np.random.randn(8, 100000)
predictions = predict_mode_id(best_model, new_signals, sample_rate=100000)
print(predictions["nodal_diameter"])
```

---

## See also

- [Signals API](signals.md) -- Signal generation for training data
- [Data API](data.md) -- HDF5 dataset creation
- [Optimization API](optimization.md) -- SHAP values, Grad-CAM, sensor placement
- [Core API](core.md) -- Solver and mesh

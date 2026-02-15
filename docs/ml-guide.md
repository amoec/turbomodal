# Machine Learning Guide

## Overview

The turbomodal ML pipeline transforms raw blade-tip timing or casing
accelerometer signals into calibrated modal identification predictions.
The pipeline consists of three stages:

1. **Feature extraction** -- time-domain signals are converted to spectral,
   mel-scaled, order-tracking, or traveling-wave-decomposition features via
   `turbomodal.ml.features.extract_features`.
2. **Model training** -- a complexity ladder (`turbomodal.ml.pipeline.train_mode_id_model`)
   iterates through six model tiers, stopping at the first that meets
   performance targets or when diminishing returns are detected.
3. **Post-processing** -- physics consistency checks and confidence calibration
   (`turbomodal.optimization.explainability`) validate and refine predictions
   before they are consumed downstream.

All models are multi-task: they simultaneously predict four targets from a
single feature vector:

| Target             | Type           | Key in output dict     |
|--------------------|----------------|------------------------|
| Nodal diameter     | Classification | `nodal_diameter`       |
| Whirl direction    | Classification | `whirl_direction`      |
| Amplitude          | Regression     | `amplitude`            |
| Wave velocity      | Regression     | `wave_velocity`        |

Every model also returns a `confidence` score (from the mode classification
head) and a `nodal_circle` value (decoded alongside nodal diameter via the
`nd * 100 + nc` label encoding scheme in `_encode_mode_labels`).

---

## The Complexity Ladder

### Why a Complexity Ladder?

The complexity ladder embodies the principle of *minimal sufficient
complexity*. Simple models are faster to train, easier to interpret, and less
prone to overfitting. Complex models are only warranted when simpler ones
cannot meet the performance thresholds. Concretely:

- Tier 1 trains in under a second and provides full coefficient-level
  interpretability.
- Tier 6 may take minutes on a GPU and requires gradient-based attribution to
  explain its decisions.

The pipeline trains tiers in ascending order (1 through `config.max_tier`,
default 6) and stops when either:

- All four performance targets are met (`_check_targets`), or
- The composite score improvement over the previous tier falls below
  `config.performance_gap_threshold` (default 0.02).

### Tier Selection Guide

| Tier | Class                   | Best for                                    | Interpretability |
|------|-------------------------|---------------------------------------------|------------------|
| 1    | `LinearModeIDModel`     | Baseline; small datasets; coefficient audit | Full             |
| 2    | `TreeModeIDModel`       | Tabular features; feature-importance needs  | High             |
| 3    | `SVMModeIDModel`        | Moderate-size datasets; nonlinear boundaries| Medium           |
| 4    | `ShallowNNModeIDModel`  | Larger datasets; multi-task joint training  | Medium           |
| 5    | `CNNModeIDModel`        | Raw spectrograms; spatial patterns in freq  | Low (Grad-CAM)   |
| 6    | `TemporalModeIDModel`   | Temporal sequences; transient events        | Low (SHAP)       |

### Tier Details

#### Tier 1 -- LinearModeIDModel

- **Architecture**: four independent scikit-learn estimators.
  - `LogisticRegression(max_iter=1000, C=1.0)` for mode classification
    (encoded `(ND, NC)` labels, OVR strategy by default).
  - `LogisticRegression(max_iter=1000)` for whirl direction.
  - `Ridge(alpha=1.0)` for amplitude.
  - `Ridge(alpha=1.0)` for wave velocity.
- **Strengths**: extremely fast, full interpretability via coefficient weights,
  reliable baseline.
- **Weaknesses**: cannot capture nonlinear feature interactions.
- **Persistence**: `joblib` (`.joblib` files).

**Variant**: `LinearModeIDModel(variant="lasso")` replaces Ridge regression
with Lasso (L1 regularization) for amplitude and velocity estimation,
producing sparser coefficient vectors.

#### Tier 2 -- TreeModeIDModel

- **Architecture**: four independent tree ensemble estimators with a
  three-level fallback: LightGBM → XGBoost → RandomForest. The pipeline
  tries `LGBMClassifier`/`LGBMRegressor` first, falls back to
  `XGBClassifier`/`XGBRegressor`, and finally to scikit-learn
  `RandomForestClassifier`/`RandomForestRegressor`.
- **Strengths**: handles nonlinear interactions, exposes `feature_importances_`
  (averaged across all four estimators), robust to feature scaling.
- **Weaknesses**: larger memory footprint; less interpretable than linear.
- **Persistence**: `joblib`.

#### Tier 3 -- SVMModeIDModel

- **Architecture**: `SVC(kernel="rbf", probability=True)` for classification,
  `SVR(kernel="rbf")` for regression. All inputs scaled via
  `StandardScaler`.
- **Strengths**: effective with moderate sample sizes; probability estimates via
  Platt scaling are built in.
- **Weaknesses**: scales poorly to very large datasets; scaling must be stored
  and applied consistently at inference.
- **Persistence**: `joblib` (includes stored `StandardScaler`).

#### Tier 4 -- ShallowNNModeIDModel

- **Architecture**: PyTorch two-hidden-layer multi-task net.
  - Backbone: `Linear(n_features, 128) -> ReLU -> Linear(128, 64) -> ReLU`.
  - Four heads: `Linear(64, n_mode_classes)`, `Linear(64, 3)` (whirl),
    `Linear(64, 1)` (amplitude), `Linear(64, 1)` (velocity).
- **Optimizer**: `Adam(lr=config.learning_rate, weight_decay=config.weight_decay)`.
- **Loss**: sum of `CrossEntropyLoss` (mode + whirl) and `MSELoss`
  (amplitude + velocity).
- **Early stopping**: patience of 10 epochs on validation loss; restores best
  state dict.
- **Strengths**: joint multi-task learning; moderate training time.
- **Weaknesses**: requires careful feature scaling (`StandardScaler`); less
  interpretable than tree models.
- **Persistence**: `torch.save` (state dict + scaler parameters + label maps).

#### Tier 5 -- CNNModeIDModel

- **Architecture**: PyTorch 1-D CNN (`CNN1DNet`).
  - Features reshaped from `(B, n_features)` to `(B, n_channels, n_freq_bins)`
    via `_infer_channel_shape`.
  - `Conv1d(n_channels, 32, kernel_size=7, padding=3) -> BN -> ReLU ->
    Conv1d(32, 64, kernel_size=5, padding=2) -> BN -> ReLU ->
    AdaptiveAvgPool1d(1)`.
  - Same four heads as Tier 4.
- **Grad-CAM support**: registers a forward hook on the last pooling layer;
  `last_conv_activations` is populated after each forward pass.
- **Strengths**: learns spatial patterns in spectral features automatically;
  supports Grad-CAM attribution.
- **Weaknesses**: requires sufficient data to avoid overfitting; interpretation
  requires Grad-CAM.
- **Persistence**: `torch.save` (state dict + channel shape + scaler + labels).

**Variant**: `CNNModeIDModel(variant="resnet")` replaces the plain CNN
backbone with a 1-D ResNet containing two residual blocks (each with two
`Conv1d` layers and a skip connection). The deeper architecture can capture
more complex spectral patterns at the cost of additional parameters.

#### Tier 6 -- TemporalModeIDModel

- **Architecture**: PyTorch Conv + BiLSTM (`TemporalNet`).
  - `Conv1d(n_channels, 32, kernel_size=15, padding=7) -> BN -> ReLU ->
    Conv1d(32, 64, kernel_size=7, padding=3) -> BN -> ReLU`.
  - `LSTM(64, 32, batch_first=True, bidirectional=True)` --
    output dimension is 64 (32 forward + 32 backward).
  - `Linear(64, 64) -> ReLU` then four task heads.
  - Uses last LSTM timestep `h[:, -1, :]` as the summary representation.
- **Strengths**: captures temporal dynamics across the sequence; suitable for
  transient blade events.
- **Weaknesses**: slowest to train; lowest interpretability; most data-hungry.
- **Persistence**: `torch.save`.

**Variant**: `TemporalModeIDModel(variant="transformer")` replaces the
BiLSTM with a Transformer encoder (2 layers, 4 attention heads, d_model=64)
with sinusoidal positional encoding. Global average pooling over the
sequence dimension produces the summary representation. Suitable for long
sequences where self-attention can capture distant dependencies.

### Model Registry

All six model classes are registered in `turbomodal.ml.models.TIER_MODELS`:

```python
from turbomodal.ml.models import TIER_MODELS

TIER_MODELS = {
    1: LinearModeIDModel,
    2: TreeModeIDModel,
    3: SVMModeIDModel,
    4: ShallowNNModeIDModel,
    5: CNNModeIDModel,
    6: TemporalModeIDModel,
}
```

Every class conforms to the `ModeIDModel` protocol defined in
`turbomodal.ml.pipeline`, which requires `train`, `predict`, `save`, and
`load` methods.

---

## Feature Engineering

All feature extraction is handled by `turbomodal.ml.features`. The central
entry point is `extract_features`, configured via a `FeatureConfig` dataclass.

### FeatureConfig Parameters

| Parameter             | Type              | Default       | Description                                        |
|-----------------------|-------------------|---------------|----------------------------------------------------|
| `fft_size`            | `int`             | `2048`        | FFT window length (number of samples per segment)  |
| `hop_size`            | `int`             | `512`         | Hop length between consecutive STFT frames         |
| `window`              | `str`             | `"hann"`      | Window function name (passed to `scipy.signal.stft`)|
| `feature_type`        | `str`             | `"spectrogram"`| One of `"spectrogram"`, `"mel"`, `"order_tracking"`, `"twd"`, `"physics"` |
| `n_mels`              | `int`             | `128`         | Number of mel bands (when `feature_type="mel"`)    |
| `f_min`               | `float`           | `0.0`         | Minimum frequency for mel filterbank (Hz)          |
| `f_max`               | `float`           | `0.0`         | Maximum frequency for mel filterbank (0 = Nyquist) |
| `max_engine_order`    | `int`             | `48`          | Highest engine order to extract                    |
| `rpm`                 | `float`           | `0.0`         | Rotational speed for order tracking (RPM)          |
| `sensor_angles`       | `ndarray or None` | `None`        | Circumferential sensor positions in radians (TWD)  |
| `include_cross_spectra` | `bool`          | `False`       | Append cross-spectral density features             |
| `coherence_threshold` | `float`           | `0.5`         | Zero CSD bins where coherence falls below this     |
| `blade_alone_frequencies` | `ndarray or None` | `None`       | Blade-alone natural frequencies in Hz (physics features) |
| `centrifugal_alpha`   | `float`           | `0.0`         | Centrifugal stiffening coefficient                     |
| `reference_temperature` | `float`         | `293.0`       | Reference temperature in K                             |
| `temperature`         | `float`           | `293.0`       | Current temperature in K                               |
| `youngs_modulus_ratio_fn` | `callable or None` | `None`   | Callable `T -> E(T)/E_ref` for temperature correction  |

### Spectrogram Features (STFT)

When `feature_type="spectrogram"`, `extract_features` computes the short-time
Fourier transform of each sensor channel via `scipy.signal.stft`, takes the
magnitude, and averages over time frames:

```
for each sensor:
    _, _, Zxx = stft(signal, fs, window, nperseg=fft_size,
                     noverlap=fft_size - hop_size)
    mean_mag = |Zxx|.mean(axis=1)   # (n_freq_bins,)
feature_vec = concatenate(all mean_mag)  # (n_sensors * n_freq_bins,)
```

The number of frequency bins is `fft_size // 2 + 1`.

### Mel Spectrogram

When `feature_type="mel"`, the same STFT is computed, but the magnitude
spectrum is projected onto a mel-scale triangular filterbank via
`_build_mel_filterbank(n_mels, n_fft_bins, sample_rate, f_min, f_max)`:

```
mel_spec = mel_fb @ mean_mag   # (n_mels,)
```

The mel scale better represents human-perceived frequency spacing and
compresses the feature dimension from `n_freq_bins` to `n_mels`.
Output shape: `(n_sensors * n_mels,)`.

### Order Tracking

`compute_order_spectrum(signal, sample_rate, rpm, max_order=48)` extracts
the complex amplitude at each integer engine order by locating the nearest
FFT bin to `f_n = n * rpm / 60`:

```python
orders, amplitudes = compute_order_spectrum(signal, fs, rpm, max_order=10)
# orders: array([1, 2, ..., 10])
# amplitudes: complex128 array, scaled by 2/N
```

When `feature_type="order_tracking"`, the real and imaginary parts of these
amplitudes are concatenated for each sensor. Output shape:
`(n_sensors * max_engine_order * 2,)`.

Raises `ValueError` if `rpm <= 0`.

### Traveling Wave Decomposition (TWD)

`traveling_wave_decomposition(signals, sensor_angles, frequencies, sample_rate)`
performs a spatial DFT across circumferentially placed sensors to separate
forward and backward traveling wave components:

```
C_n+(f) = (1/K) * sum_k x_k(f) * exp(-j*n*theta_k)   (forward)
C_n-(f) = (1/K) * sum_k x_k(f) * exp(+j*n*theta_k)   (backward)
```

Returns two complex arrays of shape `(n_freq, max_nd)` where
`max_nd = n_sensors // 2 + 1`.

When `feature_type="twd"`, the magnitudes of forward and backward components
are flattened and concatenated into the feature vector.

### Cross-Spectral Density

When `include_cross_spectra=True`, cross-spectral features are appended to the
base feature vector. For each sensor pair `(i, j)` where `i < j`:

1. The cross-spectral density `Pxy` is computed via `scipy.signal.csd`.
2. The coherence `Cxy` is computed via `scipy.signal.coherence`.
3. CSD magnitude bins where coherence falls below `coherence_threshold` are
   zeroed out.

This requires at least 2 sensors. The number of appended features is
`C(n_sensors, 2) * n_freq_bins`.

### Physics-Informed Features

When `feature_type="physics"`, domain-specific features are computed from
the spectral data combined with physical parameters:

- **Frequency ratios**: ratios of observed peak frequencies to
  `blade_alone_frequencies`, indicating how much the system deviates from
  the isolated-blade case.
- **Centrifugal correction**: applies `f_corrected = f * sqrt(1 + alpha * (rpm/60)^2)`
  to normalize frequencies for rotational speed effects.
- **Temperature correction**: scales frequencies by
  `sqrt(youngs_modulus_ratio_fn(temperature))` when a Young's modulus ratio
  function is provided.

These features encode known turbomachinery physics directly into the feature
vector, improving model accuracy especially at low sample counts where
data-driven features alone may be insufficient.

### build_feature_matrix

`build_feature_matrix(dataset_path, sensor_array, signal_config, feature_config)`
is the end-to-end function for converting an HDF5 modal dataset into a
feature matrix `X` and label dict `y`:

1. Loads modal results via `turbomodal.dataset.load_modal_results`.
2. Synthesizes time-domain signals for each operating condition via
   `turbomodal.signal_gen.generate_signals_for_condition`.
3. Calls `extract_features` on each condition's signals.
4. Creates one sample per `(harmonic, mode)` pair with labels:
   - `nodal_diameter`: from the harmonic index.
   - `nodal_circle`: always 0 (NC identification requires C++ mode shape analysis).
   - `whirl_direction`: from the dataset.
   - `frequency`: eigenvalue in Hz.
   - `wave_velocity`: `2 * pi * f / nd` for `nd > 0`, else `0`.

Returns `(X, y)` where `X` has shape `(n_total_samples, n_features)`.

---

## Training Pipeline

### Data Splitting

The pipeline supports two splitting modes controlled by
`config.split_by_condition` (default `True`):

**Condition-based splitting** (`_condition_based_split`): uses scikit-learn's
`GroupShuffleSplit` to ensure no operating condition appears in more than one
split. This prevents data leakage because samples from the same condition
share the same condition-level feature vector.

The split is performed in two stages:
1. `(train+val)` vs `test` with `test_size=config.test_split` (default 0.15).
2. `train` vs `val` from the remaining portion with an adjusted validation
   fraction `min(val_frac / (1 - test_frac), 0.5)`.

If `GroupShuffleSplit` requires at least 2 unique condition groups for the
validation split; otherwise all non-test data goes to training.

**Random splitting**: a simple index permutation with seed 42, producing
test/val/train segments proportionally.

The target split ratios are **70% train / 15% validation / 15% test**
(`validation_split=0.15`, `test_split=0.15`).

### TrainingConfig Parameters

| Parameter                     | Default  | Description                                       |
|-------------------------------|----------|---------------------------------------------------|
| `max_tier`                    | `6`      | Maximum complexity tier to attempt                |
| `performance_gap_threshold`   | `0.02`   | Minimum composite score improvement to justify next tier |
| `batch_size`                  | `32`     | Mini-batch size (PyTorch tiers)                   |
| `epochs`                      | `100`    | Maximum training epochs (PyTorch tiers)           |
| `learning_rate`               | `1e-3`   | Adam learning rate                                |
| `weight_decay`                | `1e-4`   | L2 regularization weight                          |
| `validation_split`            | `0.15`   | Fraction held out for validation                  |
| `test_split`                  | `0.15`   | Fraction held out for final test                  |
| `cv_folds`                    | `5`      | Cross-validation folds (for Optuna, when used)    |
| `split_by_condition`          | `True`   | Use condition-based GroupShuffleSplit              |
| `use_optuna`                  | `True`   | Enable Optuna hyperparameter optimization         |
| `optuna_trials`               | `50`     | Number of Optuna trials                           |
| `output_dir`                  | `"ml_output"` | Directory for saved artifacts                |
| `experiment_name`             | `"turbomodal_mode_id"` | MLflow experiment name             |
| `device`                      | `"auto"` | Compute device: `"cpu"`, `"cuda"`, `"mps"`, `"auto"` |
| `mode_detection_f1_min`       | `0.92`   | Minimum weighted F1 for mode detection            |
| `whirl_accuracy_min`          | `0.95`   | Minimum accuracy for whirl classification         |
| `amplitude_mape_max`          | `0.08`   | Maximum MAPE for amplitude estimation             |
| `velocity_r2_min`             | `0.93`   | Minimum R-squared for velocity estimation         |

### The Training Loop

`train_mode_id_model` accepts either pre-built `(X, y)` arrays or a path to a
`.npz` file. The main loop:

```python
for tier in range(1, config.max_tier + 1):
    model = TIER_MODELS[tier]()
    train_metrics = model.train(X_train, y_train, config)
    val_metrics = evaluate_model(model, X_val, y_val)
    score = _composite_score(val_metrics, config)

    if _check_targets(val_metrics, config):
        break   # all four targets met
    if tier > 1 and (score - prev_score) < config.performance_gap_threshold:
        break   # diminishing returns
```

The **composite score** combines all four metrics into a single scalar:

```
score = 0.30 * mode_detection_f1
      + 0.20 * whirl_accuracy
      + 0.25 * (1 - amplitude_mape)
      + 0.25 * velocity_r2
```

The function returns `(best_model, report)` where `report` contains:
- `best_tier`: the tier number of the selected model.
- `val_metrics`: validation metrics of the best model.
- `test_metrics`: final test-set evaluation of the best model.
- `tier_history`: list of dicts with per-tier metrics, training time, and
  composite score.

### PyTorch Training Infrastructure

Tiers 4-6 share a common training loop (`_train_pytorch_model`) that:

1. Encodes mode labels via `_encode_mode_labels(nd, nc)` into contiguous class
   indices.
2. Offsets whirl direction by +1 so that `{-1, 0, +1}` becomes `{0, 1, 2}` for
   `CrossEntropyLoss`.
3. Optionally splits into train/validation via `_split_validation`.
4. Trains with Adam, summing four losses:
   `CE(mode) + CE(whirl) + MSE(amplitude) + MSE(velocity)`.
5. Implements early stopping with patience of 10 epochs on validation loss;
   restores the best model state dict.

Device selection (`_get_device`) auto-detects CUDA, then MPS (Apple Silicon),
falling back to CPU.

### MLflow Integration

Experiment tracking is handled by `_MLflowProxy`, which delegates to `mlflow`
when installed and silently no-ops otherwise. During the complexity ladder:

- `set_experiment(config.experiment_name)` is called once.
- A parent run `"complexity_ladder"` wraps nested per-tier runs.
- Each tier logs:
  - Parameters: `tier`, `n_train`.
  - Metrics: all six evaluation metrics plus `train_time_s`.
- After the ladder, final test metrics are logged with a `test_` prefix.

To view results when `mlflow` is installed:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## Model Evaluation

### evaluate_model

`evaluate_model(model, X_test, y_test)` calls `model.predict(X_test)` and
computes six metrics:

| Metric key                  | Computation                                         |
|-----------------------------|-----------------------------------------------------|
| `mode_detection_f1`         | Macro F1 on encoded `(ND, NC)` labels (sklearn `f1_score`) |
| `whirl_accuracy`            | Balanced accuracy on whirl direction (sklearn `balanced_accuracy_score`) |
| `amplitude_mape`            | Mean absolute percentage error, using `max(|y_true|, 1e-8)` as denominator |
| `amplitude_r2`              | R-squared on amplitude (sklearn `r2_score`)         |
| `velocity_rmse`             | Root mean squared error on wave velocity            |
| `velocity_r2`               | R-squared on wave velocity                          |
| `ece`                       | Expected Calibration Error (15-bin confidence histogram) |
| `inference_latency_mean_ms` | Mean single-sample inference time in milliseconds   |
| `inference_latency_p95_ms`  | 95th-percentile inference latency in milliseconds   |

When scikit-learn is unavailable, manual fallback implementations
(`_manual_f1`, `_manual_r2`) are used.

### predict_mode_id

`predict_mode_id(model, signals, sample_rate, rpm=0.0)` is a convenience
function that:

1. Builds a `FeatureConfig(rpm=rpm)` with default STFT parameters.
2. Calls `extract_features(signals, sample_rate, feat_config)`.
3. Reshapes the 1-D feature vector to `(1, n_features)`.
4. Returns `model.predict(features)`.

### Hyperparameter Optimization

When `config.use_optuna=True` (the default), the pipeline runs Optuna TPE
(Tree-structured Parzen Estimator) optimization before training each tier.
Per-tier search spaces are defined in `_suggest_hyperparams`:

| Tier | Parameters | Range |
|------|-----------|-------|
| 1 | `C`, `alpha` | `[1e-3, 100]`, `[1e-4, 10]` (log scale) |
| 2 | `n_estimators`, `max_depth`, `learning_rate` | `[50, 300]`, `[3, 12]`, `[0.01, 0.3]` |
| 3 | `C`, `gamma` | `[1e-2, 100]`, `{"scale", "auto"}` |
| 4-6 | `lr`, `batch_size`, `weight_decay` | `[1e-5, 1e-2]`, `{16, 32, 64}`, `[1e-6, 1e-2]` |

Each trial evaluates the candidate hyperparameters via `cv_folds`-fold
cross-validation (default 5 folds, `GroupKFold` when condition IDs are
available). The number of trials is controlled by `config.optuna_trials`
(default 50). Optuna is an optional dependency; when not installed, default
hyperparameters are used.

### Cross-Validation

The pipeline uses `GroupKFold` (from scikit-learn) when condition IDs are
provided, ensuring no operating condition appears in multiple folds. This
prevents data leakage across folds just as `GroupShuffleSplit` does for the
main train/val/test split. When condition IDs are unavailable or when fewer
unique conditions than folds exist, standard `KFold` with `shuffle=True` is
used instead. The number of folds is set via `config.cv_folds` (default 5).

### CompositeModel

When `config.independent_subtasks=True`, the complexity ladder runs
independently for each of the four sub-tasks (mode classification, whirl
classification, amplitude regression, velocity regression). Each sub-task
may select a different tier. The resulting models are wrapped in a
`CompositeModel` that merges predictions from all four sub-task models into
a single output dict.

This is useful when sub-tasks have different complexity requirements -- for
example, mode classification may need a Tier 4 neural network while wave
velocity regression is well-served by Tier 1 linear regression.

### Out-of-Distribution Evaluation

When `config.ood_fraction > 0` and `condition_params` are provided to
`train_mode_id_model`, the pipeline extracts the most extreme operating
conditions from the training set as an out-of-distribution (OOD) test set.

Extremeness is determined by the maximum absolute z-score across all
parametric dimensions. The top `ood_fraction` conditions (by extremeness)
are removed from training and evaluated separately. The training report
includes `ood_metrics` alongside the standard `test_metrics`.

### Uncertainty Quantification

Three mechanisms provide uncertainty estimates for predictions:

**MC Dropout** (`mc_dropout_predict`): enables dropout at inference time and
runs `n_forward_passes` stochastic forward passes through PyTorch models
(Tiers 4-6). Returns mean predictions plus `amplitude_epistemic_var`,
`velocity_epistemic_var`, and `mode_entropy`.

**Deep Ensembles** (`DeepEnsemble`): trains `n_members` independent models
(default 5) with different random seeds. Predictions are aggregated via
majority vote (classification) and mean (regression). Variance across
members provides epistemic uncertainty.

**Heteroscedastic output heads**: all PyTorch network architectures support
optional `heteroscedastic=True` mode, which adds log-variance output heads
for amplitude and velocity. This provides per-sample aleatoric uncertainty
estimates.

The unified entry point `predict_with_uncertainty(model, X, method)` returns
a dict with all standard prediction keys plus:

| Key | Description |
|-----|-------------|
| `amplitude_aleatoric_var` | Data noise variance (from heteroscedastic heads) |
| `amplitude_epistemic_var` | Model uncertainty variance |
| `amplitude_total_var` | Sum of aleatoric + epistemic |
| `velocity_aleatoric_var` | Data noise variance |
| `velocity_epistemic_var` | Model uncertainty variance |
| `velocity_total_var` | Sum of aleatoric + epistemic |
| `mode_entropy` | Classification entropy (MC Dropout only) |

---

## Explainability

All explainability tools live in `turbomodal.optimization.explainability`.

### SHAP Values

`compute_shap_values(model, signals, feature_names=None)` determines the
appropriate SHAP explainer based on model type:

- **TreeModeIDModel**: uses `shap.TreeExplainer` on each of the four
  estimators (`_mode_clf`, `_whirl_clf`, `_amp_reg`, `_vel_reg`). Multi-class
  SHAP values are reduced via `mean(abs(...))` across classes. Returns shape
  `(n_samples, n_features, 4)` -- one slice per estimator.

- **All other models**: uses `shap.KernelExplainer` with a prediction wrapper
  that stacks `[nodal_diameter, whirl_direction, amplitude, wave_velocity]`
  into a `(n_samples, 4)` output. A background dataset of up to 100 samples
  is randomly selected. Returns shape `(n_samples, n_features, n_outputs)`.

Requires the `shap` package (`pip install -e ".[ml]"`).

### Grad-CAM

`compute_grad_cam(model, signals, target_class, layer_name=None)` computes
Gradient-weighted Class Activation Mapping for CNN-based models only (Tiers 5
and 6). It:

1. Registers forward and backward hooks on the target convolutional layer.
   - For `CNNModeIDModel`: the layer before `AdaptiveAvgPool1d`
     (`features[-3]`).
   - For `TemporalModeIDModel`: the last sub-layer of the `conv` sequential
     (`conv[-1]`).
2. Runs a forward pass, then backpropagates a one-hot gradient for
   `target_class` through the mode classification head.
3. Computes channel weights via global average pooling of gradients.
4. Produces a weighted combination `ReLU(sum(weights * activations))`.
5. Normalizes to `[0, 1]` per sample.

Returns shape `(batch, length)`.

Raises `TypeError` for non-CNN/Temporal models.

### Confidence Calibration

`calibrate_confidence(model, X_val, y_val, method)` wraps a trained model in a
`CalibratedModel` that transforms raw confidence scores via a fitted
calibration function.

| Method        | How it works                                                              | When to use                           |
|---------------|---------------------------------------------------------------------------|---------------------------------------|
| `"platt"`     | Fits a `LogisticRegression` on `(raw_confidence, correctness)`.  Transform: `1 / (1 + exp(-(a*c + b)))`. | General purpose; fast; smooth mapping |
| `"isotonic"`  | Fits `IsotonicRegression(out_of_bounds="clip")` on `(raw_confidence, correctness)`. | Non-parametric; large calibration sets |
| `"temperature"` | Finds optimal temperature `T` by minimizing binary cross-entropy. Transform: `c^(1/T) / (c^(1/T) + (1-c)^(1/T))`. | Neural network outputs; simple tuning |
| `"conformal"` | Computes nonconformity scores on amplitude and velocity residuals. Adds `prediction_interval_lower` and `prediction_interval_upper` keys with 90% coverage. | When prediction intervals are needed  |

The `CalibratedModel` wrapper preserves the `ModeIDModel` protocol: it
delegates `train`, `save`, and `load` to the base model and only transforms
the `"confidence"` key in `predict` output.

### Physics Consistency Check

`physics_consistency_check(predictions, num_sectors, rpm=0.0, blade_radius=0.3)`
validates predictions against five physical constraints (see the
[Validation Criteria](validation.md) document for details). Returns a dict
with:

- `is_consistent`: `(N,)` bool array, True when all checks pass.
- `violations`: `(N,)` list of string descriptions for each violation.
- `consistency_score`: `(N,)` float in `[0, 1]`, fraction of applicable checks
  passed.
- `anomaly_flag`: `(N,)` bool, True when `consistency_score < 0.8`.

### Model Selection Report

`generate_model_selection_report(training_report)` in
`turbomodal.optimization.explainability` takes the report dict returned by
`train_mode_id_model` and produces a structured summary including:

- `summary`: human-readable text describing the selected tier and its metrics.
- `per_tier_metrics`: dict mapping tier number to validation metrics.
- `gap_analysis`: list of score deltas between consecutive tiers.
- `selected_tier`: the tier number of the best model.

### Explanation Cards

`generate_explanation_card(model, X_single, predictions, ...)` produces a
per-prediction explanation dict containing:

- `predicted_values`: ND, whirl direction, amplitude, wave velocity.
- `confidence`: calibrated confidence score.
- `physics_check`: results of `physics_consistency_check` for this sample.
- `shap_values`: SHAP attributions (best-effort; None if SHAP unavailable).
- `confidence_interval`: 95% CI from uncertainty if available.
- `anomaly_flag`: True if physics check or epistemic uncertainty flags the sample.
- `explanation_text`: human-readable summary of the prediction.

---

## Best Practices

1. **Start with Tier 1.** Always establish a baseline with `LinearModeIDModel`
   before escalating. Many turbomachinery problems are well-served by linear
   models on well-engineered features.

2. **Use condition-based splitting.** Set `config.split_by_condition=True`
   (the default) to prevent data leakage. Samples from the same operating
   condition share the same underlying signal, so they must not appear in
   both training and test sets.

3. **Always run `physics_consistency_check` on predictions.** This catches
   physically impossible outputs (negative frequencies, out-of-range nodal
   diameters, inconsistent velocities) that the ML model may produce,
   especially on out-of-distribution inputs.

4. **Calibrate confidence before deploying.** Raw model confidence scores are
   often miscalibrated. Use `calibrate_confidence` with a held-out validation
   set. For downstream systems that need prediction intervals (e.g., fatigue
   life estimates), use the `"conformal"` method.

5. **Monitor feature importances.** For Tier 2 models, inspect
   `model.feature_importances_` to verify that the model is attending to
   physically meaningful features. For Tiers 5-6, use Grad-CAM or SHAP.

6. **Match feature configuration at training and inference time.** The
   `FeatureConfig` used during `build_feature_matrix` or manual feature
   extraction must be identical to the one used at inference via
   `predict_mode_id`. Mismatched `fft_size` or `feature_type` will produce
   wrong-shaped feature vectors.

7. **Check the `tier_history` in the training report.** This reveals whether
   additional complexity actually improved performance, which tiers failed to
   train, and whether the performance gap threshold was triggered.

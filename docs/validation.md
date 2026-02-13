# Validation Criteria

## Overview

This document defines the acceptance thresholds, physics consistency rules,
composite scoring formula, and testing strategy that govern turbomodal's modal
identification pipeline. Every model produced by the complexity ladder is
evaluated against these criteria before it is selected or promoted to the next
tier.

---

## Performance Thresholds

The pipeline defines four primary performance targets in `TrainingConfig`
(see `turbomodal.ml.pipeline`). All four must be satisfied simultaneously for
`_check_targets` to return `True` and halt the complexity ladder.

### Mode Detection (Nodal Diameter)

| Property  | Value    |
|-----------|----------|
| Metric    | Weighted (macro) F1 score |
| Threshold | >= 0.92  |
| Config key | `mode_detection_f1_min` |

Mode labels are encoded as `nd * 100 + nc` via `_encode_mode_labels`, so the
F1 score covers the joint `(ND, NC)` classification. The macro average gives
equal weight to each class regardless of support, which is important when
certain nodal diameters are rare in the dataset.

**Why this threshold**: Correct nodal diameter identification is the
foundation of resonance avoidance in bladed disk design. Misidentifying the
ND means misplacing the mode on the Campbell diagram, which can lead to
undetected crossings with engine order excitations.

Computed in `evaluate_model` via:
```python
from sklearn.metrics import f1_score
mode_f1 = f1_score(true_mode, pred_mode, average="macro", zero_division=0)
```

### Whirl Classification

| Property  | Value    |
|-----------|----------|
| Metric    | Balanced accuracy |
| Threshold | >= 0.95  |
| Config key | `whirl_accuracy_min` |

Whirl direction is encoded as `{-1, 0, +1}` corresponding to backward whirl,
standing wave, and forward whirl respectively.

**Why this threshold**: Forward and backward whirl modes have different
frequencies in the rotating frame. Confusing them distorts the Campbell diagram
and can mask dangerous resonance crossings. The high threshold reflects the
safety-critical nature of this classification.

Computed in `evaluate_model` via:
```python
from sklearn.metrics import balanced_accuracy_score
whirl_acc = balanced_accuracy_score(true_whirl, pred_whirl)
```

### Amplitude Estimation

| Property  | Value    |
|-----------|----------|
| Metric    | Mean Absolute Percentage Error (MAPE) |
| Threshold | <= 8% (0.08) |
| Config key | `amplitude_mape_max` |

MAPE is computed with a floor on the denominator to avoid division by zero:
```python
denom = max(|y_true|, 1e-8)
amplitude_mape = mean(|y_true - y_pred| / denom)
```

**Why this threshold**: Amplitude directly affects blade stress levels and
fatigue life predictions. An 8% tolerance balances measurement uncertainty
(BTT systems typically have 2-5% noise) against the engineering need for
accurate forced-response estimation.

### Velocity Estimation

| Property  | Value    |
|-----------|----------|
| Metric    | R-squared (coefficient of determination) |
| Threshold | >= 0.93  |
| Config key | `velocity_r2_min` |

Wave velocity is the circumferential phase speed of the traveling wave
pattern, computed as `v = 2 * pi * f * R / ND` for `ND > 0`.

**Why this threshold**: Wave velocity serves as a cross-check for modal
identification consistency. If the model correctly identifies frequency and
nodal diameter, the velocity should follow the physical relationship. An R²
of 0.93 allows for measurement noise while ensuring the model has learned the
underlying physics.

Computed in `evaluate_model` via:
```python
from sklearn.metrics import r2_score
velocity_r2 = r2_score(true_vel, pred_vel)
```

---

## Physics Consistency Checks

`physics_consistency_check` in `turbomodal.optimization.explainability`
validates predictions against five physical constraints. These checks run
independently of ML performance metrics and serve as a safety net for
detecting physically impossible outputs.

### Check 1: Positive Frequency

```
frequency > 0
```

Natural frequencies of a structural system are always positive. A zero or
negative predicted frequency indicates a model failure or a degenerate input.
Applied to every sample.

### Check 2: Valid Nodal Diameter Range

```
0 <= ND <= num_sectors // 2
```

For a bladed disk with `N` sectors, the maximum independent nodal diameter is
`N/2` (integer division). Predictions outside this range are physically
meaningless because they would alias to a lower harmonic.

### Check 3: Valid Whirl Direction

```
whirl_direction in {-1, 0, 1}
```

Only three whirl states are physically meaningful: backward (-1), standing
wave (0), and forward (+1). Any other value indicates a model or encoding
error.

### Check 4: Whirl Ordering

```
For each ND group where rpm > 0:
    min(forward whirl frequencies) >= max(backward whirl frequencies)
```

In the rotating frame, Coriolis effects split the forward and backward
traveling wave frequencies. For a given nodal diameter, the forward whirl
frequency should always be greater than or equal to the backward whirl
frequency. This check is only applied when `rpm > 0` (stationary analysis has
no whirl splitting).

The check groups predictions by nodal diameter, finds the minimum forward
frequency and maximum backward frequency within each group, and flags
violations where forward < backward.

### Check 5: Wave Velocity Consistency

```
When ND > 0 and frequency > 0:
    v_expected = 2 * pi * f * R / ND
    |v_predicted - v_expected| / v_expected < 0.50
```

The circumferential wave velocity must be consistent with the predicted
frequency and nodal diameter. The tolerance is 50% relative error, which
accounts for the fact that the predicted velocity and predicted ND/frequency
may all contain independent errors.

### Output Format

`physics_consistency_check` returns a dict with four keys:

| Key                | Shape    | Description                                      |
|--------------------|----------|--------------------------------------------------|
| `is_consistent`    | `(N,)`   | Boolean, True if all applicable checks pass      |
| `violations`       | `(N,)`   | List of string descriptions per sample           |
| `consistency_score`| `(N,)`   | Float in [0, 1], fraction of checks passed       |
| `anomaly_flag`     | `(N,)`   | Boolean, True when `consistency_score < 0.8`     |

---

## Composite Score

The complexity ladder uses a weighted composite score to compare tiers and
detect diminishing returns. Implemented in `_composite_score`:

```
score = 0.30 * mode_detection_f1
      + 0.20 * whirl_accuracy
      + 0.25 * (1.0 - amplitude_mape)
      + 0.25 * velocity_r2
```

| Component              | Weight | Rationale                                     |
|------------------------|--------|-----------------------------------------------|
| `mode_detection_f1`    | 0.30   | Highest weight: ND is the primary output      |
| `whirl_accuracy`       | 0.20   | Important but binary; easier to achieve high accuracy |
| `1 - amplitude_mape`   | 0.25   | Inverted so higher is better; regression quality |
| `velocity_r2`          | 0.25   | Cross-validation of physical consistency      |

The score ranges from 0 to 1 (approximately). A perfect model scores 1.0.

**Tier promotion rule**: if the composite score improvement from tier `t-1` to
tier `t` is less than `config.performance_gap_threshold` (default 0.02), the
ladder stops and returns the best model seen so far.

---

## Test Suite

### Overview

The project has five Python test files under
`/Users/adam/Projects/modal-identification/turbomodal/python/tests/`:

| File                     | Subsystem | Description                                  |
|--------------------------|-----------|----------------------------------------------|
| `test_ml.py`             | C         | Feature extraction, all 6 model tiers, evaluation, pipeline, label encoding |
| `test_optimization.py`   | D         | Fisher information, observability, sensor placement, physics checks, calibration |
| `test_python_bindings.py`| A         | C++ binding tests: Material, Mesh, Solver    |
| `test_io.py`             | A         | Mesh import (Gmsh `.msh`), CAD loading       |
| `test_viz.py`            | A         | Visualization: mesh plots, mode shape animation |

### Running Tests

Run the full suite:

```bash
pytest python/tests/ -v
```

Run only ML tests:

```bash
pytest python/tests/test_ml.py -v
```

Run only optimization/explainability tests:

```bash
pytest python/tests/test_optimization.py -v
```

With coverage reporting:

```bash
pytest python/tests/ --cov=turbomodal --cov-report=term-missing
```

### Test Categories

#### Feature Extraction Tests (test_ml.py)

| Test class                          | What it validates                                      |
|-------------------------------------|--------------------------------------------------------|
| `TestExtractFeaturesSpectrogram`    | Output shape `(n_sensors * n_freq_bins,)`, 1-D input handling, empty signal edge case |
| `TestExtractFeaturesMel`            | Output shape `(n_sensors * n_mels,)` for mel features  |
| `TestOrderSpectrum`                 | Correct peak at target engine order; `ValueError` on `rpm=0` |
| `TestTravelingWaveDecomposition`    | Forward wave energy concentrates at the correct ND     |
| `TestExtractFeaturesOrderTracking`  | Output shape `(n_sensors * max_order * 2,)` for order tracking; `ValueError` on missing RPM |
| `TestCrossSpectra`                  | Feature vector grows when `include_cross_spectra=True` |

#### Model Tests (test_ml.py)

| Test class         | What it validates                                               |
|--------------------|-----------------------------------------------------------------|
| `TestLinearModel`  | Train-predict roundtrip; all output keys present; save/load fidelity; RuntimeError on predict-before-train |
| `TestTreeModel`    | Train-predict; `feature_importances_` shape and non-negativity  |
| `TestSVMModel`     | Train-predict; confidence key present                           |
| `TestTierModels`   | All 6 tiers in `TIER_MODELS` registry; each has `train`, `predict`, `save`, `load` methods |
| `TestLabelEncoding`| `_encode_mode_labels` / `_decode_mode_labels` roundtrip          |

#### Pipeline Tests (test_ml.py)

| Test class                   | What it validates                                      |
|------------------------------|--------------------------------------------------------|
| `TestConditionBasedSplit`    | No condition overlap between train/val/test splits     |
| `TestTrainModeIdModel`       | Returns `(model, report)` with correct structure; diminishing returns stops early when gap threshold is set very high |
| `TestPredictModeId`          | End-to-end inference from raw signals through feature extraction to predictions |
| `TestEvaluateModel`          | All 6 metric keys returned; F1 and accuracy in [0, 1]; MAPE and RMSE non-negative |

#### Sensor Optimization Tests (test_optimization.py)

| Test class                        | What it validates                                  |
|-----------------------------------|----------------------------------------------------|
| `TestComputeFisherInformation`    | FIM shape, symmetry, positive semi-definiteness, identity noise equivalence, position-based input |
| `TestComputeObservability`        | Output keys; MAC diagonal = 1.0; condition number >= 1.0 |
| `TestOptimizeSensorPlacement`     | Greedy selection monotonicity; correct sensor count bounds; empty modal results edge case |

#### Physics and Explainability Tests (test_optimization.py)

| Test class                     | What it validates                                     |
|--------------------------------|-------------------------------------------------------|
| `TestPhysicsConsistencyCheck`  | Valid predictions pass all checks; invalid ND flagged; negative frequency flagged; empty predictions handled; whirl ordering violation detected |
| `TestCalibrateConfidence`      | Platt, isotonic, temperature, and conformal methods produce valid confidence in [0, 1]; conformal adds prediction intervals; invalid method raises `ValueError` |
| `TestCalibratedModel`          | Wrapper correctly delegates predict and applies transform |

---

## Continuous Integration

### Required Dependencies

The test suite requires at minimum:

- `numpy`
- `scipy`
- `scikit-learn`
- `pytest`

Optional dependencies for full coverage:

- `xgboost` -- enables XGBoost path in Tier 2 (falls back to RandomForest)
- `torch` -- enables Tiers 4-6 tests
- `shap` -- enables SHAP value tests
- `mlflow` -- enables experiment tracking (falls back to no-op proxy)
- `optuna` -- enables Bayesian sensor optimization refinement

### CI Configuration Notes

1. **Seed determinism**: most tests use explicit RNG seeds
   (`np.random.default_rng(seed)`) for reproducibility. The condition-based
   split uses `seed=42`. PyTorch tiers may exhibit minor variation across
   platforms due to floating-point non-determinism.

2. **Device selection**: `TrainingConfig.device="auto"` auto-selects CUDA >
   MPS > CPU. In CI environments without GPU, tests run on CPU with no
   configuration changes needed.

3. **Test data**: tests use synthetically generated data via
   `_make_synthetic_dataset(n_samples, n_features, seed)`. No external data
   files are required for the ML and optimization test suites.

4. **Timeouts**: Tier 1-3 tests complete in under 5 seconds. Tier 4-6 tests
   (when PyTorch is available) may take 10-30 seconds depending on hardware.
   Setting `config.epochs` to a small value (2-5) in test configurations
   keeps training fast.

5. **Fixture dependencies**: `test_io.py`, `test_viz.py`, and
   `test_python_bindings.py` require the compiled C++ extension module
   (`turbomodal._core`) and mesh fixture files. These tests will be skipped
   if the C++ extension is not built.

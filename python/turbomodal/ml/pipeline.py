"""ML Training & Inference Pipeline (Subsystem C).

Implements the iterative complexity ladder for modal identification:
  Tier 1: Logistic/Linear Regression, Ridge, Lasso
  Tier 2: Decision Trees, Random Forest, XGBoost/LightGBM
  Tier 3: Support Vector Machines (RBF kernel)
  Tier 4: Shallow Neural Networks (1-2 hidden layers, < 500 params)
  Tier 5: 1D-CNN / ResNet on spectral inputs
  Tier 6: Temporal CNN / LSTM / Transformer

The ladder trains and evaluates each tier, stopping as soon as
performance targets are met.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ML model training."""

    # Model selection
    max_tier: int = 6                    # Maximum complexity tier to try
    performance_gap_threshold: float = 0.02  # Min improvement to justify next tier

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Data splits
    validation_split: float = 0.15
    test_split: float = 0.15
    cv_folds: int = 5
    split_by_condition: bool = True      # Split by operating condition, not sample

    # Hyperparameter optimization
    use_optuna: bool = True
    optuna_trials: int = 50

    # Output
    output_dir: str = "ml_output"
    experiment_name: str = "turbomodal_mode_id"
    device: str = "auto"                 # "cpu", "cuda", "mps", "auto"

    # Performance targets (from design doc)
    mode_detection_f1_min: float = 0.92
    whirl_accuracy_min: float = 0.95
    amplitude_mape_max: float = 0.08
    velocity_r2_min: float = 0.93


@runtime_checkable
class ModeIDModel(Protocol):
    """Protocol for mode identification models."""

    def train(self, X: np.ndarray, y: dict[str, np.ndarray],
              config: TrainingConfig) -> dict[str, float]:
        """Train the model. Returns metrics dict."""
        ...

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode IDs from features.

        Returns dict with keys:
            'nodal_diameter': (N,) int
            'nodal_circle': (N,) int
            'frequency': (N,) float
            'whirl_direction': (N,) int
            'amplitude': (N,) float
            'wave_velocity': (N,) float
            'confidence': (N,) float
        """
        ...

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


# ---------------------------------------------------------------------------
# MLflow no-op fallback
# ---------------------------------------------------------------------------

class _NoOpRun:
    info: Any = None


@contextmanager
def _noop_context(*args: Any, **kwargs: Any):
    yield _NoOpRun()


class _MLflowProxy:
    """Delegates to mlflow when installed, otherwise silently no-ops."""

    def __init__(self) -> None:
        try:
            import mlflow as _mlflow
            self._mlflow = _mlflow
        except ImportError:
            self._mlflow = None

    def set_experiment(self, name: str) -> None:
        if self._mlflow is not None:
            self._mlflow.set_experiment(name)

    def start_run(self, **kwargs: Any):
        if self._mlflow is not None:
            return self._mlflow.start_run(**kwargs)
        return _noop_context()

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics)

    def log_metric(self, key: str, value: float) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metric(key, value)


# ---------------------------------------------------------------------------
# Condition-based data splitting
# ---------------------------------------------------------------------------

def _condition_based_split(
    condition_ids: np.ndarray,
    n_samples: int,
    test_frac: float,
    val_frac: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by operating condition via GroupShuffleSplit.

    Returns ``(train_idx, val_idx, test_idx)`` ensuring no condition
    appears in multiple splits.
    """
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:
        logger.warning("sklearn not available; using random (non-grouped) split.")
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_samples)
        n_test = max(1, int(n_samples * test_frac))
        n_val = max(1, int(n_samples * val_frac))
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]

    dummy_X = np.arange(n_samples)

    # First split: (train+val) vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(dummy_X, groups=condition_ids))

    # Second split: train vs val from the remaining portion
    trainval_conds = condition_ids[trainval_idx]
    adjusted_val = min(val_frac / (1.0 - test_frac), 0.5)

    if adjusted_val > 0 and len(np.unique(trainval_conds)) >= 2:
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=adjusted_val, random_state=seed + 1,
        )
        train_rel, val_rel = next(gss_val.split(trainval_idx, groups=trainval_conds))
        return trainval_idx[train_rel], trainval_idx[val_rel], test_idx

    return trainval_idx, np.array([], dtype=np.intp), test_idx


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _composite_score(metrics: dict[str, float], config: TrainingConfig) -> float:
    """Weighted composite: 0.3*f1 + 0.2*whirl + 0.25*(1-mape) + 0.25*vel_r2."""
    return (
        0.30 * metrics.get("mode_detection_f1", 0.0)
        + 0.20 * metrics.get("whirl_accuracy", 0.0)
        + 0.25 * (1.0 - metrics.get("amplitude_mape", 1.0))
        + 0.25 * metrics.get("velocity_r2", 0.0)
    )


def _check_targets(metrics: dict[str, float], config: TrainingConfig) -> bool:
    """True when all four performance targets are satisfied."""
    return (
        metrics.get("mode_detection_f1", 0.0) >= config.mode_detection_f1_min
        and metrics.get("whirl_accuracy", 0.0) >= config.whirl_accuracy_min
        and metrics.get("amplitude_mape", 1.0) <= config.amplitude_mape_max
        and metrics.get("velocity_r2", 0.0) >= config.velocity_r2_min
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(
    dataset_path: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Load features and labels from a ``.npz`` file."""
    if dataset_path.endswith(".npz"):
        data = np.load(dataset_path, allow_pickle=False)
        X = data["X"]
        condition_ids = data.get("condition_ids", None)
        y: dict[str, np.ndarray] = {}
        for key in (
            "nodal_diameter", "nodal_circle", "whirl_direction",
            "frequency", "amplitude", "wave_velocity",
        ):
            if key in data:
                y[key] = data[key]
        if "amplitude" not in y:
            y["amplitude"] = np.ones(X.shape[0], dtype=np.float64)
        return X, y, condition_ids

    raise ValueError(
        f"Cannot load '{dataset_path}' directly. HDF5 loading requires a "
        "sensor_array and signal_config; pass X and y directly or use .npz."
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_mode_id_model(
    dataset_path: str | None = None,
    config: TrainingConfig = TrainingConfig(),
    *,
    X: np.ndarray | None = None,
    y: dict[str, np.ndarray] | None = None,
    condition_ids: np.ndarray | None = None,
) -> tuple[object, dict]:
    """Train a mode identification model via the complexity ladder.

    Accepts pre-built ``(X, y)`` arrays or a path to a ``.npz`` file.

    Returns ``(best_model, report)`` where *report* contains
    ``best_tier``, ``val_metrics``, ``test_metrics``, and ``tier_history``.
    """
    from turbomodal.ml.models import TIER_MODELS

    # 1. Load / validate data
    if X is not None and y is not None:
        logger.info("Using pre-built feature matrix (%d samples).", X.shape[0])
    elif dataset_path is not None:
        X, y, condition_ids = _load_dataset(dataset_path)
    else:
        raise ValueError("Either (X, y) or dataset_path must be provided.")

    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Dataset is empty; cannot train.")
    if condition_ids is None:
        condition_ids = np.arange(n_samples)

    # 2. Split data
    if config.split_by_condition:
        train_idx, val_idx, test_idx = _condition_based_split(
            condition_ids, n_samples, config.test_split,
            config.validation_split, seed=42,
        )
    else:
        rng = np.random.RandomState(42)
        idx = rng.permutation(n_samples)
        n_test = max(1, int(n_samples * config.test_split))
        n_val = max(1, int(n_samples * config.validation_split))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

    X_train = X[train_idx]
    y_train = {k: v[train_idx] for k, v in y.items()}
    X_test = X[test_idx]
    y_test = {k: v[test_idx] for k, v in y.items()}

    if len(val_idx) > 0:
        X_val = X[val_idx]
        y_val = {k: v[val_idx] for k, v in y.items()}
    else:
        X_val, y_val = X_test, y_test

    logger.info(
        "Data split: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # 3. Complexity ladder
    mlf = _MLflowProxy()
    mlf.set_experiment(config.experiment_name)

    best_model: object | None = None
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    best_tier = 1
    prev_score = float("-inf")
    tier_history: list[dict[str, Any]] = []

    with mlf.start_run(run_name="complexity_ladder"):
        for tier in range(1, config.max_tier + 1):
            logger.info("--- Tier %d ---", tier)
            with mlf.start_run(run_name=f"tier_{tier}", nested=True):
                model = TIER_MODELS[tier]()

                start_time = time.time()
                try:
                    train_metrics = model.train(X_train, y_train, config)
                except Exception:
                    logger.exception("Tier %d training failed; skipping.", tier)
                    continue
                train_time = time.time() - start_time

                val_metrics = evaluate_model(model, X_val, y_val)

                mlf.log_params({"tier": str(tier), "n_train": str(len(train_idx))})
                mlf.log_metrics(val_metrics)
                mlf.log_metric("train_time_s", train_time)

                score = _composite_score(val_metrics, config)
                targets_met = _check_targets(val_metrics, config)

                tier_history.append({
                    "tier": tier,
                    "train_time_s": train_time,
                    "val_metrics": val_metrics,
                    "composite_score": score,
                    "targets_met": targets_met,
                })

                logger.info(
                    "Tier %d: score=%.4f  f1=%.3f  whirl=%.3f  "
                    "mape=%.3f  vel_r2=%.3f  (%.1fs)",
                    tier, score,
                    val_metrics.get("mode_detection_f1", 0),
                    val_metrics.get("whirl_accuracy", 0),
                    val_metrics.get("amplitude_mape", 1),
                    val_metrics.get("velocity_r2", 0),
                    train_time,
                )

                if score > best_score:
                    best_model = model
                    best_score = score
                    best_metrics = val_metrics
                    best_tier = tier

                if targets_met:
                    logger.info("All targets met at Tier %d.", tier)
                    break

                if tier > 1 and (score - prev_score) < config.performance_gap_threshold:
                    logger.info(
                        "Diminishing returns at Tier %d (delta=%.4f < %.4f).",
                        tier, score - prev_score, config.performance_gap_threshold,
                    )
                    break

                prev_score = score

        # 4. Final test-set evaluation
        if best_model is not None:
            test_metrics = evaluate_model(best_model, X_test, y_test)
            mlf.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            logger.info("Test metrics (Tier %d): %s", best_tier, test_metrics)
        else:
            test_metrics = {}
            logger.warning("No model was successfully trained.")

    return best_model, {
        "best_tier": best_tier,
        "val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "tier_history": tier_history,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_mode_id(
    model: ModeIDModel,
    signals: np.ndarray,
    sample_rate: float,
    rpm: float = 0.0,
) -> dict[str, np.ndarray]:
    """Run mode identification on raw sensor signals.

    Extracts features via ``extract_features`` then calls ``model.predict``.
    """
    from turbomodal.ml.features import extract_features, FeatureConfig

    feat_config = FeatureConfig(rpm=rpm)
    features = extract_features(signals, sample_rate, feat_config)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict(features)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: ModeIDModel,
    X_test: np.ndarray,
    y_test: dict[str, np.ndarray],
) -> dict[str, float]:
    """Evaluate model on all six sub-task metrics.

    Returns dict with keys: ``mode_detection_f1``, ``whirl_accuracy``,
    ``amplitude_mape``, ``amplitude_r2``, ``velocity_rmse``, ``velocity_r2``.
    """
    from turbomodal.ml.models import _encode_mode_labels

    preds = model.predict(X_test)

    # Mode detection F1 (macro)
    true_mode = _encode_mode_labels(y_test["nodal_diameter"], y_test["nodal_circle"])
    pred_mode = _encode_mode_labels(preds["nodal_diameter"], preds["nodal_circle"])
    try:
        from sklearn.metrics import f1_score
        mode_f1 = float(f1_score(true_mode, pred_mode, average="macro", zero_division=0))
    except ImportError:
        mode_f1 = _manual_f1(true_mode, pred_mode)

    # Whirl balanced accuracy
    true_whirl = np.asarray(y_test["whirl_direction"], dtype=np.int64)
    pred_whirl = np.asarray(preds["whirl_direction"], dtype=np.int64)
    try:
        from sklearn.metrics import balanced_accuracy_score
        whirl_acc = float(balanced_accuracy_score(true_whirl, pred_whirl))
    except ImportError:
        whirl_acc = float(np.mean(true_whirl == pred_whirl))

    # Amplitude MAPE
    true_amp = np.asarray(y_test["amplitude"], dtype=np.float64)
    pred_amp = np.asarray(preds["amplitude"], dtype=np.float64)
    denom = np.maximum(np.abs(true_amp), 1e-8)
    amplitude_mape = float(np.mean(np.abs(true_amp - pred_amp) / denom))

    # Amplitude R2
    try:
        from sklearn.metrics import r2_score
        amplitude_r2 = float(r2_score(true_amp, pred_amp))
    except ImportError:
        amplitude_r2 = _manual_r2(true_amp, pred_amp)

    # Velocity RMSE and R2
    true_vel = np.asarray(y_test["wave_velocity"], dtype=np.float64)
    pred_vel = np.asarray(preds["wave_velocity"], dtype=np.float64)
    velocity_rmse = float(np.sqrt(np.mean((true_vel - pred_vel) ** 2)))
    try:
        from sklearn.metrics import r2_score as _r2
        velocity_r2 = float(_r2(true_vel, pred_vel))
    except ImportError:
        velocity_r2 = _manual_r2(true_vel, pred_vel)

    return {
        "mode_detection_f1": mode_f1,
        "whirl_accuracy": whirl_acc,
        "amplitude_mape": amplitude_mape,
        "amplitude_r2": amplitude_r2,
        "velocity_rmse": velocity_rmse,
        "velocity_r2": velocity_r2,
    }


# ---------------------------------------------------------------------------
# Manual metric fallbacks (when sklearn is unavailable)
# ---------------------------------------------------------------------------

def _manual_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else 0.0


def _manual_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lab in labels:
        tp = float(np.sum((y_pred == lab) & (y_true == lab)))
        fp = float(np.sum((y_pred == lab) & (y_true != lab)))
        fn = float(np.sum((y_pred != lab) & (y_true == lab)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

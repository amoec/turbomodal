"""Explainability & confidence framework (Subsystem D).

Provides:
- SHAP value computation for sensor channel importance
- Grad-CAM attribution for time-frequency regions
- Physics consistency checks for mode identification predictions
- Confidence score calibration
- Anomaly detection for out-of-distribution inputs
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CalibratedModel wrapper
# ---------------------------------------------------------------------------


class CalibratedModel:
    """Wrapper that applies confidence calibration to a base model.

    Delegates all core methods to the underlying *base_model* while
    transforming the ``"confidence"`` key in :meth:`predict` output
    through a fitted calibration function.

    Parameters
    ----------
    base_model : ModeIDModel
        Any trained model instance from the complexity ladder.
    calibration_transform : callable
        ``(raw_confidence: np.ndarray) -> np.ndarray`` mapping.
    method : str
        Name of the calibration method used (for bookkeeping).
    """

    def __init__(
        self,
        base_model: Any,
        calibration_transform: Callable[[np.ndarray], np.ndarray],
        method: str,
    ) -> None:
        self.base_model = base_model
        self._calibration_transform = calibration_transform
        self._method = method

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict with calibrated confidence scores."""
        preds = self.base_model.predict(X)
        raw_confidence = preds["confidence"]
        preds["confidence"] = self._calibration_transform(raw_confidence)
        return preds

    # -- delegate persistence and training ----------------------------------

    def save(self, path: str) -> None:  # noqa: D401
        self.base_model.save(path)

    def load(self, path: str) -> None:  # noqa: D401
        self.base_model.load(path)

    def train(self, X: np.ndarray, y: Any, config: Any) -> dict[str, float]:
        return self.base_model.train(X, y, config)


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    signals: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Compute SHAP values for sensor channel importance.

    Shows which sensors contributed most to the mode identification.
    Uses TreeSHAP for tree models, KernelSHAP for neural networks.

    Parameters
    ----------
    model : trained ModeIDModel
    signals : (n_samples, n_features) input features
    feature_names : optional feature names for labeling

    Returns
    -------
    shap_values : np.ndarray
        For tree models: ``(n_samples, n_features, 4)`` — one slice per
        estimator (mode_clf, whirl_clf, amp_reg, vel_reg).
        For other models: ``(n_samples, n_features, n_outputs)``.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP explainability requires the 'shap' package (>=0.42). "
            "Install with: pip install turbomodal[ml]"
        ) from exc

    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if not getattr(model, "_is_fitted", False):
        raise RuntimeError("Model must be fitted before computing SHAP values.")

    # Detect model type for appropriate explainer
    from turbomodal.ml.models import TreeModeIDModel

    # ---- Tree-based models: use TreeExplainer (fast, exact) ---------------
    if isinstance(model, TreeModeIDModel) and model._is_fitted:
        shap_values_list: list[np.ndarray] = []
        for estimator_name in ("_mode_clf", "_whirl_clf", "_amp_reg", "_vel_reg"):
            estimator = getattr(model, estimator_name)
            try:
                explainer = shap.TreeExplainer(estimator)
                sv = explainer.shap_values(signals)
                if isinstance(sv, list):
                    # Multi-class: take mean absolute across classes
                    sv = np.mean(np.abs(np.array(sv)), axis=0)
                shap_values_list.append(np.asarray(sv, dtype=np.float64))
            except Exception:
                logger.warning(
                    "TreeExplainer failed for %s; using zeros.", estimator_name
                )
                shap_values_list.append(np.zeros_like(signals))

        # Stack: (n_samples, n_features, 4)
        return np.stack(shap_values_list, axis=-1)

    # ---- All other models: use KernelExplainer ----------------------------
    def predict_fn(X: np.ndarray) -> np.ndarray:
        preds = model.predict(X)
        return np.column_stack([
            preds["nodal_diameter"].astype(float),
            preds["whirl_direction"].astype(float),
            preds["amplitude"],
            preds["wave_velocity"],
        ])

    n_bg = min(100, signals.shape[0])
    rng = np.random.RandomState(42)
    background = signals[rng.choice(signals.shape[0], n_bg, replace=False)]

    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(signals, nsamples=100)

    if isinstance(sv, list):
        # (n_outputs, n_samples, n_features) -> (n_samples, n_features, n_outputs)
        return np.stack(sv, axis=-1)

    sv = np.asarray(sv, dtype=np.float64)
    return sv[..., np.newaxis] if sv.ndim == 2 else sv


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


def compute_grad_cam(
    model: Any,
    signals: np.ndarray,
    target_class: int,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute Grad-CAM attribution for mode identification decision.

    Shows which time-frequency regions of the signal drove the classification.
    Only applicable to CNN-based models (Tier 5-6).

    Parameters
    ----------
    model : trained CNN/Temporal model (CNNModeIDModel or TemporalModeIDModel)
    signals : (batch, n_features) input features
    target_class : class index to explain (for mode detection head)
    layer_name : CNN layer to use for attribution (default: last conv layer)

    Returns
    -------
    attribution : (batch, length) Grad-CAM heatmap normalised to [0, 1]
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Grad-CAM requires PyTorch. Install with: pip install turbomodal[ml]"
        ) from exc

    from turbomodal.ml.models import CNNModeIDModel, TemporalModeIDModel

    if not isinstance(model, (CNNModeIDModel, TemporalModeIDModel)):
        raise TypeError(
            f"Grad-CAM only supports CNN/Temporal models (Tiers 5-6), "
            f"got {type(model).__name__}"
        )

    if not model._is_fitted:
        raise RuntimeError("Model must be fitted before computing Grad-CAM.")

    nn_model = model._model
    device = torch.device(model._device_str)
    nn_model.eval()

    # Prepare input
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if hasattr(model, "_scaler") and model._scaler is not None:
        signals = model._scaler.transform(signals).astype(np.float32)

    X_t = torch.as_tensor(signals, dtype=torch.float32).to(device)

    # Reshape for CNN / Temporal and select target layer
    if isinstance(model, CNNModeIDModel):
        n_ch, n_freq = model._n_channels, model._n_freq_bins
        X_t = X_t.view(X_t.size(0), n_ch, n_freq)
        target_layer = nn_model.features[-3]  # last Conv+BN+ReLU before pool
    else:  # TemporalModeIDModel
        n_ch, seq_len = model._n_channels, model._seq_len
        X_t = X_t.view(X_t.size(0), n_ch, seq_len)
        target_layer = nn_model.conv[-1]  # last conv sub-layer

    # Register hooks
    activations: dict[str, Any] = {}
    gradients: dict[str, Any] = {}

    def fwd_hook(module: Any, inp: Any, out: Any) -> None:
        activations["value"] = out

    def bwd_hook(module: Any, grad_in: Any, grad_out: Any) -> None:
        gradients["value"] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Forward pass
        X_t.requires_grad_(True)
        out_mode, _, _, _ = nn_model(X_t)

        # Backward for target class
        nn_model.zero_grad()
        one_hot = torch.zeros_like(out_mode)
        one_hot[:, target_class] = 1.0
        out_mode.backward(gradient=one_hot)

        # Compute Grad-CAM
        act = activations["value"]   # (batch, channels, length)
        grad = gradients["value"]    # (batch, channels, length)

        # Global-average-pool gradients -> per-channel weights
        weights = grad.mean(dim=-1, keepdim=True)  # (batch, channels, 1)

        # Weighted combination
        cam = (weights * act).sum(dim=1)  # (batch, length)
        cam = torch.relu(cam)

        # Normalise each sample to [0, 1]
        cam_min = cam.min(dim=-1, keepdim=True).values
        cam_max = cam.max(dim=-1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# ---------------------------------------------------------------------------
# Physics consistency check
# ---------------------------------------------------------------------------


def physics_consistency_check(
    predictions: dict[str, np.ndarray],
    num_sectors: int,
    rpm: float = 0.0,
    blade_radius: float = 0.3,
    epistemic_uncertainty: Optional[np.ndarray] = None,
    epistemic_threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Check predictions against known physical constraints.

    Constraints checked:
    1. Frequency should be positive.
    2. ND must be in range [0, N/2] for N sectors.
    3. Whirl direction must be in {-1, 0, 1}.
    4. Forward whirl frequency >= backward whirl frequency per ND group
       (when RPM > 0).
    5. Wave velocity consistent with ``v = 2*pi*f*R / ND`` within 50%
       tolerance (when ND > 0).

    Parameters
    ----------
    predictions : dict from model.predict() with keys:
        'nodal_diameter', 'frequency', 'whirl_direction',
        'amplitude', 'wave_velocity'
    num_sectors : number of blades/sectors
    rpm : rotational speed (0 = stationary, no whirl check)
    blade_radius : nominal blade tip radius in metres (default 0.3 m)

    Returns
    -------
    dict with keys:
        'is_consistent': (N,) bool array, True if all checks pass
        'violations': (N,) list of violation descriptions
        'consistency_score': (N,) float in [0, 1], fraction of checks passed
        'anomaly_flag': (N,) bool, True if prediction should be flagged
    """
    n = len(predictions.get("frequency", []))
    if n == 0:
        return {
            "is_consistent": np.array([], dtype=bool),
            "violations": [],
            "consistency_score": np.array([]),
            "anomaly_flag": np.array([], dtype=bool),
        }

    freq = np.asarray(predictions.get("frequency", np.zeros(n)))
    nd = np.asarray(predictions.get("nodal_diameter", np.zeros(n, dtype=int)))
    whirl = np.asarray(predictions.get("whirl_direction", np.zeros(n, dtype=int)))
    wave_vel = np.asarray(predictions.get("wave_velocity", np.zeros(n)))

    max_nd = num_sectors // 2
    violations: list[list[str]] = [[] for _ in range(n)]
    checks_passed = np.zeros(n)
    total_checks = np.zeros(n)

    for i in range(n):
        # Check 1: frequency > 0
        total_checks[i] += 1
        if freq[i] > 0:
            checks_passed[i] += 1
        else:
            violations[i].append(f"Negative/zero frequency: {freq[i]:.1f} Hz")

        # Check 2: ND in valid range
        total_checks[i] += 1
        if 0 <= nd[i] <= max_nd:
            checks_passed[i] += 1
        else:
            violations[i].append(f"ND={nd[i]} out of range [0, {max_nd}]")

        # Check 3: whirl direction valid
        total_checks[i] += 1
        if whirl[i] in (-1, 0, 1):
            checks_passed[i] += 1
        else:
            violations[i].append(f"Invalid whirl direction: {whirl[i]}")

    # ------------------------------------------------------------------
    # Check 4: whirl ordering (forward freq >= backward freq per ND)
    # Only applies when RPM > 0 and we have frequency data.
    # ------------------------------------------------------------------
    if rpm > 0:
        # Build groups: ND value -> list of (index, whirl, frequency)
        nd_groups: dict[int, list[tuple[int, int, float]]] = {}
        for i in range(n):
            nd_val = int(nd[i])
            nd_groups.setdefault(nd_val, []).append(
                (i, int(whirl[i]), float(freq[i]))
            )

        for nd_val, members in nd_groups.items():
            fwd_freqs = [f for (_, w, f) in members if w == 1]
            bwd_freqs = [f for (_, w, f) in members if w == -1]

            if fwd_freqs and bwd_freqs:
                min_fwd = min(fwd_freqs)
                max_bwd = max(bwd_freqs)
                for idx, w, _ in members:
                    total_checks[idx] += 1
                    if min_fwd >= max_bwd:
                        checks_passed[idx] += 1
                    else:
                        violations[idx].append(
                            f"Whirl ordering violated for ND={nd_val}: "
                            f"min forward freq ({min_fwd:.1f} Hz) "
                            f"< max backward freq ({max_bwd:.1f} Hz)"
                        )

    # ------------------------------------------------------------------
    # Check 5: velocity consistency  v_expected = 2*pi*f*R / ND
    # Only for samples with ND > 0 and freq > 0.
    # ------------------------------------------------------------------
    tolerance = 0.5
    for i in range(n):
        if nd[i] > 0 and freq[i] > 0:
            total_checks[i] += 1
            v_expected = 2.0 * np.pi * freq[i] * blade_radius / float(nd[i])
            if v_expected > 0:
                relative_error = abs(wave_vel[i] - v_expected) / v_expected
                if relative_error < tolerance:
                    checks_passed[i] += 1
                else:
                    violations[i].append(
                        f"Velocity inconsistent: predicted {wave_vel[i]:.2f} m/s, "
                        f"expected {v_expected:.2f} m/s "
                        f"(relative error {relative_error:.1%} > {tolerance:.0%})"
                    )

    # ------------------------------------------------------------------
    # Check 6: Epistemic uncertainty threshold (D12)
    # ------------------------------------------------------------------
    if epistemic_uncertainty is not None:
        epistemic_uncertainty = np.asarray(epistemic_uncertainty, dtype=np.float64)
        for i in range(min(n, len(epistemic_uncertainty))):
            total_checks[i] += 1
            if epistemic_uncertainty[i] <= epistemic_threshold:
                checks_passed[i] += 1
            else:
                violations[i].append(
                    f"High epistemic uncertainty: {epistemic_uncertainty[i]:.4f} "
                    f"> threshold {epistemic_threshold:.4f}"
                )

    consistency_score = np.where(total_checks > 0,
                                  checks_passed / total_checks, 1.0)
    is_consistent = consistency_score >= 1.0
    anomaly_flag = consistency_score < 0.8

    return {
        "is_consistent": is_consistent,
        "violations": violations,
        "consistency_score": consistency_score,
        "anomaly_flag": anomaly_flag,
    }


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


def calibrate_confidence(
    model: Any,
    X_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    method: str = "platt",
) -> CalibratedModel:
    """Calibrate model confidence scores on validation data.

    Methods:
    - ``'platt'``:  Platt scaling (logistic regression on raw confidence).
    - ``'isotonic'``:  Isotonic regression on raw confidence.
    - ``'temperature'``:  Temperature scaling of raw confidence scores.
    - ``'conformal'``:  Conformal prediction intervals (regression tasks).

    Parameters
    ----------
    model : trained ModeIDModel
    X_val : (n_val, n_features) validation features
    y_val : dict with at least ``'nodal_diameter'`` (classification target)
            and ``'amplitude'``, ``'wave_velocity'`` (regression targets).
    method : calibration method name

    Returns
    -------
    CalibratedModel
        Wrapped model with calibrated confidence (and, for conformal,
        additional prediction-interval keys).
    """
    if not getattr(model, "_is_fitted", False):
        raise RuntimeError("Model must be fitted before calibration.")

    X_val = np.asarray(X_val)
    preds = model.predict(X_val)
    raw_conf = preds["confidence"].astype(np.float64)

    # Correctness labels: 1 if mode prediction matches ground truth
    from turbomodal.ml.models import _encode_mode_labels

    true_labels = _encode_mode_labels(
        np.asarray(y_val["nodal_diameter"]),
        np.asarray(y_val.get("nodal_circle", np.zeros_like(y_val["nodal_diameter"]))),
    )
    pred_labels = _encode_mode_labels(
        preds["nodal_diameter"], preds.get("nodal_circle", np.zeros_like(preds["nodal_diameter"]))
    )
    correct = (true_labels == pred_labels).astype(np.float64)

    # ---- Platt scaling ----------------------------------------------------
    if method == "platt":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError(
                "Platt calibration requires scikit-learn."
            ) from exc

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(raw_conf.reshape(-1, 1), correct)
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        logger.info("Platt calibration fitted: a=%.4f, b=%.4f", a, b)

        def platt_transform(conf: np.ndarray) -> np.ndarray:
            logits = a * conf + b
            return 1.0 / (1.0 + np.exp(-logits))

        return CalibratedModel(model, platt_transform, method)

    # ---- Isotonic regression ----------------------------------------------
    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "Isotonic calibration requires scikit-learn."
            ) from exc

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_conf, correct)
        logger.info("Isotonic calibration fitted on %d samples.", len(raw_conf))

        def isotonic_transform(conf: np.ndarray) -> np.ndarray:
            return iso.predict(conf)

        return CalibratedModel(model, isotonic_transform, method)

    # ---- Temperature scaling ----------------------------------------------
    if method == "temperature":
        try:
            from scipy.optimize import minimize_scalar
        except ImportError as exc:
            raise ImportError(
                "Temperature calibration requires scipy."
            ) from exc

        # Find T that minimises negative log-likelihood of correct labels
        # under the model:  calibrated = conf^(1/T) / Z
        def nll(T: float) -> float:
            T = max(T, 0.1)
            scaled = np.power(np.clip(raw_conf, 1e-12, 1.0), 1.0 / T)
            # Normalise so scaled + (1 - scaled) still forms valid prob
            cal_prob = scaled / (scaled + np.power(
                np.clip(1.0 - raw_conf, 1e-12, 1.0), 1.0 / T
            ))
            cal_prob = np.clip(cal_prob, 1e-12, 1.0 - 1e-12)
            # Binary cross-entropy
            return -float(np.mean(
                correct * np.log(cal_prob)
                + (1.0 - correct) * np.log(1.0 - cal_prob)
            ))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T_opt = float(result.x)
        logger.info("Temperature calibration: T=%.4f", T_opt)

        def temperature_transform(conf: np.ndarray) -> np.ndarray:
            scaled = np.power(np.clip(conf, 1e-12, 1.0), 1.0 / T_opt)
            denom = scaled + np.power(
                np.clip(1.0 - conf, 1e-12, 1.0), 1.0 / T_opt
            )
            return scaled / denom

        return CalibratedModel(model, temperature_transform, method)

    # ---- Conformal prediction intervals -----------------------------------
    if method == "conformal":
        amp_true = np.asarray(y_val["amplitude"], dtype=np.float64)
        vel_true = np.asarray(y_val["wave_velocity"], dtype=np.float64)
        amp_pred = preds["amplitude"].astype(np.float64)
        vel_pred = preds["wave_velocity"].astype(np.float64)

        amp_scores = np.sort(np.abs(amp_true - amp_pred))
        vel_scores = np.sort(np.abs(vel_true - vel_pred))

        # 90 % coverage quantile
        alpha = 0.10
        n_cal = len(amp_scores)
        q_idx = min(int(np.ceil((1.0 - alpha) * (n_cal + 1))) - 1, n_cal - 1)
        q_idx = max(q_idx, 0)
        amp_q = float(amp_scores[q_idx])
        vel_q = float(vel_scores[q_idx])
        logger.info(
            "Conformal calibration (90%%): amp_q=%.4f, vel_q=%.4f",
            amp_q,
            vel_q,
        )

        base_model_ref = model

        class _ConformalModel(CalibratedModel):
            """CalibratedModel that also adds prediction intervals."""

            def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
                preds = base_model_ref.predict(X)
                amp = preds["amplitude"].astype(np.float64)
                vel = preds["wave_velocity"].astype(np.float64)
                preds["prediction_interval_lower"] = np.column_stack([
                    amp - amp_q, vel - vel_q,
                ])
                preds["prediction_interval_upper"] = np.column_stack([
                    amp + amp_q, vel + vel_q,
                ])
                return preds

        return _ConformalModel(model, lambda c: c, method)

    raise ValueError(
        f"Unknown calibration method '{method}'. "
        "Choose from: 'platt', 'isotonic', 'temperature', 'conformal'."
    )


# ---------------------------------------------------------------------------
# Model selection report (D9)
# ---------------------------------------------------------------------------


def generate_model_selection_report(
    training_report: dict,
) -> dict[str, Any]:
    """Generate a structured model selection report from training results.

    Parameters
    ----------
    training_report : dict from ``train_mode_id_model()``.

    Returns
    -------
    dict with keys: ``summary``, ``per_tier_metrics``, ``gap_analysis``,
    ``selected_tier``.
    """
    tier_history = training_report.get("tier_history", [])
    best_tier = training_report.get("best_tier", 0)

    per_tier_metrics = {}
    for entry in tier_history:
        per_tier_metrics[entry["tier"]] = entry["val_metrics"]

    # Gap analysis: score delta between consecutive tiers
    gap_analysis: list[dict] = []
    for i in range(1, len(tier_history)):
        delta = tier_history[i]["composite_score"] - tier_history[i - 1]["composite_score"]
        gap_analysis.append({
            "from_tier": tier_history[i - 1]["tier"],
            "to_tier": tier_history[i]["tier"],
            "score_delta": delta,
        })

    best_metrics = training_report.get("val_metrics", {})
    summary = (
        f"Selected Tier {best_tier} with composite score "
        f"{best_metrics.get('mode_detection_f1', 0):.3f} F1, "
        f"{best_metrics.get('whirl_accuracy', 0):.3f} whirl acc, "
        f"{best_metrics.get('amplitude_mape', 1):.3f} MAPE, "
        f"{best_metrics.get('velocity_r2', 0):.3f} vel R2."
    )

    return {
        "summary": summary,
        "per_tier_metrics": per_tier_metrics,
        "gap_analysis": gap_analysis,
        "selected_tier": best_tier,
    }


# ---------------------------------------------------------------------------
# Per-prediction explanation card (D10)
# ---------------------------------------------------------------------------


def generate_explanation_card(
    model: Any,
    X_single: np.ndarray,
    predictions: dict[str, np.ndarray],
    sample_idx: int = 0,
    num_sectors: int = 36,
    rpm: float = 0.0,
    feature_names: Optional[list[str]] = None,
    uncertainty: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Generate a per-prediction explanation card.

    Parameters
    ----------
    model : trained model
    X_single : (1, n_features) or (n_features,) single sample
    predictions : full predictions dict from model.predict()
    sample_idx : index into predictions arrays
    num_sectors : number of sectors
    rpm : RPM for physics check
    feature_names : optional feature names for SHAP
    uncertainty : optional uncertainty dict from predict_with_uncertainty()

    Returns
    -------
    dict with keys: ``predicted_values``, ``confidence``, ``physics_check``,
    ``shap_values``, ``confidence_interval``, ``anomaly_flag``,
    ``explanation_text``.
    """
    X_single = np.atleast_2d(X_single)

    # Predicted values
    predicted_values = {
        k: float(predictions[k][sample_idx])
        for k in ("nodal_diameter", "whirl_direction", "amplitude", "wave_velocity")
        if k in predictions
    }

    confidence = float(predictions.get("confidence", np.ones(1))[sample_idx])

    # Physics check
    single_preds = {k: v[sample_idx:sample_idx + 1] for k, v in predictions.items()
                    if isinstance(v, np.ndarray) and v.ndim >= 1}
    physics = physics_consistency_check(single_preds, num_sectors, rpm)
    anomaly_flag = bool(physics["anomaly_flag"][0]) if len(physics["anomaly_flag"]) > 0 else False

    # SHAP values (best-effort)
    shap_vals = None
    try:
        sv = compute_shap_values(model, X_single, feature_names)
        shap_vals = sv[0] if sv.ndim > 1 else sv
    except Exception:
        pass

    # Confidence interval from uncertainty
    confidence_interval = None
    if uncertainty is not None:
        amp_var = uncertainty.get("amplitude_total_var", np.zeros(1))
        vel_var = uncertainty.get("velocity_total_var", np.zeros(1))
        amp_std = np.sqrt(amp_var[sample_idx]) if sample_idx < len(amp_var) else 0
        vel_std = np.sqrt(vel_var[sample_idx]) if sample_idx < len(vel_var) else 0
        confidence_interval = {
            "amplitude_pm": float(1.96 * amp_std),
            "velocity_pm": float(1.96 * vel_std),
        }

    # Check epistemic anomaly
    if uncertainty is not None:
        epi = uncertainty.get("amplitude_epistemic_var", None)
        if epi is not None and sample_idx < len(epi) and epi[sample_idx] > 0.1:
            anomaly_flag = True

    explanation_text = (
        f"Predicted ND={predicted_values.get('nodal_diameter', '?')}, "
        f"whirl={'FW' if predicted_values.get('whirl_direction', 0) > 0 else 'BW' if predicted_values.get('whirl_direction', 0) < 0 else 'standing'}, "
        f"amp={predicted_values.get('amplitude', 0):.4f}, "
        f"vel={predicted_values.get('wave_velocity', 0):.2f} m/s "
        f"(confidence={confidence:.2f})."
    )

    return {
        "predicted_values": predicted_values,
        "confidence": confidence,
        "physics_check": physics,
        "shap_values": shap_vals,
        "confidence_interval": confidence_interval,
        "anomaly_flag": anomaly_flag,
        "explanation_text": explanation_text,
    }

"""Sensor Optimization & Explainability Module (Subsystem D).

This package provides:
- Sensor placement optimization using Fisher Information Matrix
  and Bayesian optimization
- Explainability tools (SHAP, Grad-CAM) for model predictions
- Physics consistency checks for mode identification results
- Confidence calibration (Platt, isotonic, temperature scaling, conformal)
"""

from turbomodal.optimization.sensor_placement import (
    SensorOptimizationConfig,
    SensorOptimizationResult,
    optimize_sensor_placement,
    compute_fisher_information,
    compute_observability,
)
from turbomodal.optimization.explainability import (
    CalibratedModel,
    compute_shap_values,
    compute_grad_cam,
    physics_consistency_check,
    calibrate_confidence,
    generate_model_selection_report,
    generate_explanation_card,
)

__all__ = [
    "SensorOptimizationConfig",
    "SensorOptimizationResult",
    "optimize_sensor_placement",
    "compute_fisher_information",
    "compute_observability",
    "CalibratedModel",
    "compute_shap_values",
    "compute_grad_cam",
    "physics_consistency_check",
    "calibrate_confidence",
    "generate_model_selection_report",
    "generate_explanation_card",
]

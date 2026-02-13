"""ML Training & Inference Pipeline (Subsystem C).

This package provides the machine learning pipeline for modal identification
from sensor signals. It implements the iterative complexity ladder (Tiers 1-6)
and feature extraction for the four sub-tasks:

1. Mode Detection (multi-label classification → ND, NC pairs)
2. Whirl Classification (binary per mode → FW/BW)
3. Amplitude Estimation (regression per mode → peak amplitude)
4. Propagation Velocity (regression per mode → m/s)
"""

from turbomodal.ml.features import (
    FeatureConfig,
    extract_features,
    compute_order_spectrum,
    traveling_wave_decomposition,
    build_feature_matrix,
)
from turbomodal.ml.pipeline import (
    TrainingConfig,
    ModeIDModel,
    train_mode_id_model,
    predict_mode_id,
    evaluate_model,
)
from turbomodal.ml.models import TIER_MODELS

__all__ = [
    "FeatureConfig",
    "extract_features",
    "compute_order_spectrum",
    "traveling_wave_decomposition",
    "build_feature_matrix",
    "TrainingConfig",
    "ModeIDModel",
    "train_mode_id_model",
    "predict_mode_id",
    "evaluate_model",
    "TIER_MODELS",
]

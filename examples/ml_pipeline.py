#!/usr/bin/env python3
"""ML pipeline: solve -> signals -> features -> train -> evaluate.

Demonstrates the full Subsystem B (signal synthesis) to Subsystem C
(ML training) workflow.  Generates synthetic training data inline,
extracts features, and trains a mode identification model using the
complexity ladder.

Usage:
    pip install turbomodal[ml]   # ensure ML extras are installed
    python ml_pipeline.py
"""

import numpy as np
from pathlib import Path

import turbomodal as tm
from turbomodal import (
    NoiseConfig,
    SensorArrayConfig,
    SignalGenerationConfig,
    VirtualSensorArray,
    generate_signals_for_condition,
)
from turbomodal.ml import (
    FeatureConfig,
    TrainingConfig,
    extract_features,
    train_mode_id_model,
    evaluate_model,
)

# ---------------------------------------------------------------------------
# 1. Load mesh and solve at multiple RPM values
# ---------------------------------------------------------------------------
TEST_DATA = Path(__file__).resolve().parent.parent / "tests" / "test_data"
MESH_FILE = TEST_DATA / "wedge_sector.msh"
NUM_SECTORS = 24

print("Loading mesh and solving...")
mesh = tm.load_mesh(str(MESH_FILE), num_sectors=NUM_SECTORS)
mat = tm.Material(E=200e9, nu=0.3, rho=7800)

# Solve at a few RPM values to create multi-condition training data
rpm_values = [2000.0, 5000.0, 8000.0, 10000.0]
all_results = {}
for rpm in rpm_values:
    results = tm.solve(mesh, mat, rpm=rpm, num_modes=5, verbose=0)
    all_results[rpm] = results
    n_freqs = sum(len(r.frequencies) for r in results)
    print(f"  RPM={rpm:.0f}: {n_freqs} modes across {len(results)} harmonics")

# ---------------------------------------------------------------------------
# 2. Create sensor array
# ---------------------------------------------------------------------------
print("\nCreating sensor array...")
sensor_config = SensorArrayConfig.default_btt_array(
    num_probes=8,
    casing_radius=0.15,
    axial_positions=[0.0],
    sample_rate=100_000.0,
    duration=0.1,
)
sensor_array = VirtualSensorArray(mesh, sensor_config)
print(f"  {sensor_array.n_sensors} sensors, "
      f"SR={sensor_config.sample_rate/1e3:.0f} kHz, "
      f"dur={sensor_config.duration:.2f} s")

# ---------------------------------------------------------------------------
# 3. Generate synthetic signals for each condition
# ---------------------------------------------------------------------------
print("\nGenerating signals...")
sig_config = SignalGenerationConfig(
    sample_rate=100_000.0,
    duration=0.1,
    amplitude_mode="random",
    amplitude_scale=1e-6,
)
noise_config = NoiseConfig(gaussian_snr_db=30.0)

X_all = []
y_nd_all = []
y_freq_all = []

for rpm, results in all_results.items():
    for trial in range(3):  # 3 noise realisations per condition
        sig = generate_signals_for_condition(
            sensor_array, results, rpm=rpm,
            config=SignalGenerationConfig(
                sample_rate=sig_config.sample_rate,
                duration=sig_config.duration,
                amplitude_mode=sig_config.amplitude_mode,
                amplitude_scale=sig_config.amplitude_scale,
                seed=42 + trial,
            ),
            noise_config=noise_config,
        )

        # Extract features from the signal
        feat_config = FeatureConfig(
            feature_type="spectrogram",
            fft_size=1024,
            hop_size=256,
        )
        features = extract_features(
            sig["signals"], sample_rate=sig_config.sample_rate,
            config=feat_config,
        )
        X_all.append(features)

        # Use the dominant harmonic as the label
        dominant = max(results, key=lambda r: r.frequencies[0])
        y_nd_all.append(dominant.harmonic_index)
        y_freq_all.append(dominant.frequencies[0])

X = np.array(X_all)
y = {
    "nodal_diameter": np.array(y_nd_all),
    "frequency": np.array(y_freq_all),
}

print(f"  Feature matrix: {X.shape}")
print(f"  Labels: {len(y_nd_all)} samples, "
      f"ND range [{min(y_nd_all)}, {max(y_nd_all)}]")

# ---------------------------------------------------------------------------
# 4. Train a model using the complexity ladder
# ---------------------------------------------------------------------------
print("\nTraining model (max tier 2 for speed)...")
train_config = TrainingConfig(
    max_tier=2,                       # stop at tree ensembles
    validation_split=0.2,
    test_split=0.2,
    split_by_condition=False,         # small dataset, random split is fine
    use_optuna=False,                 # skip hyperparameter search for demo
    mode_detection_f1_min=0.80,       # relaxed targets for small dataset
    whirl_accuracy_min=0.80,
    amplitude_mape_max=0.20,
    velocity_r2_min=0.50,
)

best_model, report = train_mode_id_model(X=X, y=y, config=train_config)
print(f"  Best tier: {report.get('best_tier', 'N/A')}")

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
if best_model is not None:
    # Split off a test set for evaluation
    n_test = max(1, len(X) // 5)
    X_test = X[-n_test:]
    y_test = {k: v[-n_test:] for k, v in y.items()}

    metrics = evaluate_model(best_model, X_test, y_test, config=train_config)
    print(f"\n  Test metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
else:
    print("\n  No model met performance targets (expected with tiny dataset)")

# ---------------------------------------------------------------------------
# 6. Predict on new data
# ---------------------------------------------------------------------------
if best_model is not None:
    predictions = best_model.predict(X_test)
    print(f"\n  Predictions keys: {list(predictions.keys())}")
    if "nodal_diameter" in predictions:
        print(f"  Predicted NDs: {predictions['nodal_diameter']}")
        print(f"  Actual NDs:    {y_test['nodal_diameter']}")

print("\nDone. Full ML pipeline: solve -> signals -> features -> train -> evaluate")

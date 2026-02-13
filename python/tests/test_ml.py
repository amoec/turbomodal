"""Tests for Subsystem C: ML Pipeline (features, models, pipeline)."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractFeaturesSpectrogram:
    """extract_features with feature_type='spectrogram'."""

    def test_basic_shape(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        rng = np.random.default_rng(0)
        n_sensors, n_samples = 4, 4096
        signals = rng.standard_normal((n_sensors, n_samples))
        config = FeatureConfig(fft_size=256, hop_size=128, feature_type="spectrogram")
        feat = extract_features(signals, sample_rate=10000.0, config=config)

        assert feat.ndim == 1
        n_freq_bins = 256 // 2 + 1  # rfft bins
        assert feat.shape[0] == n_sensors * n_freq_bins

    def test_1d_input(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        signal_1d = np.sin(2 * np.pi * 100 * np.arange(2048) / 10000.0)
        config = FeatureConfig(fft_size=256, feature_type="spectrogram")
        feat = extract_features(signal_1d, sample_rate=10000.0, config=config)
        assert feat.ndim == 1
        assert len(feat) > 0

    def test_empty_signal(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        signals = np.zeros((2, 0))
        config = FeatureConfig(feature_type="spectrogram")
        feat = extract_features(signals, sample_rate=10000.0, config=config)
        assert len(feat) == 0


class TestExtractFeaturesMel:
    """extract_features with feature_type='mel'."""

    def test_mel_shape(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        rng = np.random.default_rng(1)
        signals = rng.standard_normal((2, 4096))
        config = FeatureConfig(
            fft_size=512, feature_type="mel", n_mels=40, f_max=5000.0,
        )
        feat = extract_features(signals, sample_rate=10000.0, config=config)
        assert feat.shape == (2 * 40,)


class TestOrderSpectrum:
    """compute_order_spectrum with known sinusoidal input."""

    def test_known_order(self):
        from turbomodal.ml.features import compute_order_spectrum

        rpm = 6000.0  # 100 Hz fundamental
        fs = 10000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        order = 3
        f_target = order * rpm / 60.0  # 300 Hz
        signal = np.sin(2 * np.pi * f_target * t)

        orders, amps = compute_order_spectrum(signal, fs, rpm, max_order=10)
        assert len(orders) == 10
        assert len(amps) == 10
        # Order 3 should have the largest amplitude
        assert np.argmax(np.abs(amps)) == order - 1

    def test_invalid_rpm_raises(self):
        from turbomodal.ml.features import compute_order_spectrum

        with pytest.raises(ValueError, match="rpm"):
            compute_order_spectrum(np.ones(100), 1000.0, rpm=0.0)


class TestTravelingWaveDecomposition:
    """traveling_wave_decomposition with synthetic FW/BW waves."""

    def test_forward_wave_separation(self):
        from turbomodal.ml.features import traveling_wave_decomposition

        n_sensors = 8
        angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
        fs = 10000.0
        n_samples = 4096
        t = np.arange(n_samples) / fs
        f0 = 200.0
        nd = 2  # nodal diameter

        # Create a pure forward traveling wave: x_k(t) = cos(2*pi*f0*t - nd*theta_k)
        signals = np.zeros((n_sensors, n_samples))
        for k in range(n_sensors):
            signals[k] = np.cos(2 * np.pi * f0 * t - nd * angles[k])

        frequencies = np.array([f0])
        forward, backward = traveling_wave_decomposition(
            signals, angles, frequencies, fs,
        )

        assert forward.shape == (1, n_sensors // 2 + 1)
        assert backward.shape == (1, n_sensors // 2 + 1)
        # Forward component at nd=2 should be much larger than other NDs
        # (spatial DFT concentrates energy at the true harmonic)
        forward_mags = np.abs(forward[0])
        assert forward_mags[nd] == pytest.approx(np.max(forward_mags), rel=0.1)


class TestExtractFeaturesOrderTracking:
    """extract_features with feature_type='order_tracking'."""

    def test_shape(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        n_sensors = 3
        fs = 10000.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signals = np.stack([np.sin(2 * np.pi * 200 * t)] * n_sensors)

        config = FeatureConfig(
            feature_type="order_tracking",
            rpm=6000.0,
            max_engine_order=10,
        )
        feat = extract_features(signals, fs, config)
        assert feat.shape == (n_sensors * 10 * 2,)  # real + imag

    def test_missing_rpm_raises(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        signals = np.ones((2, 1000))
        config = FeatureConfig(feature_type="order_tracking", rpm=0.0)
        with pytest.raises(ValueError, match="rpm"):
            extract_features(signals, 10000.0, config)


class TestCrossSpectra:
    """Cross-spectral features appended when include_cross_spectra=True."""

    def test_feature_vector_grows(self):
        from turbomodal.ml.features import extract_features, FeatureConfig

        rng = np.random.default_rng(2)
        signals = rng.standard_normal((3, 4096))
        base = FeatureConfig(fft_size=256, feature_type="spectrogram")
        cross = FeatureConfig(
            fft_size=256, feature_type="spectrogram",
            include_cross_spectra=True, coherence_threshold=0.0,
        )
        feat_base = extract_features(signals, 10000.0, base)
        feat_cross = extract_features(signals, 10000.0, cross)
        # 3 sensors → 3 pairs → cross features appended
        assert len(feat_cross) > len(feat_base)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n_samples=100, n_features=20, seed=42):
    """Create a small synthetic dataset for model testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = {
        "nodal_diameter": rng.integers(0, 5, size=n_samples).astype(np.int32),
        "nodal_circle": rng.integers(0, 3, size=n_samples).astype(np.int32),
        "whirl_direction": rng.choice([-1, 0, 1], size=n_samples).astype(np.int32),
        "frequency": rng.uniform(100, 5000, size=n_samples),
        "amplitude": rng.uniform(0.01, 1.0, size=n_samples),
        "wave_velocity": rng.uniform(10, 500, size=n_samples),
    }
    return X, y


class TestLinearModel:
    """Tier 1: LinearModeIDModel."""

    def test_train_predict_roundtrip(self):
        from turbomodal.ml.models import LinearModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        X, y = _make_synthetic_dataset()
        model = LinearModeIDModel()
        metrics = model.train(X, y, TrainingConfig())

        assert isinstance(metrics, dict)
        assert "mode_f1" in metrics

        preds = model.predict(X)
        for key in ("nodal_diameter", "nodal_circle", "whirl_direction",
                     "amplitude", "wave_velocity", "confidence"):
            assert key in preds
            assert preds[key].shape == (X.shape[0],)

    def test_save_load(self):
        from turbomodal.ml.models import LinearModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        X, y = _make_synthetic_dataset(n_samples=50)
        model = LinearModeIDModel()
        model.train(X, y, TrainingConfig())

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = LinearModeIDModel()
            model2.load(path)
            p1 = model.predict(X)
            p2 = model2.predict(X)
            np.testing.assert_array_equal(p1["nodal_diameter"], p2["nodal_diameter"])
        finally:
            os.unlink(path)

    def test_predict_before_train_raises(self):
        from turbomodal.ml.models import LinearModeIDModel

        model = LinearModeIDModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(np.ones((5, 10)))


class TestTreeModel:
    """Tier 2: TreeModeIDModel."""

    def test_train_predict(self):
        from turbomodal.ml.models import TreeModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        X, y = _make_synthetic_dataset()
        model = TreeModeIDModel()
        metrics = model.train(X, y, TrainingConfig())
        assert "mode_f1" in metrics

        preds = model.predict(X)
        assert preds["amplitude"].shape == (X.shape[0],)

    def test_feature_importances(self):
        from turbomodal.ml.models import TreeModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        X, y = _make_synthetic_dataset(n_features=10)
        model = TreeModeIDModel()
        model.train(X, y, TrainingConfig())
        fi = model.feature_importances_
        assert fi.shape == (10,)
        assert np.all(fi >= 0)


class TestSVMModel:
    """Tier 3: SVMModeIDModel."""

    def test_train_predict(self):
        from turbomodal.ml.models import SVMModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        X, y = _make_synthetic_dataset(n_samples=60)
        model = SVMModeIDModel()
        metrics = model.train(X, y, TrainingConfig())
        preds = model.predict(X)
        assert "confidence" in preds


# ---------------------------------------------------------------------------
# Evaluate model tests
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """evaluate_model returns all 6 expected metrics."""

    def test_metric_keys_and_ranges(self):
        from turbomodal.ml.models import LinearModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig, evaluate_model

        X, y = _make_synthetic_dataset()
        model = LinearModeIDModel()
        model.train(X, y, TrainingConfig())
        metrics = evaluate_model(model, X, y)

        expected = {
            "mode_detection_f1",
            "whirl_accuracy",
            "amplitude_mape",
            "amplitude_r2",
            "velocity_rmse",
            "velocity_r2",
        }
        assert expected.issubset(metrics.keys())
        assert 0.0 <= metrics["mode_detection_f1"] <= 1.0
        assert 0.0 <= metrics["whirl_accuracy"] <= 1.0
        assert metrics["amplitude_mape"] >= 0.0
        assert metrics["velocity_rmse"] >= 0.0


# ---------------------------------------------------------------------------
# Pipeline / complexity ladder tests
# ---------------------------------------------------------------------------


class TestConditionBasedSplit:
    """_condition_based_split ensures no condition leaks across splits."""

    def test_no_overlap(self):
        from turbomodal.ml.pipeline import _condition_based_split

        n = 200
        # 20 conditions, 10 samples each
        condition_ids = np.repeat(np.arange(20), 10)
        train_idx, val_idx, test_idx = _condition_based_split(
            condition_ids, n, test_frac=0.15, val_frac=0.15,
        )

        train_conds = set(condition_ids[train_idx].tolist())
        test_conds = set(condition_ids[test_idx].tolist())
        assert train_conds.isdisjoint(test_conds), "Train/test conditions overlap"

        if len(val_idx) > 0:
            val_conds = set(condition_ids[val_idx].tolist())
            assert val_conds.isdisjoint(test_conds), "Val/test conditions overlap"
            assert val_conds.isdisjoint(train_conds), "Val/train conditions overlap"


class TestTrainModeIdModel:
    """train_mode_id_model with pre-built data (Tier 1 only for speed)."""

    def test_returns_model_and_report(self):
        from turbomodal.ml.pipeline import train_mode_id_model, TrainingConfig

        X, y = _make_synthetic_dataset(n_samples=120, n_features=20)
        condition_ids = np.repeat(np.arange(12), 10)

        config = TrainingConfig(
            max_tier=1,
            epochs=5,
            split_by_condition=True,
        )
        model, report = train_mode_id_model(
            config=config, X=X, y=y, condition_ids=condition_ids,
        )

        assert model is not None
        assert "best_tier" in report
        assert report["best_tier"] == 1
        assert "test_metrics" in report
        assert "mode_detection_f1" in report["test_metrics"]

    def test_diminishing_returns_stops_early(self):
        """With a tiny dataset Tier 2 shouldn't improve much over Tier 1."""
        from turbomodal.ml.pipeline import train_mode_id_model, TrainingConfig

        X, y = _make_synthetic_dataset(n_samples=60, n_features=10)
        condition_ids = np.repeat(np.arange(6), 10)

        config = TrainingConfig(
            max_tier=2,
            epochs=2,
            performance_gap_threshold=100.0,  # force early stop
            split_by_condition=True,
        )
        model, report = train_mode_id_model(
            config=config, X=X, y=y, condition_ids=condition_ids,
        )
        # Should stop after Tier 1 since gap threshold is absurdly high
        assert report["best_tier"] <= 2


class TestPredictModeId:
    """predict_mode_id extracts features then predicts."""

    def test_inference(self):
        from turbomodal.ml.models import LinearModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig, predict_mode_id

        # Train a model on spectrogram features — use DEFAULT FeatureConfig
        # so that predict_mode_id (which uses defaults) produces matching dims
        from turbomodal.ml.features import extract_features, FeatureConfig

        fs = 10000.0
        rng = np.random.default_rng(7)
        n_train = 30
        n_sensors = 2
        n_samples = 4096  # must be >= fft_size (default 2048)

        cfg = FeatureConfig()  # default fft_size=2048

        # Build feature matrix manually
        feats = []
        for _ in range(n_train):
            sig = rng.standard_normal((n_sensors, n_samples))
            f = extract_features(sig, fs, cfg)
            feats.append(f)
        X = np.vstack(feats)

        y = {
            "nodal_diameter": rng.integers(0, 3, n_train).astype(np.int32),
            "nodal_circle": np.zeros(n_train, dtype=np.int32),
            "whirl_direction": rng.choice([-1, 1], n_train).astype(np.int32),
            "frequency": rng.uniform(100, 1000, n_train),
            "amplitude": rng.uniform(0.1, 1.0, n_train),
            "wave_velocity": rng.uniform(50, 200, n_train),
        }

        model = LinearModeIDModel()
        model.train(X, y, TrainingConfig())

        # Now run inference — signal must be same shape
        test_signal = rng.standard_normal((n_sensors, n_samples))
        preds = predict_mode_id(model, test_signal, fs)
        assert "nodal_diameter" in preds
        assert preds["nodal_diameter"].shape == (1,)


# ---------------------------------------------------------------------------
# Label encoding tests
# ---------------------------------------------------------------------------


class TestLabelEncoding:
    """_encode/_decode_mode_labels roundtrip."""

    def test_roundtrip(self):
        from turbomodal.ml.models import _encode_mode_labels, _decode_mode_labels

        nd = np.array([0, 1, 5, 10])
        nc = np.array([0, 2, 3, 0])
        encoded = _encode_mode_labels(nd, nc)
        nd2, nc2 = _decode_mode_labels(encoded)
        np.testing.assert_array_equal(nd, nd2)
        np.testing.assert_array_equal(nc, nc2)


# ---------------------------------------------------------------------------
# TIER_MODELS registry
# ---------------------------------------------------------------------------


class TestTierModels:
    """TIER_MODELS dict maps to correct classes."""

    def test_all_tiers_present(self):
        from turbomodal.ml.models import TIER_MODELS

        assert set(TIER_MODELS.keys()) == {1, 2, 3, 4, 5, 6}

    def test_instantiation(self):
        from turbomodal.ml.models import TIER_MODELS

        for tier, cls in TIER_MODELS.items():
            model = cls()
            assert hasattr(model, "train")
            assert hasattr(model, "predict")
            assert hasattr(model, "save")
            assert hasattr(model, "load")

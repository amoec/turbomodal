"""Tests for Subsystem D: Sensor Optimization & Explainability."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fisher Information Matrix
# ---------------------------------------------------------------------------


class TestComputeFisherInformation:
    """compute_fisher_information correctness checks."""

    def test_shape_and_symmetry(self):
        from turbomodal.optimization.sensor_placement import compute_fisher_information

        n_modes, n_dof = 3, 20
        rng = np.random.default_rng(0)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j

        # Pass interpolation matrix directly (n_sensors, n_dof)
        n_sensors = 8
        H = np.zeros((n_sensors, n_dof))
        for i in range(n_sensors):
            H[i, i * (n_dof // n_sensors) % n_dof] = 1.0

        FIM = compute_fisher_information(mode_shapes, H)
        assert FIM.shape == (n_modes, n_modes)
        # FIM should be symmetric (Hermitian real)
        np.testing.assert_allclose(FIM, FIM.T, atol=1e-12)

    def test_positive_semidefinite(self):
        from turbomodal.optimization.sensor_placement import compute_fisher_information

        n_modes, n_dof = 4, 30
        rng = np.random.default_rng(1)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j
        H = np.eye(10, n_dof)

        FIM = compute_fisher_information(mode_shapes, H)
        eigenvalues = np.linalg.eigvalsh(FIM)
        assert np.all(eigenvalues >= -1e-12)  # PSD

    def test_identity_noise(self):
        from turbomodal.optimization.sensor_placement import compute_fisher_information

        n_modes, n_dof = 2, 10
        rng = np.random.default_rng(2)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j
        H = np.eye(5, n_dof)

        FIM_default = compute_fisher_information(mode_shapes, H)
        FIM_identity = compute_fisher_information(mode_shapes, H, np.eye(5))
        np.testing.assert_allclose(FIM_default, FIM_identity, atol=1e-12)

    def test_position_based_input(self):
        """When sensor_positions is (n, 3) it should build H automatically."""
        from turbomodal.optimization.sensor_placement import compute_fisher_information

        n_modes, n_dof = 2, 50
        rng = np.random.default_rng(3)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j

        # 4 sensors at different angular positions
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        positions = np.column_stack([
            0.5 * np.cos(angles), 0.5 * np.sin(angles), np.zeros(4),
        ])
        FIM = compute_fisher_information(mode_shapes, positions)
        assert FIM.shape == (n_modes, n_modes)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestComputeObservability:
    """compute_observability returns expected keys and shapes."""

    def test_keys_and_shapes(self):
        from turbomodal.optimization.sensor_placement import compute_observability

        n_modes, n_dof = 3, 20
        rng = np.random.default_rng(4)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j
        H = np.eye(6, n_dof)

        obs = compute_observability(mode_shapes, H)
        assert "condition_number" in obs
        assert "min_singular_value" in obs
        assert "mac_matrix" in obs
        assert obs["mac_matrix"].shape == (n_modes, n_modes)

    def test_mac_diagonal_is_one(self):
        from turbomodal.optimization.sensor_placement import compute_observability

        n_modes, n_dof = 2, 15
        rng = np.random.default_rng(5)
        mode_shapes = rng.standard_normal((n_modes, n_dof)) + 0j
        H = np.eye(8, n_dof)

        obs = compute_observability(mode_shapes, H)
        np.testing.assert_allclose(
            np.diag(obs["mac_matrix"]), 1.0, atol=1e-10,
        )

    def test_condition_number_positive(self):
        from turbomodal.optimization.sensor_placement import compute_observability

        mode_shapes = np.eye(3, 10, dtype=np.complex128)
        H = np.eye(5, 10)
        obs = compute_observability(mode_shapes, H)
        assert obs["condition_number"] >= 1.0


# ---------------------------------------------------------------------------
# Greedy sensor selection
# ---------------------------------------------------------------------------


class TestOptimizeSensorPlacement:
    """optimize_sensor_placement smoke test with small problem."""

    def test_greedy_monotonic_objective(self):
        from turbomodal.optimization.sensor_placement import (
            optimize_sensor_placement,
            SensorOptimizationConfig,
        )

        n_modes, n_dof = 3, 30
        rng = np.random.default_rng(6)
        mode_shapes = rng.standard_normal((n_dof, n_modes)) + 0j  # (n_dof, n_modes)
        freqs = np.array([100.0, 200.0, 300.0])

        class _FakeModalResult:
            def __init__(self, shapes, f):
                self.mode_shapes = shapes
                self.frequencies = f

        modal_results = [_FakeModalResult(mode_shapes, freqs)]

        config = SensorOptimizationConfig(
            max_sensors=5,
            min_sensors=2,
            optimization_method="greedy",
            min_angular_spacing=10.0,
            feasible_radii=(0.4, 0.6),
            robustness_trials=0,
        )
        result = optimize_sensor_placement(None, modal_results, config)

        assert result.num_sensors > 0
        assert result.num_sensors <= config.max_sensors
        assert result.sensor_positions.shape[0] == result.num_sensors

        # Objective history should be monotonically non-decreasing
        if len(result.objective_history) > 1:
            diffs = np.diff(result.objective_history)
            assert np.all(diffs >= -1e-10), "Objective not monotonic"

    def test_empty_modal_results(self):
        from turbomodal.optimization.sensor_placement import (
            optimize_sensor_placement,
            SensorOptimizationConfig,
        )

        result = optimize_sensor_placement(None, [], SensorOptimizationConfig())
        assert result.num_sensors == 0


# ---------------------------------------------------------------------------
# Physics consistency check
# ---------------------------------------------------------------------------


class TestPhysicsConsistencyCheck:
    """physics_consistency_check validates predictions against constraints."""

    def test_valid_predictions_pass(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        # Use physically consistent velocities: v = 2*pi*f*R / ND
        R = 0.3  # default blade_radius
        nd_arr = np.array([1, 2, 3])
        freq_arr = np.array([500.0, 1000.0, 1500.0])
        vel_arr = 2 * np.pi * freq_arr * R / nd_arr

        preds = {
            "frequency": freq_arr,
            "nodal_diameter": nd_arr,
            "whirl_direction": np.array([1, -1, 0]),
            "wave_velocity": vel_arr,
        }
        # rpm=0 skips whirl ordering check
        result = physics_consistency_check(preds, num_sectors=24, rpm=0.0)
        assert result["is_consistent"].all()
        assert len(result["violations"]) == 3
        assert all(len(v) == 0 for v in result["violations"])

    def test_invalid_nd_flagged(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        preds = {
            "frequency": np.array([500.0]),
            "nodal_diameter": np.array([99]),  # out of range for 24 sectors
            "whirl_direction": np.array([1]),
        }
        result = physics_consistency_check(preds, num_sectors=24)
        assert not result["is_consistent"][0]
        assert any("ND=" in v for v in result["violations"][0])

    def test_negative_frequency_flagged(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        preds = {
            "frequency": np.array([-10.0]),
            "nodal_diameter": np.array([1]),
            "whirl_direction": np.array([1]),
        }
        result = physics_consistency_check(preds, num_sectors=24)
        assert not result["is_consistent"][0]

    def test_empty_predictions(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        result = physics_consistency_check({}, num_sectors=24)
        assert len(result["is_consistent"]) == 0

    def test_whirl_ordering_check(self):
        """Forward whirl freq should be >= backward whirl freq for same ND."""
        from turbomodal.optimization.explainability import physics_consistency_check

        # ND=2: forward=1000 Hz, backward=2000 Hz → violation
        preds = {
            "frequency": np.array([1000.0, 2000.0]),
            "nodal_diameter": np.array([2, 2]),
            "whirl_direction": np.array([1, -1]),  # 1=FW, -1=BW
            "wave_velocity": np.zeros(2),
        }
        result = physics_consistency_check(preds, num_sectors=24, rpm=6000.0)
        # At least one should have a whirl ordering violation
        all_violations = [v for vlist in result["violations"] for v in vlist]
        has_whirl_violation = any("hirl" in v.lower() or "order" in v.lower()
                                  for v in all_violations)
        assert has_whirl_violation


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


class TestCalibrateConfidence:
    """calibrate_confidence wraps a model with calibrated confidence."""

    def _make_fitted_model(self):
        """Return a small fitted LinearModeIDModel."""
        from turbomodal.ml.models import LinearModeIDModel
        from turbomodal.ml.pipeline import TrainingConfig

        rng = np.random.default_rng(10)
        X = rng.standard_normal((60, 10))
        y = {
            "nodal_diameter": rng.integers(0, 3, 60).astype(np.int32),
            "nodal_circle": np.zeros(60, dtype=np.int32),
            "whirl_direction": rng.choice([-1, 1], 60).astype(np.int32),
            "frequency": rng.uniform(100, 1000, 60),
            "amplitude": rng.uniform(0.1, 1.0, 60),
            "wave_velocity": rng.uniform(50, 200, 60),
        }
        model = LinearModeIDModel()
        model.train(X, y, TrainingConfig())
        return model, X, y

    def test_platt(self):
        from turbomodal.optimization.explainability import calibrate_confidence

        model, X, y = self._make_fitted_model()
        cal_model = calibrate_confidence(model, X, y, method="platt")
        preds = cal_model.predict(X)
        assert "confidence" in preds
        assert np.all(preds["confidence"] >= 0.0)
        assert np.all(preds["confidence"] <= 1.0)

    def test_isotonic(self):
        from turbomodal.optimization.explainability import calibrate_confidence

        model, X, y = self._make_fitted_model()
        cal_model = calibrate_confidence(model, X, y, method="isotonic")
        preds = cal_model.predict(X)
        assert "confidence" in preds

    def test_temperature(self):
        from turbomodal.optimization.explainability import calibrate_confidence

        model, X, y = self._make_fitted_model()
        cal_model = calibrate_confidence(model, X, y, method="temperature")
        preds = cal_model.predict(X)
        assert np.all(preds["confidence"] >= 0.0)

    def test_conformal(self):
        from turbomodal.optimization.explainability import calibrate_confidence

        model, X, y = self._make_fitted_model()
        cal_model = calibrate_confidence(model, X, y, method="conformal")
        preds = cal_model.predict(X)
        # Conformal adds prediction interval keys
        assert "prediction_interval_lower" in preds or "confidence" in preds

    def test_invalid_method(self):
        from turbomodal.optimization.explainability import calibrate_confidence

        model, X, y = self._make_fitted_model()
        with pytest.raises(ValueError, match="[Uu]nknown|[Uu]nsupported"):
            calibrate_confidence(model, X, y, method="invalid_method")


# ---------------------------------------------------------------------------
# CalibratedModel wrapper
# ---------------------------------------------------------------------------


class TestCalibratedModel:
    """CalibratedModel preserves base model interface."""

    def test_delegates_predict(self):
        from turbomodal.optimization.explainability import CalibratedModel

        class _FakeModel:
            def predict(self, X):
                return {"confidence": np.ones(X.shape[0]) * 0.5,
                        "nodal_diameter": np.zeros(X.shape[0])}
            def save(self, path): pass
            def load(self, path): pass
            def train(self, X, y, config): return {}

        cal = CalibratedModel(
            _FakeModel(),
            calibration_transform=lambda c: c * 2.0,
            method="test",
        )
        preds = cal.predict(np.ones((3, 5)))
        np.testing.assert_allclose(preds["confidence"], 1.0)


# ---------------------------------------------------------------------------
# D1: Minimize sensors mode
# ---------------------------------------------------------------------------


class TestMinimizeSensorsMode:
    """optimize_sensor_placement with mode='minimize_sensors' (D1)."""

    def test_minimize_finds_fewer_sensors(self):
        from turbomodal.optimization.sensor_placement import (
            optimize_sensor_placement,
            SensorOptimizationConfig,
        )

        n_modes, n_dof = 3, 30
        rng = np.random.default_rng(10)
        mode_shapes = rng.standard_normal((n_dof, n_modes)) + 0j
        freqs = np.array([100.0, 200.0, 300.0])

        class _FakeModalResult:
            def __init__(self, shapes, f):
                self.mode_shapes = shapes
                self.frequencies = f

        modal_results = [_FakeModalResult(mode_shapes, freqs)]

        config = SensorOptimizationConfig(
            max_sensors=8,
            min_sensors=2,
            mode="minimize_sensors",
            optimization_method="greedy",
            min_angular_spacing=10.0,
            feasible_radii=(0.4, 0.6),
            robustness_trials=0,
        )
        result = optimize_sensor_placement(None, modal_results, config)
        assert result.num_sensors > 0
        assert result.num_sensors <= config.max_sensors


# ---------------------------------------------------------------------------
# D2: ML model factory in greedy selection
# ---------------------------------------------------------------------------


class TestMLModelFactory:
    """Greedy selection with ml_model_factory callable (D2)."""

    def test_greedy_uses_ml_factory(self):
        from turbomodal.optimization.sensor_placement import (
            optimize_sensor_placement,
            SensorOptimizationConfig,
        )

        n_modes, n_dof = 3, 30
        rng = np.random.default_rng(11)
        mode_shapes = rng.standard_normal((n_dof, n_modes)) + 0j
        freqs = np.array([100.0, 200.0, 300.0])

        class _FakeModalResult:
            def __init__(self, shapes, f):
                self.mode_shapes = shapes
                self.frequencies = f

        modal_results = [_FakeModalResult(mode_shapes, freqs)]

        call_count = [0]

        def ml_factory(positions):
            call_count[0] += 1
            return float(len(positions))  # score = number of sensors

        config = SensorOptimizationConfig(
            max_sensors=5,
            min_sensors=2,
            optimization_method="greedy",
            min_angular_spacing=10.0,
            feasible_radii=(0.4, 0.6),
            robustness_trials=0,
        )
        result = optimize_sensor_placement(None, modal_results, config,
                                           ml_model_factory=ml_factory)
        assert result.num_sensors > 0
        # The factory should have been called at least once
        assert call_count[0] > 0


# ---------------------------------------------------------------------------
# D3: Observability penalty in Bayesian refinement
# ---------------------------------------------------------------------------


class TestObservabilityPenalty:
    """observability_penalty_weight is respected in config (D3)."""

    def test_penalty_weight_in_config(self):
        from turbomodal.optimization.sensor_placement import SensorOptimizationConfig

        config = SensorOptimizationConfig(observability_penalty_weight=0.5)
        assert config.observability_penalty_weight == 0.5

    def test_zero_penalty_default(self):
        """Default penalty weight is 0.1."""
        from turbomodal.optimization.sensor_placement import SensorOptimizationConfig

        config = SensorOptimizationConfig()
        assert config.observability_penalty_weight == 0.1


# ---------------------------------------------------------------------------
# D8: Sensor contribution heatmap
# ---------------------------------------------------------------------------


class TestPlotSensorContribution:
    """plot_sensor_contribution creates a heatmap figure (D8)."""

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from turbomodal.viz import plot_sensor_contribution

        rng = np.random.default_rng(20)
        n_samples, n_features, n_outputs = 10, 12, 4
        shap_values = rng.standard_normal((n_samples, n_features, n_outputs))
        sensor_names = ["S1", "S2", "S3"]
        mode_names = ["ND", "Whirl", "Amp", "Vel"]
        features_per_sensor = 4

        fig = plot_sensor_contribution(shap_values, sensor_names, mode_names,
                                       features_per_sensor)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_2d_shap_input(self):
        """When shap_values is 2D, it should be auto-expanded."""
        import matplotlib
        matplotlib.use("Agg")
        from turbomodal.viz import plot_sensor_contribution

        rng = np.random.default_rng(21)
        shap_values = rng.standard_normal((10, 8))  # 2D
        sensor_names = ["S1", "S2"]
        mode_names = ["Output"]
        features_per_sensor = 4

        fig = plot_sensor_contribution(shap_values, sensor_names, mode_names,
                                       features_per_sensor)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# D9: Model selection report
# ---------------------------------------------------------------------------


class TestModelSelectionReport:
    """generate_model_selection_report returns structured output (D9)."""

    def test_report_structure(self):
        from turbomodal.optimization.explainability import generate_model_selection_report

        training_report = {
            "best_tier": 3,
            "val_metrics": {
                "mode_detection_f1": 0.95,
                "whirl_accuracy": 0.97,
                "amplitude_mape": 0.05,
                "velocity_r2": 0.94,
            },
            "tier_history": [
                {"tier": 1, "composite_score": 0.7,
                 "val_metrics": {"mode_detection_f1": 0.80}},
                {"tier": 2, "composite_score": 0.85,
                 "val_metrics": {"mode_detection_f1": 0.90}},
                {"tier": 3, "composite_score": 0.93,
                 "val_metrics": {"mode_detection_f1": 0.95}},
            ],
        }

        report = generate_model_selection_report(training_report)
        assert "summary" in report
        assert "per_tier_metrics" in report
        assert "gap_analysis" in report
        assert "selected_tier" in report
        assert report["selected_tier"] == 3
        assert len(report["gap_analysis"]) == 2  # 2 transitions

    def test_empty_tier_history(self):
        from turbomodal.optimization.explainability import generate_model_selection_report

        report = generate_model_selection_report({
            "best_tier": 0, "val_metrics": {}, "tier_history": []
        })
        assert report["selected_tier"] == 0
        assert len(report["gap_analysis"]) == 0


# ---------------------------------------------------------------------------
# D10: Explanation card
# ---------------------------------------------------------------------------


class TestExplanationCard:
    """generate_explanation_card returns structured output (D10)."""

    def test_card_structure(self):
        from turbomodal.optimization.explainability import generate_explanation_card

        # Use a minimal fake model
        class _FakeModel:
            _is_fitted = True
            def predict(self, X):
                n = X.shape[0]
                return {
                    "nodal_diameter": np.array([2] * n),
                    "whirl_direction": np.array([1] * n),
                    "amplitude": np.array([0.5] * n),
                    "wave_velocity": np.array([100.0] * n),
                    "confidence": np.array([0.9] * n),
                }

        model = _FakeModel()
        X = np.random.default_rng(30).standard_normal((1, 10))
        preds = model.predict(X)

        card = generate_explanation_card(
            model, X, preds, sample_idx=0,
            num_sectors=24, rpm=5000.0,
        )
        assert "predicted_values" in card
        assert "confidence" in card
        assert "physics_check" in card
        assert "anomaly_flag" in card
        assert "explanation_text" in card
        assert isinstance(card["explanation_text"], str)

    def test_card_with_uncertainty(self):
        from turbomodal.optimization.explainability import generate_explanation_card

        class _FakeModel:
            _is_fitted = True
            def predict(self, X):
                n = X.shape[0]
                return {
                    "nodal_diameter": np.array([1] * n),
                    "whirl_direction": np.array([-1] * n),
                    "amplitude": np.array([0.3] * n),
                    "wave_velocity": np.array([50.0] * n),
                    "confidence": np.array([0.8] * n),
                }

        model = _FakeModel()
        X = np.random.default_rng(31).standard_normal((1, 10))
        preds = model.predict(X)

        uncertainty = {
            "amplitude_total_var": np.array([0.01]),
            "velocity_total_var": np.array([4.0]),
            "amplitude_epistemic_var": np.array([0.005]),
        }

        card = generate_explanation_card(
            model, X, preds, sample_idx=0,
            num_sectors=24, rpm=3000.0,
            uncertainty=uncertainty,
        )
        assert card["confidence_interval"] is not None
        assert "amplitude_pm" in card["confidence_interval"]
        assert "velocity_pm" in card["confidence_interval"]


# ---------------------------------------------------------------------------
# D11: Confidence bands and crossing markers in Campbell
# ---------------------------------------------------------------------------


class TestCampbellConfidenceBands:
    """plot_campbell with confidence_bands and crossing_markers (D11)."""

    def test_confidence_bands_accepted(self):
        """plot_campbell accepts confidence_bands kwarg without error."""
        import matplotlib
        matplotlib.use("Agg")
        from turbomodal.viz import plot_campbell
        from turbomodal._core import ModalResult

        # Minimal 2-RPM, 1-harmonic sweep
        results = []
        for rpm_val in [3000.0, 6000.0]:
            mr = ModalResult()
            mr.rpm = rpm_val
            mr.harmonic_index = 0
            mr.frequencies = np.array([500.0 + rpm_val * 0.01])
            mr.whirl_direction = np.array([1])
            mode_shapes = np.zeros((10, 1), dtype=np.complex128)
            mode_shapes[:, 0] = 1.0
            mr.mode_shapes = mode_shapes
            results.append([mr])

        fig = plot_campbell(results, engine_orders=[1],
                            confidence_bands=None, crossing_markers=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_crossing_markers_accepted(self):
        """plot_campbell with crossing_markers=True runs without error."""
        import matplotlib
        matplotlib.use("Agg")
        from turbomodal.viz import plot_campbell
        from turbomodal._core import ModalResult

        results = []
        for rpm_val in [3000.0, 6000.0, 9000.0]:
            mr = ModalResult()
            mr.rpm = rpm_val
            mr.harmonic_index = 1
            mr.frequencies = np.array([rpm_val * 0.02])
            mr.whirl_direction = np.array([1])
            mode_shapes = np.zeros((10, 1), dtype=np.complex128)
            mode_shapes[:, 0] = 1.0
            mr.mode_shapes = mode_shapes
            results.append([mr])

        fig = plot_campbell(results, engine_orders=[1],
                            crossing_markers=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# D12: Epistemic uncertainty in physics check
# ---------------------------------------------------------------------------


class TestEpistemicAnomalyFlag:
    """physics_consistency_check flags high epistemic uncertainty (D12)."""

    def test_high_epistemic_flagged(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        preds = {
            "frequency": np.array([500.0]),
            "nodal_diameter": np.array([2]),
            "whirl_direction": np.array([1]),
        }
        result = physics_consistency_check(
            preds, num_sectors=24,
            epistemic_uncertainty=np.array([0.5]),
            epistemic_threshold=0.1,
        )
        # High uncertainty should cause a violation
        assert not result["is_consistent"][0]
        all_v = [v for vlist in result["violations"] for v in vlist]
        assert any("epistemic" in v.lower() for v in all_v)

    def test_low_epistemic_passes(self):
        from turbomodal.optimization.explainability import physics_consistency_check

        # Use ND=0 to avoid velocity consistency check (only triggers for ND > 0)
        preds = {
            "frequency": np.array([500.0]),
            "nodal_diameter": np.array([0]),
            "whirl_direction": np.array([1]),
        }
        result = physics_consistency_check(
            preds, num_sectors=24,
            epistemic_uncertainty=np.array([0.01]),
            epistemic_threshold=0.1,
        )
        assert result["is_consistent"][0]

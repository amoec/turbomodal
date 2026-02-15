"""Tests for turbomodal.signal_gen module."""

import numpy as np
import pytest

from turbomodal.signal_gen import SignalGenerationConfig


# ---- SignalGenerationConfig defaults ----

def test_signal_generation_config_defaults():
    cfg = SignalGenerationConfig()
    assert cfg.sample_rate == 100000.0
    assert cfg.duration == 1.0
    assert cfg.num_revolutions == 0
    assert cfg.seed == 42
    assert cfg.amplitude_mode == "unit"
    assert cfg.amplitude_scale == 1e-6
    assert cfg.max_frequency == 0.0
    assert cfg.max_modes_per_harmonic == 0


# ---- Signal generation (requires solved results + sensors) ----

@pytest.fixture(scope="module")
def signal_gen_setup(wedge_mesh_path):
    """Set up mesh, solve, and sensor array for signal generation tests."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver
    from turbomodal.sensors import SensorArrayConfig, VirtualSensorArray

    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    mat = Material(200e9, 0.3, 7800)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(3000.0, 3)

    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
        sample_rate=100_000.0,
        duration=0.01,  # short for test speed
    )
    vsa = VirtualSensorArray(mesh, cfg)
    return vsa, results


def test_generate_signals_returns_dict(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01)
    output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    assert isinstance(output, dict)
    assert "signals" in output
    assert "time" in output


def test_generate_signals_shape(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01)
    output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    signals = output["signals"]
    n_samples = int(cfg.sample_rate * cfg.duration)
    assert signals.shape[0] == vsa.n_sensors
    assert signals.shape[1] == n_samples


def test_generate_signals_time_array(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01)
    output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    t = output["time"]
    n_samples = int(cfg.sample_rate * cfg.duration)
    assert len(t) == n_samples
    np.testing.assert_allclose(t[-1], (n_samples - 1) / cfg.sample_rate,
                               rtol=1e-10)


def test_generate_signals_no_noise(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01)
    output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    # Without noise config, signals should equal clean_signals
    if "clean_signals" in output:
        np.testing.assert_array_equal(output["signals"], output["clean_signals"])


def test_generate_signals_with_noise(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition
    from turbomodal.noise import NoiseConfig

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01)
    noise = NoiseConfig(gaussian_snr_db=20.0)
    output = generate_signals_for_condition(
        vsa, results, 3000.0, cfg, noise_config=noise)
    # With noise, signals should differ from clean
    if "clean_signals" in output:
        assert not np.allclose(output["signals"], output["clean_signals"])


def test_generate_signals_reproducible(signal_gen_setup):
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01, seed=42)
    out1 = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    out2 = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    np.testing.assert_array_equal(out1["signals"], out2["signals"])

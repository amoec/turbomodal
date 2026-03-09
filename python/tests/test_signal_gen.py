"""Tests for turbomodal.signal_gen module."""

import numpy as np
import pytest

from turbomodal.signal_gen import (
    SignalGenerationConfig,
    _build_time_vector,
    _sensor_is_stationary,
    filter_modal_results,
)


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
    assert cfg.time is None
    assert cfg.t_start == 0.0
    assert cfg.t_end == 0.0
    assert cfg.damping_ratio == 0.0


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


# ---- _build_time_vector ----

def test_build_time_vector_custom_array():
    """Custom time array should be used verbatim."""
    t_custom = np.array([0.0, 0.1, 0.2, 0.5])
    cfg = SignalGenerationConfig(time=t_custom)
    t = _build_time_vector(cfg, 3000.0)
    np.testing.assert_array_equal(t, t_custom)


def test_build_time_vector_t_start_t_end():
    """t_start/t_end should override duration."""
    cfg = SignalGenerationConfig(sample_rate=1000.0, t_start=1.0, t_end=1.5)
    t = _build_time_vector(cfg, 3000.0)
    assert t[0] == pytest.approx(1.0)
    assert len(t) == 500
    assert t[-1] == pytest.approx(1.0 + 499 / 1000.0)


def test_build_time_vector_num_revolutions():
    """num_revolutions at 6000 RPM: 1 rev = 0.01 s."""
    cfg = SignalGenerationConfig(sample_rate=100_000.0, num_revolutions=2)
    t = _build_time_vector(cfg, 6000.0)
    expected_duration = 2 * 60.0 / 6000.0  # 0.02 s
    assert len(t) == int(expected_duration * 100_000)


def test_build_time_vector_duration_fallback():
    """Default duration is used when no overrides set."""
    cfg = SignalGenerationConfig(sample_rate=1000.0, duration=0.5)
    t = _build_time_vector(cfg, 3000.0)
    assert len(t) == 500


# ---- _sensor_is_stationary ----

def test_sensor_is_stationary_explicit():
    """Explicit is_stationary flag should be respected."""
    from turbomodal.sensors import SensorLocation, SensorType
    loc = SensorLocation(
        sensor_type=SensorType.DISPLACEMENT,
        position=np.zeros(3),
        direction=np.array([0, 0, 1.0]),
        is_stationary=True,
    )
    assert _sensor_is_stationary(loc) is True


def test_sensor_is_stationary_inferred():
    """Only strain gauges are rotating; all others are stationary."""
    from turbomodal.sensors import SensorLocation, SensorType
    pos = np.zeros(3)
    d = np.array([1, 0, 0.0])
    assert _sensor_is_stationary(SensorLocation(SensorType.BTT_PROBE, pos, d)) is True
    assert _sensor_is_stationary(SensorLocation(SensorType.CASING_ACCELEROMETER, pos, d)) is True
    assert _sensor_is_stationary(SensorLocation(SensorType.DISPLACEMENT, pos, d)) is True
    assert _sensor_is_stationary(SensorLocation(SensorType.STRAIN_GAUGE, pos, d)) is False


# ---- Circumferential phase tests ----

def test_two_btt_probes_phase_difference(signal_gen_setup):
    """Two BTT probes at different angles should have a phase shift of k*Δθ."""
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    # results is a list of ModalResult; use only k>0 modes
    k_results = [r for r in results if r.harmonic_index > 0]
    if not k_results:
        pytest.skip("No k>0 modes in test results")

    cfg = SignalGenerationConfig(
        sample_rate=100_000.0, duration=0.05, seed=99,
        max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa, k_results[:1], 3000.0, cfg)
    sig = out["signals"]

    # The 4 BTT probes are at 0°, 90°, 180°, 270°
    # For harmonic k, the circumferential phase difference between
    # adjacent probes is k * π/2.  Verify via cross-correlation.
    if sig.shape[0] >= 2 and np.any(sig[0] != 0) and np.any(sig[1] != 0):
        # Cross-correlate probes 0 and 1 — peak should be shifted
        corr = np.correlate(sig[0], sig[1], mode='full')
        peak = np.argmax(np.abs(corr))
        # Just verify the peak is NOT at zero lag (i.e. there IS a phase shift)
        center = len(sig[0]) - 1
        assert peak != center, "No phase shift detected between probes at different angles"


def test_k0_mode_no_circumferential_phase(signal_gen_setup):
    """k=0 mode: all sensors see the same frequency (no angular splitting)."""
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    k0_results = [r for r in results if r.harmonic_index == 0]
    if not k0_results:
        pytest.skip("No k=0 modes in test results")

    cfg = SignalGenerationConfig(
        sample_rate=100_000.0, duration=0.05, seed=99,
        max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa, k0_results[:1], 3000.0, cfg)
    sig = out["signals"]
    t = out["time"]

    # All sensors with non-zero signal should see the same dominant frequency
    freqs_fft = np.fft.rfftfreq(len(t), d=1.0 / cfg.sample_rate)
    dominant_freqs = []
    for s in range(sig.shape[0]):
        if np.max(np.abs(sig[s])) < 1e-15:
            continue
        spectrum = np.abs(np.fft.rfft(sig[s]))
        dominant_freqs.append(freqs_fft[np.argmax(spectrum[1:]) + 1])

    if len(dominant_freqs) >= 2:
        # All dominant frequencies should be the same
        np.testing.assert_allclose(
            dominant_freqs, dominant_freqs[0], atol=freqs_fft[1],
        )


def test_strain_gauge_rotating_frame_frequency(signal_gen_setup):
    """Strain gauge should see the rotating-frame frequency, not stationary."""
    from turbomodal.signal_gen import generate_signals_for_condition
    from turbomodal.sensors import SensorArrayConfig, SensorType, VirtualSensorArray

    vsa_btt, results = signal_gen_setup
    mesh = vsa_btt.mesh

    # Place a strain gauge at a node position on sector 0
    nodes = np.asarray(mesh.nodes)
    mid_node = nodes[nodes.shape[0] // 2]
    cfg_sg = SensorArrayConfig.default_strain_gauge_array(
        num_gauges=1, radial_positions=[np.linalg.norm(mid_node[:2])],
        sample_rate=100_000.0, duration=0.1,
    )
    vsa_sg = VirtualSensorArray(mesh, cfg_sg)

    # Use a single k>0 mode
    k_results = [r for r in results if r.harmonic_index > 0]
    if not k_results:
        pytest.skip("No k>0 modes")

    rpm = 3000.0
    gen_cfg = SignalGenerationConfig(
        sample_rate=100_000.0, duration=0.1, seed=42,
        max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa_sg, k_results[:1], rpm, gen_cfg)
    sig = out["signals"][0]
    t = out["time"]

    if np.max(np.abs(sig)) < 1e-15:
        pytest.skip("Zero signal at strain gauge location")

    # FFT to find dominant frequency
    freqs_fft = np.fft.rfftfreq(len(t), d=1.0 / gen_cfg.sample_rate)
    spectrum = np.abs(np.fft.rfft(sig))
    dominant_freq = freqs_fft[np.argmax(spectrum[1:]) + 1]

    # The strain gauge (rotating) should see f_rot, not f_stat
    f_rot = k_results[0].frequencies[0]
    # Allow some FFT resolution tolerance
    df = freqs_fft[1] - freqs_fft[0]
    assert abs(dominant_freq - f_rot) < 3 * df, (
        f"Strain gauge dominant freq {dominant_freq:.1f} != rotating freq {f_rot:.1f}"
    )


def test_damping_envelope(signal_gen_setup):
    """Signal with damping should decay exponentially."""
    from turbomodal.signal_gen import generate_signals_for_condition
    from turbomodal.sensors import SensorArrayConfig, VirtualSensorArray

    vsa_btt, results = signal_gen_setup
    mesh = vsa_btt.mesh

    # Use a strain gauge at a mid-radius node (more likely to see signal)
    nodes = np.asarray(mesh.nodes)
    radii = np.linalg.norm(nodes[:, :2], axis=1)
    mid_r = np.median(radii)
    cfg_sg = SensorArrayConfig.default_strain_gauge_array(
        num_gauges=1, radial_positions=[mid_r],
        sample_rate=100_000.0, duration=0.1,
    )
    vsa = VirtualSensorArray(mesh, cfg_sg)

    zeta = 0.05
    gen_cfg = SignalGenerationConfig(
        sample_rate=100_000.0, duration=0.1, seed=42,
        damping_ratio=zeta, max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa, results[:1], 3000.0, gen_cfg)
    sig = out["signals"][0]

    if np.max(np.abs(sig)) < 1e-15:
        pytest.skip("Zero signal at strain gauge location")

    # The envelope should decay: energy in first half > energy in second half
    n = len(sig)
    energy_first = np.sum(sig[:n // 2] ** 2)
    energy_second = np.sum(sig[n // 2:] ** 2)
    assert energy_first > energy_second, "Signal should decay with damping"


# ---- BTT blade passage ----

def test_btt_arrival_times(signal_gen_setup):
    """BTT arrival times should follow t = (θ_probe - 2πb/N) / Ω + n·T_rev."""
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    rpm = 6000.0
    cfg = SignalGenerationConfig(
        sample_rate=500_000.0, duration=0.02, seed=42,
        max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa, results[:1], rpm, cfg)

    if "btt_arrival_times" not in out:
        pytest.skip("No BTT data (mesh may not support blade_tip_profile)")

    # Check that arrival times exist and are within the time window
    for s_idx, arrivals in out["btt_arrival_times"].items():
        if len(arrivals) > 0:
            assert arrivals[0] >= out["time"][0]
            assert arrivals[-1] <= out["time"][-1]
            # Check spacing ≈ T_rev / N_blades for a given blade
            T_rev = 60.0 / rpm
            # Consecutive arrivals of the same blade should be ~T_rev apart
            if len(arrivals) > 1:
                blade_ids = out["btt_blade_indices"][s_idx]
                for b in np.unique(blade_ids):
                    b_times = arrivals[blade_ids == b]
                    if len(b_times) > 1:
                        diffs = np.diff(b_times)
                        np.testing.assert_allclose(
                            diffs, T_rev, rtol=0.01,
                            err_msg=f"Blade {b} arrival spacing != T_rev",
                        )


# ---- filter_modal_results ----

def test_filter_by_nd(signal_gen_setup):
    """Filtering by ND should keep only matching harmonic indices."""
    _, results = signal_gen_setup
    # results has k=0,1,2,3 (from solve_at_rpm(..., 3))
    filtered = filter_modal_results(results, nd=0)
    assert len(filtered) == 1
    assert filtered[0].harmonic_index == 0


def test_filter_by_nd_list(signal_gen_setup):
    """Filtering by a list of NDs."""
    _, results = signal_gen_setup
    filtered = filter_modal_results(results, nd=[0, 2])
    nds = {r.harmonic_index for r in filtered}
    assert nds == {0, 2}


def test_filter_by_nc(signal_gen_setup):
    """Filtering by NC should keep only modes with matching nodal circles."""
    vsa, results = signal_gen_setup
    mesh = vsa.mesh
    filtered = filter_modal_results(results, mesh=mesh, nc=0)
    # All kept modes should have NC=0
    from turbomodal._core import identify_modes
    for r in filtered:
        ids = identify_modes(r, mesh)
        for mid in ids:
            assert mid.nodal_circle == 0


def test_filter_nd_and_nc(signal_gen_setup):
    """Combined ND + NC filter."""
    vsa, results = signal_gen_setup
    mesh = vsa.mesh
    filtered = filter_modal_results(results, mesh=mesh, nd=1, nc=0)
    assert all(r.harmonic_index == 1 for r in filtered)
    from turbomodal._core import identify_modes
    for r in filtered:
        for mid in identify_modes(r, mesh):
            assert mid.nodal_circle == 0


def test_filter_none_keeps_all(signal_gen_setup):
    """No filter should return all results unchanged."""
    _, results = signal_gen_setup
    filtered = filter_modal_results(results)
    assert len(filtered) == len(results)


def test_filter_nc_requires_mesh(signal_gen_setup):
    """Filtering by NC without mesh should raise ValueError."""
    _, results = signal_gen_setup
    with pytest.raises(ValueError, match="mesh is required"):
        filter_modal_results(results, nc=0)

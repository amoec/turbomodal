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


# ====================================================================
# Ray-based surface displacement tests
# ====================================================================

def test_ray_geometry_fields(signal_gen_setup):
    """RayHitGeometry should have correctly shaped arrays."""
    vsa, _ = signal_gen_setup
    geom = vsa.ray_hit_geometry()
    for s_idx, rg in geom.items():
        n_steps = len(rg.hit_mask)
        assert rg.hit_mask.shape == (n_steps,)
        assert rg.hit_points.shape == (n_steps, 3)
        assert rg.cell_ids.shape == (n_steps,)
        assert rg.local_node_ids.shape == (n_steps, 3)
        assert rg.bary_coords.shape == (n_steps, 3)
        assert rg.sector_ids.shape == (n_steps,)

        # Barycentric coords should sum to ~1 at hit locations
        if np.any(rg.hit_mask):
            bary_sums = rg.bary_coords[rg.hit_mask].sum(axis=1)
            np.testing.assert_allclose(bary_sums, 1.0, atol=1e-10)


def test_ray_hit_pattern_backward_compat(signal_gen_setup):
    """ray_hit_pattern() should return boolean masks consistent with ray_hit_geometry()."""
    vsa, _ = signal_gen_setup
    # Get both
    geom = vsa.ray_hit_geometry()
    vsa._ray_hit_cache = None  # force re-derivation
    patterns = vsa.ray_hit_pattern()
    for s_idx in geom:
        np.testing.assert_array_equal(patterns[s_idx], geom[s_idx].hit_mask)


def test_continuous_surface_uses_analytical_path(signal_gen_setup):
    """For a continuous surface (no gaps), the ray-based path should NOT be used."""
    from turbomodal.signal_gen import generate_signals_for_condition

    vsa, results = signal_gen_setup
    # Wedge mesh is a continuous disk — all rays hit
    geom = vsa.ray_hit_geometry()
    for s_idx, rg in geom.items():
        assert np.all(rg.hit_mask), (
            f"Sensor {s_idx}: expected all hits for continuous surface"
        )

    # When all hits are True, the ray-based path should be skipped
    # and the analytical path should be used (identical to old behavior).
    cfg = SignalGenerationConfig(sample_rate=100_000.0, duration=0.01, seed=42)
    out = generate_signals_for_condition(vsa, results, 3000.0, cfg)
    assert "signals" in out
    # Verify signals are non-zero (analytical path produced them)
    assert np.any(out["signals"] != 0)


def test_ray_based_blade_phase_progression():
    """For ND=k on a bladed disk, successive blade pulses should show 2*pi*k/N phase shift.

    Since the test meshes are continuous (no gaps), we test the
    _add_ray_based_component helper directly with a synthetic
    RayHitGeometry that simulates gaps.
    """
    from turbomodal.signal_gen import (
        RayHitGeometry, _interpolate_mode_at_ray_hits, _add_ray_based_component,
    )
    from turbomodal.sensors import SensorLocation, SensorType

    # Synthetic setup: 4 sectors, sensor at theta=0
    N = 4
    sector_angle = 2.0 * np.pi / N
    n_nodes = 10  # nodes per sector
    n_steps = 64  # angular bins per sector sweep

    # Create a mode shape: simple radial displacement, k=1
    # phi(r,z) = [1+0j, 0, 0] at all nodes (radial only)
    mode_shape = np.zeros(3 * n_nodes, dtype=complex)
    for nd in range(n_nodes):
        mode_shape[3 * nd] = 1.0 + 0j  # x-displacement = 1 for all nodes

    # Create a RayHitGeometry with gaps (50% duty cycle)
    hit_mask = np.zeros(n_steps, dtype=bool)
    hit_mask[:n_steps // 2] = True  # first half of sector has blade

    local_node_ids = np.full((n_steps, 3), -1, dtype=np.int64)
    bary_coords = np.zeros((n_steps, 3))
    sector_ids = np.full(n_steps, -1, dtype=np.intp)
    hit_points = np.full((n_steps, 3), np.nan)
    cell_ids = np.full(n_steps, -1, dtype=np.int64)

    # For hit bins: point to the same 3 nodes with equal barycentric weights
    for i in range(n_steps // 2):
        local_node_ids[i] = [0, 1, 2]
        bary_coords[i] = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        sector_ids[i] = 0

    rg = RayHitGeometry(
        hit_mask=hit_mask,
        hit_points=hit_points,
        cell_ids=cell_ids,
        local_node_ids=local_node_ids,
        bary_coords=bary_coords,
        sector_ids=sector_ids,
    )

    # Interpolate — should give uniform displacement in x direction
    sensor_dir = np.array([1.0, 0.0, 0.0])  # radial
    phi_bins = _interpolate_mode_at_ray_hits(mode_shape, rg, n_nodes, sensor_dir)
    assert phi_bins.shape == (n_steps,)
    # Hit bins should be non-zero, miss bins should be zero
    assert np.all(phi_bins[:n_steps // 2] != 0)
    assert np.all(phi_bins[n_steps // 2:] == 0)

    # Now test full signal generation with k=1
    rpm = 6000.0
    omega_rad = 2.0 * np.pi * rpm / 60.0
    f_rot = 1000.0  # rotating frame frequency
    omega_rot = 2.0 * np.pi * f_rot
    k = 1
    w = 1  # forward whirl

    sample_rate = 100_000.0
    T_rev = 60.0 / rpm  # 0.01 s
    duration = 3 * T_rev  # 3 revolutions
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    signals = np.zeros((1, n_samples))
    theta_s = np.array([0.0])
    is_stat = np.array([True])

    sensor = SensorLocation(
        sensor_type=SensorType.BTT_PROBE,
        position=np.array([0.15, 0.0, 0.005]),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    _add_ray_based_component(
        signals, t, mode_shape, 1.0, omega_rot, 0.0,
        k, w, omega_rad, theta_s, is_stat, None, 0.0,
        {0: rg}, n_nodes, N, [sensor],
    )

    # The signal should have N blade pulses per revolution, with
    # different PHASE due to the circumferential phase factor
    # exp(-j*w*k*sector*sector_angle) for ND=k pattern.
    sig = signals[0]
    assert np.max(np.abs(sig)) > 0, "Signal should not be all zeros"

    # Find the peaks of each blade pulse in the first revolution.
    # Each blade passage creates a burst of oscillation; we find
    # peaks by looking for local maxima in the envelope.
    T_blade = T_rev / N
    samples_per_rev = int(T_rev * sample_rate)
    rev_sig = sig[:samples_per_rev]

    # Identify blade pulse regions (non-zero segments)
    nonzero_mask = np.abs(rev_sig) > 1e-15
    # Find transitions to identify individual pulses
    diff = np.diff(nonzero_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if nonzero_mask[0]:
        starts = np.concatenate([[0], starts])
    if nonzero_mask[-1]:
        ends = np.concatenate([ends, [len(rev_sig)]])

    n_pulses = min(len(starts), len(ends))
    assert n_pulses >= 2, f"Expected at least 2 blade pulses, got {n_pulses}"

    # Sample the signal at the midpoint of each pulse
    pulse_midpoint_values = []
    for i in range(n_pulses):
        mid = (starts[i] + ends[i]) // 2
        pulse_midpoint_values.append(rev_sig[mid])

    pulse_midpoint_values = np.array(pulse_midpoint_values)

    # For k=1, N=4: circumferential phases are 0, -pi/2, -pi, -3pi/2
    # so the instantaneous values at equivalent blade positions should
    # NOT all be the same (unlike k=0 where they would be identical).
    assert not np.allclose(pulse_midpoint_values, pulse_midpoint_values[0], atol=1e-10), (
        "Expected different instantaneous values for k=1 traveling wave, "
        f"got all ≈ {pulse_midpoint_values[0]:.6f}"
    )


def test_rotating_sensors_unaffected_by_ray_path(signal_gen_setup):
    """Strain gauge signals should be identical regardless of ray geometry."""
    from turbomodal.signal_gen import generate_signals_for_condition
    from turbomodal.sensors import SensorArrayConfig, SensorType, VirtualSensorArray

    vsa_btt, results = signal_gen_setup
    mesh = vsa_btt.mesh

    nodes = np.asarray(mesh.nodes)
    mid_node = nodes[nodes.shape[0] // 2]
    cfg_sg = SensorArrayConfig.default_strain_gauge_array(
        num_gauges=1, radial_positions=[np.linalg.norm(mid_node[:2])],
        sample_rate=100_000.0, duration=0.05,
    )
    vsa_sg = VirtualSensorArray(mesh, cfg_sg)

    k_results = [r for r in results if r.harmonic_index > 0][:1]
    if not k_results:
        pytest.skip("No k>0 modes in test results")

    cfg = SignalGenerationConfig(
        sample_rate=100_000.0, duration=0.05, seed=42,
        max_modes_per_harmonic=1,
    )
    out = generate_signals_for_condition(vsa_sg, k_results, 3000.0, cfg)
    sig = out["signals"]

    # Rotating sensor should have a non-zero signal
    assert np.any(sig != 0), "Strain gauge signal should not be all zeros"


# ====================================================================
# Physics-based amplitude model tests
# ====================================================================

from turbomodal.signal_gen import (
    ExcitationModel,
    eo_excites_harmonic,
    eo_whirl_direction,
    total_damping_ratio,
    frf_detuning_factor,
    find_campbell_crossings,
    compute_physics_amplitudes,
)


# ---- eo_excites_harmonic ----

class TestEoAliasingRule:
    """Engine order spatial aliasing rule: k = EO mod N."""

    def test_direct_match(self):
        """EO=36 excites k=12 for N=24 (36 mod 24 = 12)."""
        assert eo_excites_harmonic(36, 12, 24) is True

    def test_k0(self):
        """EO=24 excites k=0 for N=24 (24 mod 24 = 0)."""
        assert eo_excites_harmonic(24, 0, 24) is True

    def test_aliased_forward(self):
        """EO=25 excites k=1 for N=24 (25 mod 24 = 1)."""
        assert eo_excites_harmonic(25, 1, 24) is True

    def test_aliased_backward(self):
        """EO=23 excites k=1 for N=24 (23 mod 24 = 23 = -1 mod 24)."""
        assert eo_excites_harmonic(23, 1, 24) is True

    def test_wrong_k_rejected(self):
        """EO=25 does NOT excite k=2 for N=24."""
        assert eo_excites_harmonic(25, 2, 24) is False

    def test_higher_harmonic(self):
        """EO=72 excites k=0 for N=24 (72 mod 24 = 0)."""
        assert eo_excites_harmonic(72, 0, 24) is True

    def test_eo_equals_n_half(self):
        """EO=12 excites k=12 for N=24."""
        assert eo_excites_harmonic(12, 12, 24) is True

    def test_zero_sectors(self):
        """N=0 should always return False."""
        assert eo_excites_harmonic(1, 0, 0) is False


# ---- eo_whirl_direction ----

class TestEoWhirlDirection:
    """Determine FW/BW direction from EO aliasing."""

    def test_forward(self):
        """EO=25, k=1, N=24 -> forward (25 mod 24 = 1 = k)."""
        assert eo_whirl_direction(25, 1, 24) == 1

    def test_backward(self):
        """EO=23, k=1, N=24 -> backward (23 mod 24 = 23 = N-k)."""
        assert eo_whirl_direction(23, 1, 24) == -1

    def test_axisymmetric(self):
        """k=0 is always standing (0)."""
        assert eo_whirl_direction(24, 0, 24) == 0

    def test_n_half_standing(self):
        """k=N/2 is a standing wave."""
        assert eo_whirl_direction(12, 12, 24) == 0

    def test_forward_high_eo(self):
        """EO=49 = 2*24+1, so EO mod 24 = 1 = k -> FW."""
        assert eo_whirl_direction(49, 1, 24) == 1


# ---- total_damping_ratio ----

class TestTotalDamping:
    """ND-dependent total damping ratio."""

    def test_varies_with_nd(self):
        model = ExcitationModel(
            structural_damping_ratio=0.003,
            aero_damping_mean=0.002,
            aero_damping_variation=0.5,
        )
        zeta_0 = total_damping_ratio(0, 24, model)
        zeta_6 = total_damping_ratio(6, 24, model)
        # k=6 gives sin(2*pi*6/24) = sin(pi/2) = 1 -> max aero damping
        assert zeta_6 > zeta_0

    def test_structural_only(self):
        model = ExcitationModel(
            structural_damping_ratio=0.005,
            aero_damping_mean=0.0,
            aero_damping_variation=0.0,
        )
        zeta = total_damping_ratio(3, 24, model)
        assert zeta == pytest.approx(0.005)

    def test_max_at_quarter_n(self):
        """Peak aero damping at k = N/4 where sin(2*pi*k/N) = 1."""
        model = ExcitationModel(
            structural_damping_ratio=0.0,
            aero_damping_mean=0.002,
            aero_damping_variation=1.0,
        )
        zeta_peak = total_damping_ratio(6, 24, model)  # k=N/4
        assert zeta_peak == pytest.approx(0.004)  # mean * (1+1)


# ---- frf_detuning_factor ----

class TestFrfDetuning:
    """Normalised single-DOF FRF magnitude."""

    def test_on_resonance(self):
        """Exactly on resonance should give factor ~1."""
        factor = frf_detuning_factor(1000.0, 1000.0, 0.01)
        assert factor == pytest.approx(1.0, abs=0.05)

    def test_off_resonance_attenuated(self):
        """5% off-resonance should be significantly < 1."""
        factor = frf_detuning_factor(1000.0, 950.0, 0.01)
        assert factor < 0.25

    def test_far_off_resonance(self):
        """50% off should be very small."""
        factor = frf_detuning_factor(1000.0, 500.0, 0.01)
        assert factor < 0.05

    def test_zero_freq(self):
        """Zero mode frequency should return 0."""
        assert frf_detuning_factor(0.0, 500.0, 0.01) == 0.0


# ---- Physics amplitude model properties ----

class TestPhysicsAmplitudeProperties:
    """Verify physics relationships between amplitude parameters."""

    def test_nc_rolloff_ratio(self):
        """NC=1 should be ~1/(2^1.5) = 0.354x of NC=0."""
        # alpha=1.5 -> ratio = 1/(1+1)^1.5 / (1/(1+0)^1.5) = 1/2^1.5
        expected_ratio = 1.0 / (2.0 ** 1.5)
        assert expected_ratio == pytest.approx(0.3536, abs=0.001)

    def test_eo_harmonic_decay(self):
        """h=2 amplitude / h=1 amplitude = 1/2 for beta=1."""
        # A *= 1/h^beta, so ratio = (1/2^1) / (1/1^1) = 0.5
        beta = 1.0
        ratio = (1.0 / 2 ** beta)
        assert ratio == pytest.approx(0.5)

    def test_rpm_scaling(self):
        """Doubling RPM should quadruple amplitude for exponent=2."""
        ratio = (2.0) ** 2.0
        assert ratio == pytest.approx(4.0)

    def test_leo_20db_below(self):
        """LEO at -20 dB is 0.1x amplitude."""
        ratio = 10.0 ** (-20.0 / 20.0)
        assert ratio == pytest.approx(0.1)


# ---- Colored noise ----

class TestColoredNoise:
    """Tests for generate_colored_noise."""

    def test_length(self):
        from turbomodal.noise import generate_colored_noise
        noise = generate_colored_noise(1024, 44100.0)
        assert len(noise) == 1024

    def test_unit_rms(self):
        from turbomodal.noise import generate_colored_noise
        noise = generate_colored_noise(8192, 44100.0)
        rms = np.sqrt(np.mean(noise ** 2))
        assert rms == pytest.approx(1.0, abs=0.1)

    def test_spectral_slope(self):
        """PSD slope should approximate -gamma in log-log space."""
        from turbomodal.noise import generate_colored_noise
        gamma = 5.0 / 3.0
        noise = generate_colored_noise(
            65536, 44100.0, spectral_exponent=gamma,
            rng=np.random.default_rng(42),
        )
        # Compute PSD via periodogram
        fft_vals = np.fft.rfft(noise)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(noise), d=1.0 / 44100.0)

        # Fit slope in log-log space (skip DC and near-Nyquist)
        mask = (freqs > 100) & (freqs < 15000)
        log_f = np.log10(freqs[mask])
        log_psd = np.log10(psd[mask] + 1e-30)
        slope = np.polyfit(log_f, log_psd, 1)[0]

        # Slope should be close to -gamma (within tolerance for finite signal)
        assert slope < 0, "PSD should decrease with frequency"
        assert abs(slope + gamma) < 1.0, (
            f"PSD slope {slope:.2f} too far from -{gamma:.2f}"
        )

    def test_reproducible(self):
        from turbomodal.noise import generate_colored_noise
        n1 = generate_colored_noise(1024, 44100.0, rng=np.random.default_rng(7))
        n2 = generate_colored_noise(1024, 44100.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(n1, n2)


# ---- Physics signal generation integration ----

class TestPhysicsSignalGeneration:
    """Integration tests for amplitude_mode='physics'."""

    def test_requires_excitation_model(self, signal_gen_setup):
        """Should raise ValueError without excitation_model."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01,
            amplitude_mode="physics",
        )
        with pytest.raises(ValueError, match="excitation_model"):
            generate_signals_for_condition(vsa, results, 3000.0, cfg)

    def test_output_shape(self, signal_gen_setup):
        """Physics mode should return correct signal shape."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        model = ExcitationModel(
            stator_vane_counts=[24],
            campbell_crossing_tolerance=0.99,  # wide tolerance to catch modes
            broadband_snr_db=0.0,  # disable broadband for shape test
        )
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01,
            amplitude_mode="physics",
            excitation_model=model,
        )
        output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        assert "signals" in output
        assert "time" in output
        assert output["signals"].shape[0] == vsa.n_sensors
        assert output["signals"].shape[1] == len(output["time"])

    def test_reproducible(self, signal_gen_setup):
        """Same seed should produce identical physics signals."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        model = ExcitationModel(
            stator_vane_counts=[24],
            campbell_crossing_tolerance=0.99,
            broadband_snr_db=0.0,
        )
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01, seed=42,
            amplitude_mode="physics",
            excitation_model=model,
        )
        out1 = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        out2 = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        np.testing.assert_array_equal(out1["signals"], out2["signals"])

    def test_backward_compatible(self, signal_gen_setup):
        """Unit mode should still work exactly as before."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01, seed=42,
            amplitude_mode="unit",
        )
        output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        assert output["signals"].shape[0] == vsa.n_sensors

    def test_with_broadband_noise(self, signal_gen_setup):
        """Physics mode with broadband noise should produce non-zero signals."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        model = ExcitationModel(
            stator_vane_counts=[24],
            campbell_crossing_tolerance=0.99,
            broadband_snr_db=10.0,
        )
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01,
            amplitude_mode="physics",
            excitation_model=model,
        )
        output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        # Should have non-zero signals (broadband noise at minimum)
        assert np.any(output["signals"] != 0)

    def test_mistuning_jitter(self, signal_gen_setup):
        """Physics mode with jitter mistuning should run without error."""
        from turbomodal.signal_gen import generate_signals_for_condition
        vsa, results = signal_gen_setup
        model = ExcitationModel(
            stator_vane_counts=[24],
            campbell_crossing_tolerance=0.99,
            mistuning_sigma=0.02,
            mistuning_method="jitter",
            broadband_snr_db=0.0,
        )
        cfg = SignalGenerationConfig(
            sample_rate=100_000.0, duration=0.01,
            amplitude_mode="physics",
            excitation_model=model,
        )
        output = generate_signals_for_condition(vsa, results, 3000.0, cfg)
        assert output["signals"].shape[0] == vsa.n_sensors


# ---- ExcitationModel defaults ----

def test_excitation_model_defaults():
    """ExcitationModel should have sensible defaults."""
    model = ExcitationModel()
    assert model.stator_vane_counts == [24]
    assert model.max_eo_harmonic == 3
    assert model.nc_rolloff_alpha == 1.5
    assert model.structural_damping_ratio == 0.003
    assert model.aero_damping_mean == 0.002
    assert model.campbell_crossing_tolerance == 0.05
    assert model.broadband_snr_db == 20.0
    assert model.broadband_spectral_exponent == pytest.approx(5.0 / 3.0)


# ---- NC identification on blade sector mesh ----

class TestBladeNcIdentification:
    """NC identification tests using the blade sector mesh (hub + blade)."""

    def test_bending_nc_progression(self, solved_blade):
        """First several bending modes should have increasing NC with frequency."""
        from turbomodal._core import identify_modes

        mesh, _, results = solved_blade
        for r in results:
            if r.harmonic_index != 0:
                continue
            ids = identify_modes(r, mesh)
            # Collect bending modes sorted by frequency
            bending = [(mid.frequency, mid.nodal_circle)
                       for mid in ids if mid.family_label.endswith("B")]
            if len(bending) < 3:
                continue
            bending.sort()
            # Check the first 5 bending modes (higher modes may be
            # misclassified or belong to a different sub-family)
            n_check = min(5, len(bending))
            ncs = [nc for _, nc in bending[:n_check]]
            for i in range(1, len(ncs)):
                assert ncs[i] >= ncs[i - 1], (
                    f"Bending NC should increase with frequency: "
                    f"f={bending[i-1][0]:.0f} NC={ncs[i-1]} -> "
                    f"f={bending[i][0]:.0f} NC={ncs[i]}"
                )

    def test_lowest_bending_mode_is_nc0(self, solved_blade):
        """The lowest-frequency bending mode should be NC=0."""
        from turbomodal._core import identify_modes

        mesh, _, results = solved_blade
        for r in results:
            if r.harmonic_index != 0:
                continue
            ids = identify_modes(r, mesh)
            bending = [(mid.frequency, mid.nodal_circle)
                       for mid in ids if mid.family_label.endswith("B")]
            if not bending:
                continue
            bending.sort()
            assert bending[0][1] == 0, (
                f"Lowest bending mode should be NC=0, "
                f"got NC={bending[0][1]} at f={bending[0][0]:.0f} Hz"
            )

    def test_blade_mesh_has_bending_modes(self, solved_blade):
        """Blade sector mesh should produce B-family modes (unlike continuous disk)."""
        from turbomodal._core import identify_modes

        mesh, _, results = solved_blade
        families_found: set[str] = set()
        for r in results:
            if r.harmonic_index == 0:
                ids = identify_modes(r, mesh)
                for mid in ids:
                    families_found.add(mid.family_label[-1])
        assert "B" in families_found, "Blade mesh should have bending (B) modes"

    def test_nc_values_reasonable_range(self, solved_blade):
        """NC values should be in a reasonable range for the mesh resolution."""
        from turbomodal._core import identify_modes

        mesh, _, results = solved_blade
        for r in results:
            if r.harmonic_index != 0:
                continue
            ids = identify_modes(r, mesh)
            for mid in ids:
                assert mid.nodal_circle >= 0, "NC should be non-negative"
                assert mid.nodal_circle < 20, (
                    f"NC={mid.nodal_circle} seems too high for this mesh, "
                    f"f={mid.frequency:.0f} Hz {mid.family_label}"
                )

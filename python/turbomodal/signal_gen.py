"""End-to-end signal generation pipeline (Subsystem B orchestrator).

Combines virtual sensor array, noise models, and modal results
to generate synthetic time-domain signals for ML training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys

import numpy as np

from turbomodal._utils import progress_bar as _progress_bar
from turbomodal.noise import apply_noise, NoiseConfig


@dataclass
class SignalGenerationConfig:
    """Configuration for the complete signal generation pipeline.

    Parameters
    ----------
    sample_rate : Sampling rate in Hz.
    duration : Total acquisition duration in seconds.
    num_revolutions : If > 0, overrides *duration* based on RPM.
    seed : Random seed for reproducibility.
    amplitude_mode : ``"unit"``, ``"forced_response"``, or ``"random"``.
    amplitude_scale : Base amplitude in metres (for ``"unit"`` mode).
    max_frequency : Upper frequency cutoff in Hz (0 = no limit).
    max_modes_per_harmonic : Maximum modes per harmonic index (0 = all).
    time : Custom time array; overrides all other time parameters.
    t_start : Start time in seconds.
    t_end : End time in seconds (0 = use *duration* or *num_revolutions*).
    damping_ratio : Modal damping ratio zeta (0 = undamped).
    """

    sample_rate: float = 100000.0      # Hz
    duration: float = 1.0              # seconds
    num_revolutions: int = 0           # If > 0, overrides duration
    seed: int = 42

    # Amplitude mode: how to set modal amplitudes
    amplitude_mode: str = "unit"       # "unit", "forced_response", "random"
    amplitude_scale: float = 1e-6      # meters (base amplitude for "unit" mode)

    # Which modes to include
    max_frequency: float = 0.0         # Hz (0 = all modes)
    max_modes_per_harmonic: int = 0    # 0 = all modes

    # Time vector control
    time: np.ndarray | None = None     # Custom time array (overrides all below)
    t_start: float = 0.0              # Start time (s)
    t_end: float = 0.0                # End time (0 = use duration)
    damping_ratio: float = 0.0        # Modal damping ζ (0 = undamped)


def _build_annulus_surface(mesh):
    """Build the full annulus outer surface as a PyVista PolyData.

    Replicates the sector mesh across all *N* sectors, then extracts
    the triangulated surface.  The result is cached — pass the same
    mesh object to avoid rebuilding.
    """
    import pyvista as pv
    from turbomodal._utils import rotation_matrix_3x3

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    n_nodes = mesh.num_nodes()
    n_elem = elements.shape[0]
    N = mesh.num_sectors
    sector_angle = 2.0 * np.pi / N
    axis = mesh.rotation_axis

    all_pts = np.empty((N * n_nodes, 3))
    all_cells = np.empty((N * n_elem, 11), dtype=np.int64)
    all_celltypes = np.full(N * n_elem, 24, dtype=np.uint8)  # VTK_QUADRATIC_TETRA

    for s in range(N):
        R = rotation_matrix_3x3(s * sector_angle, axis)
        all_pts[s * n_nodes:(s + 1) * n_nodes] = nodes @ R.T
        offset = s * n_elem
        all_cells[offset:offset + n_elem, 0] = 10
        all_cells[offset:offset + n_elem, 1:] = elements + s * n_nodes

    grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, all_pts)
    return grid.extract_surface(algorithm="dataset_surface")


def _precompute_ray_hits(surface, sensors, is_stat, mesh, sector_angle,
                         n_steps: int = 256) -> dict[int, np.ndarray]:
    """Pre-compute ray-surface intersections for stationary sensors.

    For each stationary sensor, simulate the disk rotating through one
    sector (all sectors are identical by cyclic symmetry).  At each
    angular step, rotate the sensor into the disk's rest frame and cast
    a ray along its measurement direction.  Record hit/miss.

    Returns ``{sensor_idx: bool_array(n_steps)}``.
    """
    from turbomodal._utils import rotation_matrix_3x3

    axis = mesh.rotation_axis
    results: dict[int, np.ndarray] = {}

    for s_idx, sensor in enumerate(sensors):
        if not is_stat[s_idx]:
            continue

        pos = np.asarray(sensor.position, dtype=np.float64)
        direction = np.asarray(sensor.direction, dtype=np.float64)
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-30:
            continue
        direction = direction / d_norm

        hit_mask = np.zeros(n_steps, dtype=bool)

        for i in range(n_steps):
            # Rotate sensor backward by (i / n_steps) * sector_angle
            # (equivalent to disk rotating forward by that amount)
            angle = -i * sector_angle / n_steps
            R = rotation_matrix_3x3(angle, axis)
            pos_rot = R @ pos
            dir_rot = R @ direction

            # Extend ray well beyond the mesh
            end_point = pos_rot + dir_rot * 10.0
            hit_pts, _ = surface.ray_trace(pos_rot, end_point)
            hit_mask[i] = len(hit_pts) > 0

        results[s_idx] = hit_mask

    return results


def _build_time_vector(config: SignalGenerationConfig, rpm: float) -> np.ndarray:
    """Build the time vector from config, respecting priority order.

    Priority: ``config.time`` > ``t_start/t_end`` > ``num_revolutions`` > ``duration``.
    """
    if config.time is not None:
        return np.asarray(config.time, dtype=np.float64)

    if config.t_end > config.t_start:
        t_start = config.t_start
        t_end = config.t_end
    elif config.num_revolutions > 0 and rpm != 0:
        t_start = config.t_start
        t_end = t_start + config.num_revolutions * 60.0 / abs(rpm)
    else:
        t_start = config.t_start
        t_end = t_start + config.duration

    n_samples = max(1, int((t_end - t_start) * config.sample_rate))
    return t_start + np.arange(n_samples) / config.sample_rate


def _sensor_is_stationary(sensor) -> bool:
    """Determine whether a sensor is stationary (casing-mounted) or rotating.

    By default, only strain gauges are rotating (mounted on the blade).
    BTT probes, casing accelerometers, and displacement sensors (eddy
    current, capacitive, laser vibrometer) are stationary.  Override
    with ``SensorLocation(is_stationary=...)``.
    """
    from turbomodal.sensors import SensorType

    if sensor.is_stationary is not None:
        return sensor.is_stationary
    # Only strain gauges rotate with the disk
    return sensor.sensor_type != SensorType.STRAIN_GAUGE


def filter_modal_results(
    modal_results: list,
    mesh=None,
    nd: int | list[int] | None = None,
    nc: int | list[int] | None = None,
    whirl: int | list[int] | None = None,
) -> list:
    """Filter modal results by nodal diameter, nodal circles, and/or whirl.

    Parameters
    ----------
    modal_results : list of ModalResult (one per harmonic index).
    mesh : Mesh object (required when filtering by *nc*).
    nd : Nodal diameter(s) to keep.  ``None`` keeps all.
    nc : Nodal circle(s) to keep.  ``None`` keeps all.
        Requires *mesh* for on-the-fly mode identification.
    whirl : Whirl direction(s) to keep: ``+1`` (FW), ``-1`` (BW),
        ``0`` (degenerate/standing).  ``None`` keeps all.

    Returns
    -------
    Filtered list of ModalResult.  When *nc* or *whirl* is specified,
    individual results may contain fewer modes than the originals.
    """
    from turbomodal._core import ModalResult

    # Normalise scalar → list
    nd_set: set[int] | None = None
    nc_set: set[int] | None = None
    whirl_set: set[int] | None = None
    if nd is not None:
        nd_set = {nd} if isinstance(nd, int) else set(nd)
    if nc is not None:
        nc_set = {nc} if isinstance(nc, int) else set(nc)
    if whirl is not None:
        whirl_set = {whirl} if isinstance(whirl, int) else set(whirl)

    # Step 1: filter by ND (harmonic_index)
    if nd_set is not None:
        modal_results = [r for r in modal_results if r.harmonic_index in nd_set]

    # Step 2: filter by whirl direction (per-mode)
    if whirl_set is not None:
        filtered_w: list = []
        for r in modal_results:
            w_arr = np.asarray(r.whirl_direction)
            keep = [i for i in range(len(r.frequencies)) if int(w_arr[i]) in whirl_set]
            if not keep:
                continue
            nr = ModalResult()
            nr.harmonic_index = r.harmonic_index
            nr.rpm = r.rpm
            nr.converged = r.converged
            idx = np.array(keep)
            nr.frequencies = np.asarray(r.frequencies)[idx]
            nr.mode_shapes = np.asarray(r.mode_shapes)[:, idx]
            nr.whirl_direction = np.asarray(r.whirl_direction)[idx]
            filtered_w.append(nr)
        modal_results = filtered_w

    # Step 3: filter by NC (requires mode identification)
    if nc_set is not None:
        if mesh is None:
            raise ValueError("mesh is required when filtering by nc")
        from turbomodal._core import identify_modes

        filtered: list = []
        for r in modal_results:
            ids = identify_modes(r, mesh)
            keep = [i for i, mid in enumerate(ids) if mid.nodal_circle in nc_set]
            if not keep:
                continue
            # Build a new ModalResult with only the kept modes
            nr = ModalResult()
            nr.harmonic_index = r.harmonic_index
            nr.rpm = r.rpm
            nr.converged = r.converged
            idx = np.array(keep)
            nr.frequencies = np.asarray(r.frequencies)[idx]
            nr.mode_shapes = np.asarray(r.mode_shapes)[:, idx]
            nr.whirl_direction = np.asarray(r.whirl_direction)[idx]
            filtered.append(nr)
        modal_results = filtered

    return modal_results


def generate_signals_for_condition(
    sensor_array,
    modal_results: list,
    rpm: float,
    config: SignalGenerationConfig,
    noise_config=None,
    forced_response_result=None,
    condition=None,
) -> dict:
    """Generate synthetic sensor signals for a single operating condition.

    Uses a **full-annulus virtual probe** model: for each mode the
    displacement field is ``u(r, θ, z, t) = Re[φ_k(r,z) ·
    exp(−j·w·k·θ) · exp(j·ω·t)]``, where *θ* is the sensor's
    circumferential angle and *w* is the whirl direction.  Stationary
    sensors (BTT probes, casing accelerometers) observe the Doppler-
    shifted stationary-frame frequency; rotating sensors (strain
    gauges) observe the rotating-frame frequency directly.

    **Blade passage gating** uses ray tracing against the full-annulus
    surface mesh.  For each stationary sensor a ray is cast along its
    measurement direction; if the ray hits the blade surface the
    displacement is read, otherwise (gap between blades) the signal
    is zero.  The hit pattern is pre-computed once for one sector
    sweep (cyclic symmetry) and tiled across the time series.  This
    naturally handles shrouded blades (surface spans the full sector →
    ray always hits), unshrouded blades (gaps → ray misses), and
    arbitrary sensor orientations.

    Parameters
    ----------
    sensor_array : VirtualSensorArray instance
    modal_results : list of ModalResult (one per harmonic)
    rpm : rotational speed in RPM
    config : signal generation configuration
    noise_config : NoiseConfig (optional, for adding noise)
    forced_response_result : ForcedResponseResult (optional, for amplitudes)
    condition : OperatingCondition (optional, included in output for tracing)

    Returns
    -------
    dict with keys:
        ``'signals'``
            ``(n_sensors, n_samples)`` float64.
        ``'time'``
            ``(n_samples,)`` float64.
        ``'clean_signals'``
            ``(n_sensors, n_samples)`` float64 (before noise).
        ``'condition'``
            OperatingCondition (if provided).
        ``'btt_arrival_times'``
            ``{sensor_idx: ndarray}`` — discrete blade arrival times
            (only for BTT probes).
        ``'btt_deflections'``
            ``{sensor_idx: ndarray}`` — deflection at each arrival.
        ``'btt_blade_indices'``
            ``{sensor_idx: ndarray}`` — which blade produced each
            arrival.
    """
    sensors = sensor_array.config.sensors
    n_sensors = len(sensors)
    t = _build_time_vector(config, rpm)
    n_samples = len(t)

    signals = np.zeros((n_sensors, n_samples))
    rng = np.random.default_rng(config.seed)

    # --- Rotation parameters ---
    omega_hz = abs(rpm) / 60.0          # rev/s
    omega_rad = 2.0 * np.pi * omega_hz  # rad/s

    # --- Sensor classification ---
    is_stat = np.array([_sensor_is_stationary(s) for s in sensors])

    # --- Circumferential geometry (when mesh is available) ---
    mesh = getattr(sensor_array, 'mesh', None)
    has_mesh = mesh is not None and hasattr(mesh, 'num_sectors')

    if has_mesh:
        theta_s = sensor_array.sensor_circumferential_angles()  # (n_sensors,)
        N = mesh.num_sectors
        sector_angle = 2.0 * np.pi / N
    else:
        theta_s = np.zeros(n_sensors)
        N = 0
        sector_angle = 0.0

    # --- Mode shape sampling ---
    # sample_mode_shape uses nearest node in sector 0; the circumferential
    # phase factor exp(-j*w*k*θ) is applied separately below.

    mode_count = 0
    for mr in modal_results:
        n_modes = len(mr.frequencies)
        k = mr.harmonic_index
        whirl_arr = (
            np.asarray(mr.whirl_direction)
            if hasattr(mr, 'whirl_direction')
            else np.zeros(n_modes, dtype=np.int32)
        )

        for m in range(n_modes):
            f_rot = mr.frequencies[m]
            if config.max_modes_per_harmonic > 0 and m >= config.max_modes_per_harmonic:
                break

            phi_s = sensor_array.sample_mode_shape(mr.mode_shapes[:, m])  # (n_sensors,) complex

            # Amplitude
            if config.amplitude_mode == "forced_response" and forced_response_result is not None:
                if mode_count < len(forced_response_result.max_response_amplitude):
                    amp = forced_response_result.max_response_amplitude[mode_count]
                else:
                    amp = config.amplitude_scale
            elif config.amplitude_mode == "random":
                amp = config.amplitude_scale * rng.exponential(1.0)
            else:
                amp = config.amplitude_scale

            omega_rot = 2.0 * np.pi * f_rot
            w = int(whirl_arr[m]) if m < len(whirl_arr) else 0
            phase = rng.uniform(0, 2 * np.pi)

            # Damping envelope: exp(-ζ·ω·(t - t0))
            if config.damping_ratio > 0:
                t0 = t[0]
                envelope = np.exp(-config.damping_ratio * omega_rot * (t - t0))
            else:
                envelope = None

            # ---- Synthesise per-mode contribution ----
            # Physics: u(r,θ,z,t) = Re[φ(r,z) · exp(-j·w·k·θ) · exp(j·ω·t)]
            # FW (w=+1): exp(-jkθ) rotates with the disk → lab freq = f_rot + kΩ
            # BW (w=-1): exp(+jkθ) rotates against disk → lab freq = |f_rot − kΩ|

            if w != 0 or k == 0:
                # --- Single component (Coriolis-split or axisymmetric k=0) ---
                _add_single_component(
                    signals, t, phi_s, amp, omega_rot, phase,
                    k, w, omega_rad, theta_s, is_stat, envelope,
                    config.max_frequency,
                )
            else:
                # --- Degenerate (w=0, k>0): emit FW + BW at half amplitude ---
                for w_comp in (+1, -1):
                    _add_single_component(
                        signals, t, phi_s, amp * 0.5, omega_rot, phase,
                        k, w_comp, omega_rad, theta_s, is_stat, envelope,
                        config.max_frequency,
                    )

            mode_count += 1

    # --- Blade passage gating via ray tracing ---
    # Any stationary sensor that looks at the rotating structure through
    # gaps (BTT probes, casing-mounted displacement probes aimed at blade
    # tips, etc.) must be gated: the signal is zero when the ray misses
    # the surface.  The ray hit pattern is geometry-only and cached on
    # the sensor array.
    #
    # Discrete blade arrival extraction (times, deflections, blade IDs)
    # is BTT-specific — only computed for BTT_PROBE sensors.
    from turbomodal.sensors import SensorType

    is_btt = np.array([s.sensor_type == SensorType.BTT_PROBE for s in sensors])

    btt_arrival_times: dict[int, np.ndarray] = {}
    btt_deflections: dict[int, np.ndarray] = {}
    btt_blade_indices: dict[int, np.ndarray] = {}

    if has_mesh and np.any(is_stat) and omega_rad > 0:
        try:
            ray_hits = sensor_array.ray_hit_pattern()
        except Exception:
            ray_hits = {}

        T_rev = 2.0 * np.pi / omega_rad

        for s in range(n_sensors):
            if not is_stat[s] or s not in ray_hits:
                continue

            hit_mask_sector = ray_hits[s]  # (n_steps,) bool for one sector
            n_steps = len(hit_mask_sector)

            # Map each time sample to the corresponding angular bin
            theta_local = (theta_s[s] - omega_rad * t) % sector_angle
            bin_idx = (theta_local / sector_angle * n_steps).astype(np.intp)
            np.clip(bin_idx, 0, n_steps - 1, out=bin_idx)
            on_blade = hit_mask_sector[bin_idx]

            # Gate the continuous signal — zero where ray misses
            signals[s, ~on_blade] = 0.0

            # Discrete blade arrivals (BTT probes only)
            if is_btt[s]:
                arrivals: list[float] = []
                deflections_list: list[float] = []
                blade_ids: list[int] = []
                for b in range(N):
                    blade_angle = 2.0 * np.pi * b / N
                    t0_blade = (theta_s[s] - blade_angle) / omega_rad
                    while t0_blade < t[0]:
                        t0_blade += T_rev
                    while t0_blade <= t[-1]:
                        arrivals.append(t0_blade)
                        idx_t = min(np.searchsorted(t, t0_blade), n_samples - 1)
                        deflections_list.append(signals[s, idx_t])
                        blade_ids.append(b)
                        t0_blade += T_rev

                btt_arrival_times[s] = np.array(arrivals)
                btt_deflections[s] = np.array(deflections_list)
                btt_blade_indices[s] = np.array(blade_ids, dtype=np.int32)

    clean_signals = signals.copy()

    # Apply noise
    if noise_config is not None:
        signals = apply_noise(signals, noise_config, config.sample_rate, rng)

    result_dict: dict = {
        "signals": signals,
        "time": t,
        "clean_signals": clean_signals,
    }
    if condition is not None:
        result_dict["condition"] = condition
    if btt_arrival_times:
        result_dict["btt_arrival_times"] = btt_arrival_times
        result_dict["btt_deflections"] = btt_deflections
        result_dict["btt_blade_indices"] = btt_blade_indices
    return result_dict


def _add_single_component(
    signals: np.ndarray,
    t: np.ndarray,
    phi_s: np.ndarray,
    amp: float,
    omega_rot: float,
    phase: float,
    k: int,
    w: int,
    omega_rad: float,
    theta_s: np.ndarray,
    is_stat: np.ndarray,
    envelope: np.ndarray | None,
    max_frequency: float,
) -> None:
    """Add one mode component (FW or BW) to the signal array in-place.

    For stationary sensors the lab-frame frequency is used with the
    circumferential phase factor.  For rotating sensors the rotating-
    frame frequency is used.
    """
    n_sensors = len(phi_s)

    # Stationary-frame frequency: f_stat = f_rot + w·k·Ω
    f_rot = omega_rot / (2.0 * np.pi)
    f_omega = omega_rad / (2.0 * np.pi)
    f_stat = abs(f_rot + w * k * f_omega)
    omega_stat = 2.0 * np.pi * f_stat

    if max_frequency > 0 and f_stat > max_frequency:
        return

    # Pre-compute trig for stationary and rotating frequencies
    cos_stat = np.cos(omega_stat * t + phase)
    sin_stat = np.sin(omega_stat * t + phase)
    cos_rot = np.cos(omega_rot * t + phase)
    sin_rot = np.sin(omega_rot * t + phase)

    for s in range(n_sensors):
        phi_mag = np.abs(phi_s[s])
        phi_phase = np.angle(phi_s[s])
        if phi_mag < 1e-30:
            continue

        if is_stat[s]:
            # Circumferential phase: -w·k·θ_s
            circ_phase = -w * k * theta_s[s]
            total_phase = phi_phase + circ_phase
            contrib = amp * phi_mag * (
                np.cos(total_phase) * cos_stat - np.sin(total_phase) * sin_stat
            )
        else:
            # Rotating sensor: frequency = f_rot, circumferential phase = -w·k·θ_s
            circ_phase = -w * k * theta_s[s]
            total_phase = phi_phase + circ_phase
            contrib = amp * phi_mag * (
                np.cos(total_phase) * cos_rot - np.sin(total_phase) * sin_rot
            )

        if envelope is not None:
            contrib *= envelope

        signals[s, :] += contrib


def generate_dataset_signals(
    mesh,
    modal_results_per_condition: list[list],
    conditions: list,
    sensor_array,
    config: SignalGenerationConfig = SignalGenerationConfig(),
    noise_config=None,
    forced_response_results: Optional[list] = None,
    verbose: int = 1,
) -> dict:
    """Generate synthetic sensor signals for a full parametric dataset.

    Parameters
    ----------
    mesh : Mesh
    modal_results_per_condition : results[cond_idx] = list of ModalResult
    conditions : list of OperatingCondition
    sensor_array : VirtualSensorArray instance
    config : signal generation config
    noise_config : NoiseConfig (optional)
    forced_response_results : list of ForcedResponseResult (optional)
    verbose : 0=silent, 1=progress

    Returns
    -------
    dict with keys:
        'signals': (n_conditions, n_sensors, n_samples) float64
        'clean_signals': (n_conditions, n_sensors, n_samples) float64
        'conditions': list of OperatingCondition
        'sample_rate': float
        'time': (n_samples,) float64
    """
    import time

    n_cond = len(modal_results_per_condition)
    n_sensors = len(sensor_array.config.sensors)

    # Use the first condition's RPM for time vector sizing
    ref_rpm = abs(conditions[0].rpm) if len(conditions) > 0 and conditions[0].rpm != 0 else 3000.0
    t_ref = _build_time_vector(config, ref_rpm)
    n_samples = len(t_ref)

    all_signals = np.zeros((n_cond, n_sensors, n_samples))
    all_clean = np.zeros((n_cond, n_sensors, n_samples))
    t_start = time.perf_counter()

    for i in range(n_cond):
        fr = forced_response_results[i] if forced_response_results else None
        cond = conditions[i] if i < len(conditions) else None
        rpm = cond.rpm if cond is not None else 0.0

        result = generate_signals_for_condition(
            sensor_array, modal_results_per_condition[i],
            rpm, config, noise_config, fr, condition=cond,
        )

        all_signals[i] = result["signals"]
        all_clean[i] = result["clean_signals"]

        if verbose >= 1:
            elapsed = time.perf_counter() - t_start
            bar = _progress_bar(i + 1, n_cond, prefix="  Signal gen: ", elapsed=elapsed)
            sys.stdout.write(bar)
            sys.stdout.flush()
            if i == n_cond - 1:
                sys.stdout.write("\n")

    return {
        "signals": all_signals,
        "clean_signals": all_clean,
        "conditions": conditions,
        "sample_rate": config.sample_rate,
        "time": t_ref,
    }

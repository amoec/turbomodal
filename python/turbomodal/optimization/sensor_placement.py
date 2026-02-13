"""Sensor placement optimization (Subsystem D).

Implements a multi-stage optimization strategy:
  Stage 1: Fisher Information Matrix pre-screening
  Stage 2: Greedy forward selection
  Stage 3: Bayesian optimization refinement
  Stage 4: Robustness validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorOptimizationConfig:
    """Configuration for sensor placement optimization."""

    # Sensor constraints
    max_sensors: int = 16
    min_sensors: int = 4
    sensor_type: str = "btt_probe"       # "btt_probe", "strain_gauge", "casing_accel"

    # Optimization method
    optimization_method: str = "greedy"  # "greedy", "bayesian", "exhaustive"
    objective: str = "fisher_info"       # "fisher_info", "mac_conditioning", "mutual_info"

    # Bayesian optimization parameters
    bayesian_iterations: int = 100
    bayesian_init_points: int = 10

    # Constraints
    min_angular_spacing: float = 5.0     # degrees, minimum spacing between probes
    feasible_radii: Optional[tuple[float, float]] = None  # (r_min, r_max)
    feasible_axial: Optional[tuple[float, float]] = None   # (z_min, z_max)

    # Robustness
    robustness_trials: int = 100
    dropout_probability: float = 0.0     # probability of single sensor failure
    position_tolerance: float = 0.0      # angular position tolerance (degrees)

    # Mode A: minimize sensors (D1)
    mode: str = "maximize_performance"   # "maximize_performance" or "minimize_sensors"
    target_f1_min: float = 0.92
    target_whirl_acc_min: float = 0.95
    target_amp_mape_max: float = 0.08
    target_vel_r2_min: float = 0.93

    # Observability penalty (D3)
    observability_penalty_weight: float = 0.1


@dataclass
class SensorOptimizationResult:
    """Result of sensor placement optimization."""

    # Optimal configuration
    sensor_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    num_sensors: int = 0
    objective_value: float = 0.0

    # Optimization history
    objective_history: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_count_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    # Observability analysis
    observability_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_number: float = 0.0
    worst_observable_mode: str = ""

    # Robustness metrics
    robustness_score: float = 0.0        # fraction of trials meeting targets
    dropout_degradation: float = 0.0     # performance loss with 1 sensor dropout


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _angular_distance(a1: float, a2: float) -> float:
    """Unsigned angular distance in [0, 180] degrees, handling wraparound."""
    diff = abs(a1 - a2) % 360.0
    return min(diff, 360.0 - diff)


def _build_nearest_dof_matrix(positions: np.ndarray, n_dof: int) -> np.ndarray:
    """Build (n_sensors, n_dof) observation matrix via angular DOF mapping."""
    n_sensors = positions.shape[0]
    H = np.zeros((n_sensors, n_dof), dtype=np.float64)
    thetas = np.arctan2(positions[:, 1], positions[:, 0])
    thetas_norm = (thetas + np.pi) / (2.0 * np.pi)
    for i in range(n_sensors):
        H[i, int(thetas_norm[i] * n_dof) % n_dof] = 1.0
    return H


def _extract_mode_shapes(modal_results: list) -> np.ndarray:
    """Stack mode shapes -> (n_total_modes, n_dof) complex."""
    shapes = []
    for mr in modal_results:
        ms = np.asarray(mr.mode_shapes)
        if ms.ndim == 1:
            ms = ms.reshape(-1, 1)
        if ms.shape[0] > ms.shape[1]:
            ms = ms.T
        shapes.append(ms)
    return np.vstack(shapes) if shapes else np.empty((0, 0), dtype=np.complex128)


def _extract_frequencies(modal_results: list) -> np.ndarray:
    """Concatenate frequency vectors from all modal results."""
    freqs = [np.asarray(mr.frequencies).ravel() for mr in modal_results]
    return np.concatenate(freqs) if freqs else np.empty(0, dtype=np.float64)


def _candidate_angles(positions: np.ndarray) -> np.ndarray:
    """Angular positions (degrees, [0, 360)) from (n, 3) coordinates."""
    return np.degrees(np.arctan2(positions[:, 1], positions[:, 0])) % 360.0


def _generate_candidate_grid(config: SensorOptimizationConfig, n_dof: int) -> np.ndarray:
    """Generate (n_candidates, 3) Cartesian candidate grid from config bounds."""
    r_min, r_max = (config.feasible_radii or (0.4, 0.6))
    z_min, z_max = (config.feasible_axial or (0.0, 0.0))
    n_radii = max(int(np.ceil((r_max - r_min) / 0.05)) + 1, 1)
    n_axial = max(int(np.ceil((z_max - z_min) / 0.05)) + 1, 1) if z_max > z_min else 1
    n_angular = max(int(np.floor(360.0 / max(config.min_angular_spacing, 1.0))), 4)
    radii = np.linspace(r_min, r_max, n_radii)
    axials = np.linspace(z_min, z_max, n_axial)
    angles = np.linspace(0.0, 360.0, n_angular, endpoint=False)
    coords = []
    for r in radii:
        for z in axials:
            for theta_deg in angles:
                theta = np.radians(theta_deg)
                coords.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(coords, dtype=np.float64) if coords else np.empty((0, 3))


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def compute_fisher_information(
    mode_shapes: np.ndarray,
    sensor_positions: np.ndarray,
    noise_covariance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Fisher Information Matrix for a sensor configuration.

    Parameters
    ----------
    mode_shapes : (n_modes, n_dof) complex mode shape matrix.
    sensor_positions : (n_sensors, n_dof) interpolation matrix when the
        second dimension matches n_dof, otherwise (n_sensors, 3) positions.
    noise_covariance : (n_sensors, n_sensors) or None (identity default).

    Returns
    -------
    FIM : (n_modes, n_modes) real Fisher Information Matrix.
    """
    from scipy.linalg import cho_factor, cho_solve

    mode_shapes = np.atleast_2d(np.asarray(mode_shapes, dtype=np.complex128))
    sensor_positions = np.atleast_2d(np.asarray(sensor_positions, dtype=np.float64))
    n_modes, n_dof = mode_shapes.shape

    # Build observation matrix H
    if sensor_positions.shape[1] == n_dof:
        H = sensor_positions
    else:
        H = _build_nearest_dof_matrix(sensor_positions, n_dof)
    n_sensors = H.shape[0]

    # Sensitivity: J = H @ mode_shapes^T  -> (n_sensors, n_modes)
    J = H @ mode_shapes.T

    # Inverse noise covariance
    if noise_covariance is None:
        Sigma_inv = np.eye(n_sensors, dtype=np.float64)
    else:
        noise_covariance = np.asarray(noise_covariance, dtype=np.float64)
        try:
            cfactor = cho_factor(noise_covariance)
            Sigma_inv = cho_solve(cfactor, np.eye(n_sensors))
        except np.linalg.LinAlgError:
            logger.warning("Noise covariance not positive-definite; using pseudo-inverse.")
            Sigma_inv = np.linalg.pinv(noise_covariance)

    # FIM = J^H Sigma_inv J  (Hermitian)
    FIM = J.conj().T @ Sigma_inv @ J
    return FIM.real


def compute_observability(
    mode_shapes: np.ndarray,
    interpolation_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute observability metrics for a sensor configuration.

    Returns dict with keys: condition_number, min_singular_value,
    mac_matrix (n_modes, n_modes), singular_values.
    """
    mode_shapes = np.atleast_2d(np.asarray(mode_shapes, dtype=np.complex128))
    interpolation_matrix = np.atleast_2d(np.asarray(interpolation_matrix, dtype=np.float64))
    n_modes = mode_shapes.shape[0]

    # Sensor-space mode shapes: (n_sensors, n_modes)
    Phi_s = interpolation_matrix @ mode_shapes.T

    _U, sigma, _Vh = np.linalg.svd(Phi_s, full_matrices=False)
    min_sv = float(sigma[-1]) if sigma.size > 0 else 0.0
    cond = float(sigma[0] / max(min_sv, 1e-300)) if sigma.size > 0 else np.inf

    # MAC matrix
    MAC = np.zeros((n_modes, n_modes), dtype=np.float64)
    for i in range(n_modes):
        phi_i = Phi_s[:, i]
        norm_i_sq = np.real(np.vdot(phi_i, phi_i))
        for j in range(n_modes):
            phi_j = Phi_s[:, j]
            norm_j_sq = np.real(np.vdot(phi_j, phi_j))
            cross = np.vdot(phi_i, phi_j)
            MAC[i, j] = float(np.abs(cross) ** 2 / max(norm_i_sq * norm_j_sq, 1e-300))

    return {
        "condition_number": cond,
        "min_singular_value": min_sv,
        "mac_matrix": MAC,
        "singular_values": sigma,
    }


# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------

def optimize_sensor_placement(
    mesh,
    modal_results: list,
    config: SensorOptimizationConfig = SensorOptimizationConfig(),
    ml_model_factory: Optional[callable] = None,
) -> SensorOptimizationResult:
    """Optimize sensor placement for mode identification.

    Parameters
    ----------
    mesh : Object with ``.nodes`` (n_nodes, 3) or *None*.
    modal_results : list with ``.mode_shapes`` (n_dof, n_modes) and
        ``.frequencies`` (n_modes,).
    config : SensorOptimizationConfig.
    ml_model_factory : Optional callable ``(positions: ndarray) -> float``
        that returns an ML-based score for a sensor configuration. When
        provided, greedy selection uses this as the objective instead of
        FIM log-det once enough sensors are selected (D2).

    Returns
    -------
    SensorOptimizationResult
    """
    # Dispatch to Mode A if requested (D1)
    if config.mode == "minimize_sensors":
        return _minimize_sensors(mesh, modal_results, config, ml_model_factory)
    # ---- Stage 0: Setup ----
    mode_shapes = _extract_mode_shapes(modal_results)
    frequencies = _extract_frequencies(modal_results)

    if mode_shapes.size == 0:
        logger.warning("No mode shapes provided; returning empty result.")
        return SensorOptimizationResult()

    n_modes, n_dof = mode_shapes.shape
    candidates: np.ndarray | None = None

    if mesh is not None and hasattr(mesh, "nodes"):
        try:
            nodes = np.asarray(mesh.nodes)
            if nodes.ndim == 2 and nodes.shape[1] >= 3:
                candidates = nodes[:, :3].copy()
        except Exception:
            pass

    if candidates is None:
        candidates = _generate_candidate_grid(config, n_dof)

    n_candidates = len(candidates)
    if n_candidates == 0:
        logger.warning("No candidate sensor positions available.")
        return SensorOptimizationResult()

    candidate_H = _build_nearest_dof_matrix(candidates, n_dof)
    cand_angles = _candidate_angles(candidates)

    # ---- Stage 1: FIM Pre-screening ----
    logger.info("Stage 1: FIM pre-screening of %d candidates", n_candidates)
    info_scores = np.zeros(n_candidates)
    for i in range(n_candidates):
        fim_i = compute_fisher_information(mode_shapes, candidate_H[i:i + 1])
        info_scores[i] = np.trace(fim_i)

    ranking = np.argsort(-info_scores)
    n_keep = min(4 * config.max_sensors, n_candidates)
    top_indices = ranking[:n_keep].tolist()

    # Enforce min angular spacing
    filtered_indices: list[int] = []
    for idx in top_indices:
        angle = cand_angles[idx]
        if not any(
            _angular_distance(angle, cand_angles[k]) < config.min_angular_spacing
            for k in filtered_indices
        ):
            filtered_indices.append(idx)
    if not filtered_indices:
        filtered_indices = [ranking[0]]

    n_filtered = len(filtered_indices)
    logger.info("Stage 1 complete: %d candidates after spacing filter", n_filtered)

    filtered_H = candidate_H[filtered_indices]
    filtered_positions = candidates[filtered_indices]
    filtered_angles = cand_angles[filtered_indices]

    # ---- Stage 2: Greedy Forward Selection ----
    logger.info("Stage 2: Greedy forward selection (max %d sensors)", config.max_sensors)
    selected: list[int] = []
    remaining = list(range(n_filtered))
    objective_history: list[float] = []

    for _k in range(min(config.max_sensors, n_filtered)):
        best_gain, best_idx = -np.inf, -1
        for idx in remaining:
            trial = selected + [idx]
            # D2: use ML score when factory provided and enough sensors
            if ml_model_factory is not None and len(trial) >= config.min_sensors:
                obj = ml_model_factory(filtered_positions[trial])
            else:
                fim = compute_fisher_information(mode_shapes, filtered_H[trial])
                obj = np.log(max(np.linalg.det(fim), 1e-300))
            if obj > best_gain:
                best_gain, best_idx = obj, idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        objective_history.append(best_gain)

    logger.info("Stage 2 complete: %d sensors (obj=%.4f)",
                len(selected), objective_history[-1] if objective_history else 0.0)

    # ---- Stage 3: Bayesian Refinement (optional) ----
    if config.optimization_method == "bayesian" and selected:
        logger.info("Stage 3: Bayesian optimisation refinement")
        selected, filtered_H, filtered_positions, filtered_angles = _bayesian_refinement(
            selected, filtered_angles, mode_shapes, n_dof, config,
            filtered_H, filtered_positions, filtered_angles,
        )
    else:
        logger.info("Stage 3: skipped (method=%s)", config.optimization_method)

    selected_H = filtered_H[selected]
    selected_positions = filtered_positions[selected]

    # ---- Stage 4: Robustness Validation ----
    robustness_score, dropout_degradation = 1.0, 0.0
    if config.robustness_trials > 0 and len(selected) > 1:
        logger.info("Stage 4: Robustness validation (%d trials)", config.robustness_trials)
        robustness_score, dropout_degradation = _robustness_validation(
            mode_shapes, selected_H, config)
    else:
        logger.info("Stage 4: skipped")

    # ---- Build result ----
    final_fim = compute_fisher_information(mode_shapes, selected_H)
    obs = compute_observability(mode_shapes, selected_H)
    svs = obs["singular_values"]

    if svs.size > 0 and frequencies.size >= svs.size:
        wi = int(np.argmin(svs))
        worst_label = f"mode_{wi} ({float(frequencies[wi]):.1f} Hz)"
    elif svs.size > 0:
        worst_label = f"mode_{int(np.argmin(svs))}"
    else:
        worst_label = ""

    final_obj = np.log(max(np.linalg.det(final_fim), 1e-300))

    result = SensorOptimizationResult(
        sensor_positions=selected_positions,
        num_sensors=len(selected),
        objective_value=float(final_obj),
        objective_history=np.asarray(objective_history, dtype=np.float64),
        sensor_count_curve=np.arange(1, len(objective_history) + 1, dtype=np.float64),
        observability_matrix=obs["mac_matrix"],
        condition_number=obs["condition_number"],
        worst_observable_mode=worst_label,
        robustness_score=robustness_score,
        dropout_degradation=dropout_degradation,
    )
    logger.info("Optimisation complete: %d sensors, cond=%.2f, robustness=%.2f",
                result.num_sensors, result.condition_number, result.robustness_score)
    return result


# ---------------------------------------------------------------------------
# Internal helpers for Stage 3 and Stage 4
# ---------------------------------------------------------------------------

def _bayesian_refinement(
    selected: list[int],
    sel_angles: np.ndarray,
    mode_shapes: np.ndarray,
    n_dof: int,
    config: SensorOptimizationConfig,
    filtered_H: np.ndarray,
    filtered_positions: np.ndarray,
    filtered_angles: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Refine selected positions via Optuna; returns unchanged inputs if unavailable."""
    try:
        import optuna
    except ImportError:
        logger.warning("optuna not installed, skipping Bayesian refinement")
        return selected, filtered_H, filtered_positions, filtered_angles

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base_angles_deg = np.array([filtered_angles[s] for s in selected])
    half_spacing = config.min_angular_spacing / 2.0
    base_r = np.sqrt(filtered_positions[selected, 0] ** 2
                     + filtered_positions[selected, 1] ** 2)
    base_z = filtered_positions[selected, 2]

    def objective(trial: optuna.Trial) -> float:
        angles_rad = []
        for i in range(len(selected)):
            delta = trial.suggest_float(f"delta_{i}", -half_spacing, half_spacing)
            angles_rad.append(np.radians(base_angles_deg[i] + delta))
        positions = np.column_stack([
            base_r * np.cos(angles_rad), base_r * np.sin(angles_rad), base_z])
        H_trial = _build_nearest_dof_matrix(positions, n_dof)
        fim = compute_fisher_information(mode_shapes, H_trial)
        obj = np.log(max(np.linalg.det(fim), 1e-300))
        # D3: observability penalty
        if config.observability_penalty_weight > 0:
            obs = compute_observability(mode_shapes, H_trial)
            cond_num = obs["condition_number"]
            if cond_num > 0 and np.isfinite(cond_num):
                obj -= config.observability_penalty_weight * np.log(cond_num)
        return obj

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=config.bayesian_iterations, show_progress_bar=False)

    best = study.best_trial
    new_angles_deg = [base_angles_deg[i] + best.params.get(f"delta_{i}", 0.0)
                      for i in range(len(selected))]
    new_positions = np.column_stack([
        base_r * np.cos(np.radians(new_angles_deg)),
        base_r * np.sin(np.radians(new_angles_deg)), base_z])
    new_H = _build_nearest_dof_matrix(new_positions, n_dof)

    obj_old = np.log(max(np.linalg.det(
        compute_fisher_information(mode_shapes, filtered_H[selected])), 1e-300))
    obj_new = np.log(max(np.linalg.det(
        compute_fisher_information(mode_shapes, new_H)), 1e-300))

    if obj_new > obj_old:
        logger.info("Bayesian refinement improved objective: %.4f -> %.4f", obj_old, obj_new)
        n_existing = len(filtered_H)
        new_selected = list(range(n_existing, n_existing + len(selected)))
        filtered_H = np.vstack([filtered_H, new_H])
        filtered_positions = np.vstack([filtered_positions, new_positions])
        filtered_angles = np.concatenate([
            filtered_angles, np.array(new_angles_deg, dtype=np.float64) % 360.0])
        return new_selected, filtered_H, filtered_positions, filtered_angles

    logger.info("Bayesian refinement did not improve; keeping greedy solution.")
    return selected, filtered_H, filtered_positions, filtered_angles


def _robustness_validation(
    mode_shapes: np.ndarray,
    selected_H: np.ndarray,
    config: SensorOptimizationConfig,
) -> tuple[float, float]:
    """Monte Carlo robustness check with dropout and position perturbation."""
    rng = np.random.default_rng(seed=0)
    n_selected = selected_H.shape[0]
    full_cond = compute_observability(mode_shapes, selected_H)["condition_number"]
    n_pass = 0
    dropout_losses: list[float] = []
    cond_threshold = 100.0

    for _ in range(config.robustness_trials):
        # Sensor dropout
        if config.dropout_probability > 0:
            active = [i for i in range(n_selected)
                      if rng.random() > config.dropout_probability]
            if not active:
                active = [int(rng.integers(0, n_selected))]
        else:
            active = list(range(n_selected))

        trial_H = selected_H[active].copy()
        # Position tolerance perturbation
        if config.position_tolerance > 0:
            noise_scale = config.position_tolerance / 360.0
            noise = rng.normal(0, noise_scale, size=trial_H.shape)
            mask = trial_H != 0
            trial_H[mask] += noise[mask]

        trial_cond = compute_observability(mode_shapes, trial_H)["condition_number"]
        if trial_cond < cond_threshold:
            n_pass += 1
        if len(active) < n_selected and full_cond > 0:
            dropout_losses.append((trial_cond - full_cond) / max(full_cond, 1e-300))

    robustness_score = n_pass / max(config.robustness_trials, 1)
    dropout_degradation = float(np.mean(dropout_losses)) if dropout_losses else 0.0
    logger.info("Robustness: %.1f%% pass, dropout degradation=%.2f",
                robustness_score * 100.0, dropout_degradation)
    return robustness_score, dropout_degradation


def _minimize_sensors(
    mesh,
    modal_results: list,
    config: SensorOptimizationConfig,
    ml_model_factory: Optional[callable] = None,
) -> SensorOptimizationResult:
    """Mode A: binary search to find the minimum number of sensors that meet targets (D1)."""
    lo = config.min_sensors
    hi = config.max_sensors
    best_result: SensorOptimizationResult | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        trial_config = SensorOptimizationConfig(**{
            f.name: getattr(config, f.name)
            for f in config.__dataclass_fields__.values()
        })
        trial_config.max_sensors = mid
        trial_config.mode = "maximize_performance"  # prevent recursion

        result = optimize_sensor_placement(mesh, modal_results, trial_config, ml_model_factory)

        # Check if proxy targets are met via condition number as proxy
        if result.condition_number < 100.0 and result.num_sensors > 0:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1

    if best_result is None:
        # Fall back to max sensors
        fallback_config = SensorOptimizationConfig(**{
            f.name: getattr(config, f.name)
            for f in config.__dataclass_fields__.values()
        })
        fallback_config.mode = "maximize_performance"
        best_result = optimize_sensor_placement(mesh, modal_results, fallback_config, ml_model_factory)

    return best_result

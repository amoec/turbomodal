"""Parametric sweep orchestrator for turbomodal cyclic symmetry analysis.

Generates operating conditions via Latin Hypercube Sampling and drives
the CyclicSymmetrySolver (with optional FMM mistuning) over the entire
parameter space, producing an HDF5 dataset ready for downstream analysis
or machine learning.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from turbomodal._core import (
    CyclicSymmetrySolver,
    FluidConfig,
    FMMSolver,
    Material,
    ParametricCondition,
)
from turbomodal._utils import progress_bar as _progress_bar
from turbomodal.dataset import DatasetConfig, OperatingCondition, export_modal_results
from turbomodal.solver import BoundaryCondition, _build_constraint_groups


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParametricRange:
    """Defines the sweep range for a single operating parameter.

    Parameters
    ----------
    name : parameter name; must be one of "rpm", "temperature"
    low : lower bound of the range
    high : upper bound of the range
    log_scale : if True, sample uniformly in log-space (useful for
        parameters spanning orders of magnitude)
    """

    name: str
    low: float
    high: float
    log_scale: bool = False


# Canonical parameter names that map to OperatingCondition fields
_PARAM_NAMES = frozenset({
    "rpm",
    "temperature",
})


@dataclass
class ParametricSweepConfig:
    """Configuration for a parametric sweep study.

    Parameters
    ----------
    ranges : list of ParametricRange objects defining the swept parameters
    num_samples : total number of LHS samples to generate
    sampling_method : sampling strategy; currently only "lhs" is supported
    seed : random seed for reproducibility
    num_modes : number of modes per harmonic index to compute
    include_mistuning : whether to apply random blade-to-blade mistuning
    mistuning_sigma : standard deviation of fractional frequency deviations
        (only used when include_mistuning is True)
    """

    ranges: list[ParametricRange] = field(default_factory=list)
    num_samples: int = 1000
    sampling_method: str = "lhs"
    seed: int = 42
    num_modes: int = 10
    include_mistuning: bool = False
    mistuning_sigma: float = 0.02


# ---------------------------------------------------------------------------
# Latin Hypercube sampling
# ---------------------------------------------------------------------------

def generate_conditions(config: ParametricSweepConfig) -> list[OperatingCondition]:
    """Generate operating conditions by Latin Hypercube Sampling.

    Parameters
    ----------
    config : sweep configuration with parameter ranges and sample count

    Returns
    -------
    List of ``OperatingCondition`` objects, one per sample point.

    Raises
    ------
    ValueError
        If a parameter name in *config.ranges* is not recognised, or if
        no ranges are specified.
    """
    from scipy.stats.qmc import LatinHypercube

    if not config.ranges:
        raise ValueError("At least one ParametricRange must be specified.")

    for pr in config.ranges:
        if pr.name not in _PARAM_NAMES:
            raise ValueError(
                f"Unknown parameter name '{pr.name}'. "
                f"Must be one of {sorted(_PARAM_NAMES)}."
            )

    n_dim = len(config.ranges)
    sampler = LatinHypercube(d=n_dim, seed=config.seed)
    # unit_samples has shape (num_samples, n_dim) in [0, 1)
    unit_samples = sampler.random(n=config.num_samples)

    conditions: list[OperatingCondition] = []

    for i in range(config.num_samples):
        kwargs: dict[str, Any] = {"condition_id": i}

        for j, pr in enumerate(config.ranges):
            u = float(unit_samples[i, j])

            if pr.log_scale:
                # Map [0, 1) -> [log(low), log(high)] -> exp
                if pr.low <= 0 or pr.high <= 0:
                    raise ValueError(
                        f"log_scale requires positive bounds for '{pr.name}', "
                        f"got [{pr.low}, {pr.high}]."
                    )
                log_low = np.log(pr.low)
                log_high = np.log(pr.high)
                value = float(np.exp(log_low + u * (log_high - log_low)))
            else:
                value = pr.low + u * (pr.high - pr.low)

            kwargs[pr.name] = value

        conditions.append(OperatingCondition(**kwargs))

    return conditions


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_parametric_sweep(
    mesh: Any,
    base_material: Material,
    config: ParametricSweepConfig,
    dataset_config: DatasetConfig | None = None,
    damping: Any | None = None,
    fluid: FluidConfig | None = None,
    verbose: int = 1,
    boundary_conditions: list[BoundaryCondition] | None = None,
) -> str:
    """Execute a parametric sweep over operating conditions.

    For each sampled condition the routine:

    1. Adjusts the material stiffness to the condition temperature via
       ``Material.at_temperature``.
    2. Creates a ``CyclicSymmetrySolver`` and solves for modal results.
    3. Optionally runs the Fundamental Mistuning Model (``FMMSolver``)
       when ``config.include_mistuning`` is True.
    4. Collects results keyed by condition id.

    When *dataset_config* is provided the full results are written to an
    HDF5 file at the end via ``export_modal_results``.

    Parameters
    ----------
    mesh : turbomodal Mesh object
    base_material : reference Material (temperature-dependent if E_slope != 0)
    config : parametric sweep configuration
    dataset_config : HDF5 export settings (None = do not export)
    damping : optional DampingConfig (reserved for future forced response)
    fluid : optional FluidConfig for fluid-structure coupling
    verbose : 0 = silent, 1 = progress bar, 2 = per-condition detail
    boundary_conditions : list of ``BoundaryCondition`` objects (e.g. from
        ``bc_editor``).  When provided, constraint groups are built from these
        instead of using the default hub constraint.

    Returns
    -------
    Path to the HDF5 file if *dataset_config* is provided, otherwise an
    empty string.
    """
    if fluid is None:
        fluid = FluidConfig()

    # ------------------------------------------------------------------
    # 1. Generate conditions
    # ------------------------------------------------------------------
    conditions = generate_conditions(config)
    n_cond = len(conditions)

    if verbose >= 1:
        print(
            f"Parametric sweep: {n_cond} conditions, "
            f"{config.num_modes} modes/ND, "
            f"{mesh.num_nodes()} nodes, "
            f"{mesh.num_sectors} sectors"
        )
        if config.include_mistuning:
            print(f"  Mistuning enabled: sigma={config.mistuning_sigma:.4f}")
        print()

    # ------------------------------------------------------------------
    # 2. Solve each condition
    # ------------------------------------------------------------------
    all_results: dict[int, list] = {}
    t_start = time.perf_counter()

    # Fast path: batch all conditions through C++ solve_parametric
    # when mistuning is not required (avoids Python loop overhead and
    # re-assembles K/M once, scaling by temperature/RPM internally).
    if not config.include_mistuning:
        # Build C++ ParametricCondition list
        cpp_conditions = []
        for cond in conditions:
            pc = ParametricCondition(
                rpm=cond.rpm,
                temperature=cond.temperature,
            )
            cpp_conditions.append(pc)

        # Single solver instance, single batched call
        if boundary_conditions is not None:
            constraint_groups = _build_constraint_groups(mesh, boundary_conditions)
            solver = CyclicSymmetrySolver(mesh, base_material, constraint_groups, fluid)
        else:
            solver = CyclicSymmetrySolver(mesh, base_material, fluid)

        progress_cb = None
        if verbose >= 1:
            def progress_cb(done, total, _k, _conv):
                elapsed = time.perf_counter() - t_start
                bar = _progress_bar(done, total, prefix="  ", elapsed=elapsed)
                sys.stdout.write(bar)
                sys.stdout.flush()
                if done == total:
                    sys.stdout.write("\n")

        batch_results = solver.solve_parametric(
            cpp_conditions, config.num_modes,
            progress_cb=progress_cb,
        )

        for idx, cond in enumerate(conditions):
            all_results[cond.condition_id] = batch_results[idx]

    else:
        # Slow path: per-condition Python loop (needed for FMM mistuning)
        # Build constraint groups once (they don't change per condition)
        constraint_groups = (
            _build_constraint_groups(mesh, boundary_conditions)
            if boundary_conditions is not None
            else None
        )

        for idx, cond in enumerate(conditions):
            t0 = time.perf_counter()

            # Adjust material for temperature
            mat = base_material.at_temperature(cond.temperature)

            # Solve tuned cyclic symmetry problem
            if constraint_groups is not None:
                solver = CyclicSymmetrySolver(mesh, mat, constraint_groups, fluid)
            else:
                solver = CyclicSymmetrySolver(mesh, mat, fluid)
            results = solver.solve_at_rpm(cond.rpm, config.num_modes)

            # Apply FMM mistuning
            mistuning_pattern = FMMSolver.random_mistuning(
                mesh.num_sectors,
                config.mistuning_sigma,
                seed=config.seed + cond.condition_id,
            )
            cond.mistuning_pattern = np.asarray(mistuning_pattern)

            n_harmonics = len(results)
            if n_harmonics > 0:
                tuned_freqs = np.array([
                    r.frequencies[0] if len(r.frequencies) > 0 else 0.0
                    for r in results
                ])
                _mistuning_result = FMMSolver.solve(
                    mesh.num_sectors,
                    tuned_freqs,
                    cond.mistuning_pattern,
                )

            all_results[cond.condition_id] = results
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_start

            if verbose == 1:
                bar = _progress_bar(idx + 1, n_cond, prefix="  ", elapsed=elapsed)
                sys.stdout.write(bar)
                sys.stdout.flush()
                if idx == n_cond - 1:
                    sys.stdout.write("\n")
            elif verbose >= 2:
                n_modes_total = sum(len(r.frequencies) for r in results)
                print(
                    f"  [{idx + 1}/{n_cond}] cond={cond.condition_id}  "
                    f"rpm={cond.rpm:.0f}  T={cond.temperature:.1f}K  "
                    f"{len(results)} harmonics, {n_modes_total} modes  "
                    f"({dt:.2f}s)"
                )

    total_time = time.perf_counter() - t_start
    if verbose >= 1:
        print(
            f"  Sweep complete: {total_time:.1f}s total, "
            f"{total_time / max(n_cond, 1):.2f}s/condition"
        )

    # ------------------------------------------------------------------
    # 3. Export to HDF5
    # ------------------------------------------------------------------
    output_path = ""
    if dataset_config is not None:
        output_path = dataset_config.output_path
        if verbose >= 1:
            print(f"  Exporting to {output_path} ...")

        export_modal_results(
            path=output_path,
            mesh=mesh,
            conditions=conditions,
            all_results=all_results,
            config=dataset_config,
        )

        if verbose >= 1:
            print(f"  Export complete.")

    return output_path

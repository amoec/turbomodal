"""High-level solver interface for turbomodal."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from turbomodal._core import (
    BCType,
    ConstraintGroup,
    CyclicSymmetrySolver,
    FluidConfig,
    Material,
    Mesh,
    ModalResult,
    SolverConfig,
)
from turbomodal._utils import progress_bar as _progress_bar

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    

@dataclass
class BoundaryCondition:
    """Specification for a boundary condition applied via a cutting plane.

    Parameters
    ----------
    name : descriptive name for this constraint group
    type : ``"fixed"``, ``"displacement"``, or ``"frictionless"``
    plane_point : (3,) point on the cutting plane
    plane_normal : (3,) outward normal — surface nodes on the positive side are selected
    constrained_components : for ``"displacement"`` type, which DOF components
        are locked (x, y, z).  Ignored for other types.
    tolerance : distance tolerance for plane selection
    node_ids : pre-selected node IDs (from interactive editor).  When provided,
        ``_build_constraint_groups`` uses these directly instead of calling
        ``select_nodes_by_plane``.
    selection_radius : radius used during interactive selection (stored for
        reproducibility / display).  Has no effect at solve time.
    """

    name: str
    type: str  # "fixed", "displacement", "frictionless"
    plane_point: np.ndarray
    plane_normal: np.ndarray
    constrained_components: tuple[bool, bool, bool] = (True, True, True)
    tolerance: float = 1e-6
    node_ids: list[int] | None = None
    selection_radius: float | None = None


def _build_constraint_groups(
    mesh: Mesh, bcs: list[BoundaryCondition]
) -> list[ConstraintGroup]:
    """Convert BoundaryCondition specs to C++ ConstraintGroup objects."""
    groups = []
    type_map = {
        "fixed": BCType.FIXED,
        "displacement": BCType.DISPLACEMENT,
        "frictionless": BCType.FRICTIONLESS,
    }
    for bc in bcs:
        cg = ConstraintGroup()
        cg.name = bc.name
        bc_type = type_map.get(bc.type.lower())
        if bc_type is None:
            raise ValueError(
                f"Unknown BC type '{bc.type}'. Must be 'fixed', 'displacement', or 'frictionless'."
            )
        cg.type = bc_type
        if bc.node_ids is not None:
            cg.node_ids = list(bc.node_ids)
        else:
            cg.node_ids = mesh.select_nodes_by_plane(
                np.asarray(bc.plane_point, dtype=np.float64),
                np.asarray(bc.plane_normal, dtype=np.float64),
                bc.tolerance,
            )
        cg.constrained_components = list(bc.constrained_components)
        if bc_type == BCType.FRICTIONLESS:
            cg.surface_normal = mesh.compute_surface_normal(cg.node_ids)
        groups.append(cg)
    return groups

# Verbosity levels
SILENT = 0
PROGRESS = 1
DETAILED = 2


def _validate_modal_results(results: list[ModalResult]) -> list[ModalResult]:
    """Post-solve sanity checks. Logs warnings for suspicious results."""
    for r in results:
        freqs = np.asarray(r.frequencies)
        if freqs.size == 0:
            continue
        if np.any(np.isnan(freqs)):
            logger.warning("ND=%d: NaN frequencies detected — solver may not have converged",
                           r.harmonic_index)
        if np.any(freqs < 0):
            logger.warning("ND=%d: negative frequencies detected", r.harmonic_index)
        shapes = np.asarray(r.mode_shapes)
        if shapes.size > 0:
            norms = np.linalg.norm(shapes, axis=0)
            if np.any(norms < 1e-15):
                logger.warning("ND=%d: zero-norm mode shapes detected", r.harmonic_index)
    return results


def solve(
    mesh: Mesh,
    material: Material,
    rpm: float = 0.0,
    num_modes: int = 20,
    fluid: FluidConfig | None = None,
    config: SolverConfig | None = None,
    verbose: int = PROGRESS,
    harmonic_indices: list[int] | None = None,
    max_threads: int = 0,
    hub_constraint: str = "fixed",
    include_coriolis: bool = False,
    min_frequency: float = 0.0,
    temperature: float | None = None,
    condition: _RemovedClass | None = None,
    boundary_conditions: list[BoundaryCondition] | None = None,
    show_convergence: bool = False,
    memory_reserve_fraction: float = 0.2,
) -> list[ModalResult]:
    """Solve cyclic symmetry modal analysis at a given RPM.

    Parameters
    ----------
    mesh : Mesh with cyclic boundaries identified
    material : Material properties
    rpm : rotation speed in RPM (0 = stationary)
    num_modes : number of modes per harmonic index
    fluid : fluid coupling configuration (None = dry)
    config : solver configuration (None = defaults)
    verbose : 0=silent, 1=progress bar, 2=detailed output
    harmonic_indices : list of harmonic indices to solve (None = all 0..N/2)
    max_threads : max concurrent solver threads (0 = auto, uses hardware_concurrency)
    hub_constraint : ``"fixed"`` clamps all hub DOFs (default).  ``"free"``
        leaves the hub unconstrained (free-rotating assembly, no shaft
        contact).  Centrifugal prestress is disabled when hub is free.
    include_coriolis : if True, include gyroscopic (Coriolis) coupling via
        Lancaster linearization of the QEP.  Produces mode-shape-dependent
        FW/BW splitting in the rotating frame for k > 0.
    min_frequency : minimum frequency threshold in Hz.  Modes below this are
        discarded as rigid body modes.  Set > 0 when using
        ``hub_constraint="free"`` to filter out zero-frequency rigid body
        modes that arise from the unconstrained system.
    temperature : bulk temperature in Kelvin.  When provided, the material
        stiffness is adjusted via ``material.at_temperature(temperature)``
        before solving.  Ignored if the material has ``E_slope == 0``.
    condition : an ``_RemovedClass`` instance.  When provided, *rpm*
        and *temperature* are taken from the condition (explicit keyword
        arguments still override).
    boundary_conditions : list of ``BoundaryCondition`` objects defining
        constraint groups via cutting planes.  When provided, overrides
        ``hub_constraint``.  Each BC selects surface nodes on the positive
        side of a plane and applies the specified constraint type.
    show_convergence : if True, open a live matplotlib bar chart showing
        per-harmonic convergence status as each nodal diameter is solved.
        Requires matplotlib.  The plot stays open after solving completes.
    memory_reserve_fraction : fraction of total system RAM to keep free
        (0.0–1.0).  Default 0.2 (use up to 80 % of RAM).

    Returns
    -------
    List of ModalResult, one per harmonic index (0 to N/2)
    """
    # Resolve operating condition overrides
    if condition is not None:
        rpm = condition.rpm
        if temperature is None:
            temperature = condition.temperature

    if temperature is not None:
        material = material.at_temperature(temperature)

    if fluid is None:
        fluid = FluidConfig()

    hi = harmonic_indices or []

    if verbose >= PROGRESS:
        max_k = mesh.num_sectors // 2
        nd_str = f"ND {hi}" if hi else f"ND 0..{max_k}"
        coriolis_str = " [Coriolis]" if include_coriolis else ""
        temp_str = f"  T={temperature:.1f}K" if temperature is not None else ""
        print(
            f"Solving at {rpm:.0f} RPM{coriolis_str}{temp_str}  ({mesh.num_nodes()} nodes, "
            f"{mesh.num_elements()} elements, {mesh.num_sectors} sectors, "
            f"{nd_str}, {num_modes} modes/ND)"
        )

    t0 = time.perf_counter()
    if boundary_conditions is not None:
        constraint_groups = _build_constraint_groups(mesh, boundary_conditions)
        solver = CyclicSymmetrySolver(mesh, material, constraint_groups, fluid)
    else:
        apply_hub = hub_constraint == "fixed"
        solver = CyclicSymmetrySolver(mesh, material, fluid, apply_hub)

    if show_convergence:
        max_k = mesh.num_sectors // 2
        hi_resolved = hi if hi else list(range(max_k + 1))
        results = _solve_with_convergence_plot(
            solver, rpm, num_modes, hi, hi_resolved, max_threads, include_coriolis,
            min_frequency, verbose, t0, memory_reserve_fraction,
        )
    else:
        progress_cb = None
        if verbose >= PROGRESS:
            def _on_progress(done: int, total: int, k: int, converged: bool) -> None:
                elapsed = time.perf_counter() - t0
                bar = _progress_bar(done, total, prefix="  ", elapsed=elapsed)
                sys.stdout.write(bar)
                sys.stdout.flush()
                if done == total:
                    sys.stdout.write("\n")
            progress_cb = _on_progress

        results = solver.solve_at_rpm(rpm, num_modes, hi, max_threads, include_coriolis,
                                       min_frequency, progress_cb,
                                       memory_reserve_fraction=memory_reserve_fraction)

    elapsed = time.perf_counter() - t0

    if verbose >= PROGRESS:
        print(f"  Solved {len(results)} harmonic indices in {elapsed:.1f}s")

    if verbose >= DETAILED:
        for r in results:
            freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:4])
            print(f"    ND={r.harmonic_index:2d}: [{freqs}] Hz (rotating frame)")

    return _validate_modal_results(results)


def _solve_with_convergence_plot(
    solver: CyclicSymmetrySolver,
    rpm: float,
    num_modes: int,
    hi: list[int],
    hi_resolved: list[int],
    max_threads: int,
    include_coriolis: bool,
    min_frequency: float,
    verbose: int,
    t0: float,
    memory_reserve_fraction: float = 0.2,
) -> list[ModalResult]:
    """Run solve_at_rpm in a background thread while updating a live bar chart."""
    import queue
    import threading

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_harmonics = len(hi_resolved)
    hi_list = hi_resolved

    update_queue: queue.Queue = queue.Queue()
    results_holder: list = []

    PENDING  = "#d0d0d0"
    DONE_OK  = "#4caf50"
    DONE_BAD = "#ff9800"

    fig, ax = plt.subplots(figsize=(max(6, n_harmonics * 0.35 + 2), 3.5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    bars = ax.bar(range(n_harmonics), [1] * n_harmonics, color=PENDING,
                  edgecolor="#444", linewidth=0.5)
    ax.set_xticks(range(n_harmonics))
    ax.set_xticklabels([str(k) for k in hi_list], fontsize=7, color="white")
    ax.set_yticks([])
    ax.set_xlim(-0.6, n_harmonics - 0.4)
    ax.set_ylim(0, 1.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.set_xlabel("Nodal Diameter (k)", color="white", fontsize=9)
    title_obj = ax.set_title(
        f"Cyclic Solver  |  {rpm:.0f} RPM  |  0 / {n_harmonics}",
        color="white", fontsize=10,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=DONE_OK,  label="Converged"),
            mpatches.Patch(color=DONE_BAD, label="Not converged"),
            mpatches.Patch(color=PENDING,  label="Pending"),
        ],
        loc="upper right", fontsize=7, facecolor="#333", labelcolor="white", framealpha=0.8,
    )
    plt.tight_layout()
    plt.ion()
    plt.show()

    k_to_bar = {k: i for i, k in enumerate(hi_list)}

    def _progress_cb(done: int, total: int, k: int, converged: bool) -> None:
        if verbose >= PROGRESS:
            elapsed = time.perf_counter() - t0
            bar = _progress_bar(done, total, prefix="  ", elapsed=elapsed)
            sys.stdout.write(bar)
            sys.stdout.flush()
            if done == total:
                sys.stdout.write("\n")
        update_queue.put((done, k, converged))

    def _run() -> None:
        results_holder.append(
            solver.solve_at_rpm(rpm, num_modes, hi, max_threads,
                                include_coriolis, min_frequency, _progress_cb,
                                memory_reserve_fraction=memory_reserve_fraction)
        )
        update_queue.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    done_count = 0
    while True:
        changed = False
        while True:
            try:
                item = update_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done_count = n_harmonics
                changed = True
                break
            done, k, converged = item
            done_count = done
            bar_idx = k_to_bar.get(k)
            if bar_idx is not None:
                bars[bar_idx].set_color(DONE_OK if converged else DONE_BAD)
                changed = True

        if changed:
            title_obj.set_text(
                f"Cyclic Solver  |  {rpm:.0f} RPM  |  {done_count} / {n_harmonics}"
            )
            fig.canvas.draw_idle()

        plt.pause(0.05)

        if done_count >= n_harmonics and not t.is_alive():
            break

    t.join()
    plt.ioff()
    return results_holder[0] if results_holder else []


def rpm_sweep(
    mesh: Mesh,
    material: Material,
    rpm_values: np.ndarray | Sequence[float],
    num_modes: int = 20,
    fluid: FluidConfig | None = None,
    verbose: int = SILENT,
    harmonic_indices: list[int] | None = None,
    max_threads: int = 0,
    hub_constraint: str = "fixed",
    include_coriolis: bool = False,
    min_frequency: float = 0.0,
    temperature: float | None = None,
    boundary_conditions: list[BoundaryCondition] | None = None,
    memory_reserve_fraction: float = 0.2,
) -> list[list[ModalResult]]:
    """Solve modal analysis over a range of RPM values.

    Parameters
    ----------
    mesh : Mesh with cyclic boundaries identified
    material : Material properties
    rpm_values : array of RPM values to solve at
    num_modes : number of modes per harmonic index
    fluid : fluid coupling configuration (None = dry)
    verbose : 0=silent, 1=progress bar, 2=detailed output
    harmonic_indices : list of harmonic indices to solve (None = all 0..N/2)
    max_threads : max concurrent solver threads (0 = auto, uses hardware_concurrency)
    hub_constraint : ``"fixed"`` clamps all hub DOFs (default).  ``"free"``
        leaves the hub unconstrained (free-rotating assembly, no shaft
        contact).  Centrifugal prestress is disabled when hub is free.
    include_coriolis : if True, include gyroscopic (Coriolis) coupling via
        Lancaster linearization of the QEP.
    min_frequency : minimum frequency threshold in Hz.  Modes below this are
        discarded as rigid body modes.  Set > 0 when using
        ``hub_constraint="free"``.
    temperature : bulk temperature in Kelvin.  When provided, the material
        stiffness is adjusted via ``material.at_temperature(temperature)``
        before solving.
    memory_reserve_fraction : fraction of total system RAM to keep free
        (0.0–1.0).  Default 0.2 (use up to 80 % of RAM).

    Returns
    -------
    List of lists: results[rpm_idx][harmonic_idx] = ModalResult
    """
    if temperature is not None:
        material = material.at_temperature(temperature)

    if fluid is None:
        fluid = FluidConfig()

    hi = harmonic_indices or []

    rpm_arr = np.asarray(rpm_values, dtype=np.float64)
    n_rpm = len(rpm_arr)

    if verbose >= PROGRESS:
        max_k = mesh.num_sectors // 2
        nd_str = f"ND {hi}" if hi else f"ND 0..{max_k}"
        coriolis_str = " [Coriolis]" if include_coriolis else ""
        temp_str = f"  T={temperature:.1f}K" if temperature is not None else ""
        print(f"RPM sweep{coriolis_str}{temp_str}: {n_rpm} points [{rpm_arr[0]:.0f} .. {rpm_arr[-1]:.0f}] RPM")
        print(
            f"  Mesh: {mesh.num_nodes()} nodes, {mesh.num_elements()} elements, "
            f"{mesh.num_sectors} sectors"
        )
        print(f"  Solving {nd_str}, {num_modes} modes/ND")
        print()

    if boundary_conditions is not None:
        constraint_groups = _build_constraint_groups(mesh, boundary_conditions)
        solver = CyclicSymmetrySolver(mesh, material, constraint_groups, fluid)
    else:
        apply_hub = hub_constraint == "fixed"
        solver = CyclicSymmetrySolver(mesh, material, fluid, apply_hub)
    all_results = []
    t_start = time.perf_counter()

    for i, rpm in enumerate(rpm_arr):
        t0 = time.perf_counter()
        results = _validate_modal_results(
            solver.solve_at_rpm(float(rpm), num_modes, hi, max_threads,
                                include_coriolis, min_frequency,
                                memory_reserve_fraction=memory_reserve_fraction))
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start

        all_results.append(results)

        if verbose == PROGRESS:
            bar = _progress_bar(
                i + 1,
                n_rpm,
                prefix="  ",
                elapsed=elapsed,
                suffix=f"  {rpm:7.0f} RPM ({dt:.1f}s)",
            )
            sys.stdout.write(bar)
            sys.stdout.flush()
            if i == n_rpm - 1:
                sys.stdout.write("\n")

        elif verbose >= DETAILED:
            print(
                f"  [{i+1}/{n_rpm}] {rpm:.0f} RPM  "
                f"({len(results)} harmonics, {dt:.1f}s)"
            )
            for r in results:
                freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:3])
                print(f"      ND={r.harmonic_index:2d}: [{freqs}] Hz")

    total = time.perf_counter() - t_start
    if verbose >= PROGRESS:
        print(f"  Sweep complete in {total:.1f}s " f"({total/n_rpm:.1f}s/point avg)")

    return all_results


def campbell_data(
    results: list[list[ModalResult]],
    num_sectors: int = 0,
) -> dict[str, np.ndarray]:
    """Extract Campbell diagram data from RPM sweep results.

    Converts rotating-frame eigenfrequencies to stationary-frame FW/BW
    frequencies for the Campbell diagram.

    Parameters
    ----------
    results : output from rpm_sweep()
    num_sectors : number of sectors (needed for FW/BW splitting).
        If 0, inferred from max harmonic index.

    Returns
    -------
    Dictionary with keys:
        'rpm' : (N,) array of RPM values
        'frequencies' : (N, H, M) array of frequencies in Hz (stationary frame)
        'harmonic_index' : (H,) array of harmonic indices
        'whirl_direction' : (N, H, M) array of whirl directions (+1=FW, -1=BW, 0=standing)
    where N=number of RPM points, H=number of harmonics, M=max modes per harmonic
    """
    if not results:
        return {
            "rpm": np.array([]),
            "frequencies": np.array([]),
            "harmonic_index": np.array([]),
            "whirl_direction": np.array([]),
        }

    n_rpm = len(results)

    # Find common harmonic indices and max modes across all RPM points
    all_harmonics = set()
    for row in results:
        for r in row:
            all_harmonics.add(r.harmonic_index)
    harmonic_index = np.array(sorted(all_harmonics))
    n_harmonics = len(harmonic_index)
    h_to_idx = {h: i for i, h in enumerate(harmonic_index)}

    # Infer num_sectors from max harmonic if not provided
    if num_sectors <= 0:
        max_k = int(harmonic_index[-1]) if len(harmonic_index) > 0 else 0
        num_sectors = 2 * max_k  # best guess: max_k = N/2

    max_k = num_sectors // 2

    # Find max modes after FW/BW splitting (2x for 0 < k < N/2)
    n_modes_rot = 0
    for row in results:
        for r in row:
            n_modes_rot = max(n_modes_rot, len(r.frequencies))
    n_modes = 2 * n_modes_rot  # FW+BW doubles the count

    rpm = np.zeros(n_rpm)
    frequencies = np.full((n_rpm, n_harmonics, n_modes), np.nan)
    whirl = np.zeros((n_rpm, n_harmonics, n_modes), dtype=np.int32)

    for i in range(n_rpm):
        if results[i]:
            rpm[i] = results[i][0].rpm
        omega = rpm[i] * 2.0 * np.pi / 60.0

        for r in results[i]:
            h_idx = h_to_idx.get(r.harmonic_index)
            if h_idx is None:
                continue

            k = r.harmonic_index
            f_rot = np.asarray(r.frequencies)
            whirl_rot = np.asarray(r.whirl_direction)
            nm = len(f_rot)

            # Check if Coriolis splitting is already present
            has_coriolis = np.any(whirl_rot != 0)

            if has_coriolis and abs(omega) > 0 and k > 0:
                # Coriolis modes: each mode already has FW/BW designation.
                # Convert to stationary frame per-mode.
                k_omega_hz = k * abs(omega) / (2.0 * np.pi)
                stat_f = np.abs(f_rot + whirl_rot * k_omega_hz)
                order = np.argsort(stat_f)
                n_out = min(n_modes, nm)
                frequencies[i, h_idx, :n_out] = stat_f[order][:n_out]
                whirl[i, h_idx, :n_out] = whirl_rot[order][:n_out]
            elif abs(omega) > 0 and 0 < k < max_k:
                # Kinematic splitting: duplicate each mode into FW + BW
                k_omega_hz = k * abs(omega) / (2.0 * np.pi)
                fw = f_rot + k_omega_hz
                bw = np.abs(f_rot - k_omega_hz)
                all_f = np.empty(2 * nm)
                all_w = np.empty(2 * nm, dtype=np.int32)
                all_f[0::2] = fw
                all_f[1::2] = bw
                all_w[0::2] = 1
                all_w[1::2] = -1
                order = np.argsort(all_f)
                n_out = min(n_modes, 2 * nm)
                frequencies[i, h_idx, :n_out] = all_f[order][:n_out]
                whirl[i, h_idx, :n_out] = all_w[order][:n_out]
            else:
                n_out = min(n_modes, nm)
                frequencies[i, h_idx, :n_out] = f_rot[:n_out]
                # whirl stays 0 (standing)

    return {
        "rpm": rpm,
        "frequencies": frequencies,
        "harmonic_index": harmonic_index,
        "whirl_direction": whirl,
    }

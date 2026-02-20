"""High-level solver interface for turbomodal."""

from __future__ import annotations

import sys
import time
from typing import Sequence

import numpy as np

from turbomodal._core import (
    CyclicSymmetrySolver,
    FluidConfig,
    Material,
    Mesh,
    ModalResult,
    SolverConfig,
)

# Verbosity levels
SILENT = 0
PROGRESS = 1
DETAILED = 2


def _progress_bar(
    current: int,
    total: int,
    width: int = 40,
    prefix: str = "",
    suffix: str = "",
    elapsed: float = 0.0,
) -> str:
    """Render a text progress bar."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + "." * (width - filled - 1)
    pct = f"{100 * frac:5.1f}%"

    eta_str = ""
    if current > 0 and elapsed > 0:
        eta = elapsed / current * (total - current)
        if eta >= 60:
            eta_str = f"  ETA {eta / 60:.1f}m"
        else:
            eta_str = f"  ETA {eta:.0f}s"

    return f"\r{prefix}[{bar}] {pct}  ({current}/{total}){suffix}{eta_str}"


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

    Returns
    -------
    List of ModalResult, one per harmonic index (0 to N/2)
    """
    if fluid is None:
        fluid = FluidConfig()

    apply_hub = hub_constraint == "fixed"
    hi = harmonic_indices or []

    if verbose >= PROGRESS:
        max_k = mesh.num_sectors // 2
        nd_str = f"ND {hi}" if hi else f"ND 0..{max_k}"
        print(
            f"Solving at {rpm:.0f} RPM  ({mesh.num_nodes()} nodes, "
            f"{mesh.num_elements()} elements, {mesh.num_sectors} sectors, "
            f"{nd_str}, {num_modes} modes/ND)"
        )

    t0 = time.perf_counter()
    solver = CyclicSymmetrySolver(mesh, material, fluid, apply_hub)
    results = solver.solve_at_rpm(rpm, num_modes, hi, max_threads)
    elapsed = time.perf_counter() - t0

    if verbose >= PROGRESS:
        print(f"  Solved {len(results)} harmonic indices in {elapsed:.1f}s")

    if verbose >= DETAILED:
        for r in results:
            freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:4])
            print(f"    ND={r.harmonic_index:2d}: [{freqs}] Hz (rotating frame)")

    return results


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

    Returns
    -------
    List of lists: results[rpm_idx][harmonic_idx] = ModalResult
    """
    if fluid is None:
        fluid = FluidConfig()

    apply_hub = hub_constraint == "fixed"
    hi = harmonic_indices or []

    rpm_arr = np.asarray(rpm_values, dtype=np.float64)
    n_rpm = len(rpm_arr)

    if verbose >= PROGRESS:
        max_k = mesh.num_sectors // 2
        nd_str = f"ND {hi}" if hi else f"ND 0..{max_k}"
        print(f"RPM sweep: {n_rpm} points [{rpm_arr[0]:.0f} .. {rpm_arr[-1]:.0f}] RPM")
        print(
            f"  Mesh: {mesh.num_nodes()} nodes, {mesh.num_elements()} elements, "
            f"{mesh.num_sectors} sectors"
        )
        print(f"  Solving {nd_str}, {num_modes} modes/ND")
        print()

    solver = CyclicSymmetrySolver(mesh, material, fluid, apply_hub)
    all_results = []
    t_start = time.perf_counter()

    for i, rpm in enumerate(rpm_arr):
        t0 = time.perf_counter()
        results = solver.solve_at_rpm(float(rpm), num_modes, hi, max_threads)
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
            nm = len(f_rot)

            if omega > 0 and 0 < k < max_k:
                k_omega_hz = k * omega / (2.0 * np.pi)
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

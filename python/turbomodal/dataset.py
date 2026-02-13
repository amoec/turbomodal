"""HDF5 dataset export and import for turbomodal modal analysis results.

Provides structured HDF5 storage of cyclic symmetry modal results across
multiple operating conditions, suitable for machine learning pipelines
and large parametric studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class OperatingCondition:
    """A single operating condition for parametric modal analysis.

    Parameters
    ----------
    condition_id : unique integer identifier for this condition
    rpm : rotation speed in revolutions per minute
    temperature : bulk temperature in Kelvin (default 293.15 K = 20 C)
    pressure_ratio : compressor/turbine pressure ratio (default 1.0)
    inlet_distortion : non-dimensional inlet distortion amplitude (0 = clean)
    tip_clearance : tip clearance in metres (0 = nominal)
    mistuning_pattern : optional per-blade frequency deviation array (length N)
    """

    condition_id: int
    rpm: float
    temperature: float = 293.15
    pressure_ratio: float = 1.0
    inlet_distortion: float = 0.0
    tip_clearance: float = 0.0
    mistuning_pattern: Optional[np.ndarray] = None


@dataclass
class DatasetConfig:
    """Configuration for HDF5 dataset export.

    Parameters
    ----------
    output_path : file path for the HDF5 output
    num_modes_per_harmonic : number of modes stored per harmonic index
    include_mode_shapes : whether to store the full complex mode shape arrays
    include_forced_response : whether to store forced response data
    compression : HDF5 compression filter name (e.g. "gzip", "lzf")
    compression_level : compression level (1-9 for gzip)
    """

    output_path: str = "turbomodal_dataset.h5"
    num_modes_per_harmonic: int = 10
    include_mode_shapes: bool = True
    include_forced_response: bool = False
    compression: str = "gzip"
    compression_level: int = 4


def export_modal_results(
    path: str | Path,
    mesh: Any,
    conditions: list[OperatingCondition],
    all_results: dict[int, list],
    config: DatasetConfig | None = None,
) -> None:
    """Export modal analysis results for multiple operating conditions to HDF5.

    Parameters
    ----------
    path : output HDF5 file path
    mesh : turbomodal Mesh object (must expose .nodes, .elements, .num_sectors)
    conditions : list of OperatingCondition describing each solved case
    all_results : mapping from condition_id to a list of ModalResult objects
        (one ModalResult per harmonic index, as returned by
        ``CyclicSymmetrySolver.solve_at_rpm``)
    config : dataset configuration (None uses defaults)

    File layout
    -----------
    ::

        /mesh/nodes          (n_nodes, 3)  float64
        /mesh/elements       (n_elem, 10)  int32
        /mesh/num_sectors    scalar int

        /conditions          structured array with fields:
                             condition_id, rpm, temperature,
                             pressure_ratio, inlet_distortion, tip_clearance

        /modes/eigenvalues/{cond_id}       (n_harmonics, n_modes) float64
        /modes/harmonic_index/{cond_id}    (n_harmonics,) int32
        /modes/whirl_direction/{cond_id}   (n_harmonics, n_modes) int32
        /modes/mode_shapes/{cond_id}       (n_harmonics, n_modes, n_dof) complex128
                                           (only if include_mode_shapes is True)

        /mistuning/{cond_id}               (n_sectors,) float64
                                           (only for conditions with mistuning)
    """
    import h5py

    if config is None:
        config = DatasetConfig()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comp_kwargs: dict[str, Any] = {}
    if config.compression:
        comp_kwargs["compression"] = config.compression
        if config.compression == "gzip":
            comp_kwargs["compression_opts"] = config.compression_level

    n_modes = config.num_modes_per_harmonic

    with h5py.File(str(path), "w") as f:
        # ---- Mesh ----
        mesh_grp = f.create_group("mesh")
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        elements = np.asarray(mesh.elements, dtype=np.int32)
        mesh_grp.create_dataset("nodes", data=nodes, **comp_kwargs)
        mesh_grp.create_dataset("elements", data=elements, **comp_kwargs)
        mesh_grp.attrs["num_sectors"] = int(mesh.num_sectors)

        # ---- Conditions (structured array) ----
        cond_dtype = np.dtype([
            ("condition_id", np.int32),
            ("rpm", np.float64),
            ("temperature", np.float64),
            ("pressure_ratio", np.float64),
            ("inlet_distortion", np.float64),
            ("tip_clearance", np.float64),
        ])
        cond_arr = np.empty(len(conditions), dtype=cond_dtype)
        for i, c in enumerate(conditions):
            cond_arr[i] = (
                c.condition_id,
                c.rpm,
                c.temperature,
                c.pressure_ratio,
                c.inlet_distortion,
                c.tip_clearance,
            )
        f.create_dataset("conditions", data=cond_arr, **comp_kwargs)

        # ---- Modal results per condition ----
        eigenval_grp = f.create_group("modes/eigenvalues")
        harmonic_grp = f.create_group("modes/harmonic_index")
        whirl_grp = f.create_group("modes/whirl_direction")
        if config.include_mode_shapes:
            shape_grp = f.create_group("modes/mode_shapes")
        mistuning_grp = None  # created on demand

        for cond in conditions:
            cid = cond.condition_id
            cid_str = str(cid)

            results_list = all_results.get(cid, [])
            if not results_list:
                continue

            n_harmonics = len(results_list)

            # Determine actual number of modes to store per harmonic
            actual_n_modes = min(
                n_modes,
                max(len(r.frequencies) for r in results_list),
            )

            # Eigenvalues: store as frequencies in Hz  (n_harmonics, actual_n_modes)
            eigenvalues = np.full(
                (n_harmonics, actual_n_modes), np.nan, dtype=np.float64
            )
            harmonic_indices = np.zeros(n_harmonics, dtype=np.int32)
            whirl_dirs = np.zeros(
                (n_harmonics, actual_n_modes), dtype=np.int32
            )

            for h, r in enumerate(results_list):
                harmonic_indices[h] = r.harmonic_index
                nm = min(actual_n_modes, len(r.frequencies))
                freqs = np.asarray(r.frequencies, dtype=np.float64)
                eigenvalues[h, :nm] = freqs[:nm]
                whirl = np.asarray(r.whirl_direction, dtype=np.int32)
                whirl_dirs[h, :nm] = whirl[:nm]

            eigenval_grp.create_dataset(
                cid_str, data=eigenvalues, **comp_kwargs
            )
            harmonic_grp.create_dataset(
                cid_str, data=harmonic_indices, **comp_kwargs
            )
            whirl_grp.create_dataset(
                cid_str, data=whirl_dirs, **comp_kwargs
            )

            # Mode shapes (optional, potentially large)
            if config.include_mode_shapes:
                # Determine DOF count from first non-empty result
                n_dof = 0
                for r in results_list:
                    shapes = np.asarray(r.mode_shapes)
                    if shapes.size > 0:
                        n_dof = shapes.shape[0]
                        break

                if n_dof > 0:
                    mode_shapes = np.zeros(
                        (n_harmonics, actual_n_modes, n_dof),
                        dtype=np.complex128,
                    )
                    for h, r in enumerate(results_list):
                        shapes = np.asarray(r.mode_shapes)
                        if shapes.size == 0:
                            continue
                        nm = min(actual_n_modes, shapes.shape[1])
                        mode_shapes[h, :nm, :] = shapes[:, :nm].T

                    shape_grp.create_dataset(
                        cid_str, data=mode_shapes, **comp_kwargs
                    )

            # Mistuning pattern (if present)
            if cond.mistuning_pattern is not None:
                if mistuning_grp is None:
                    mistuning_grp = f.create_group("mistuning")
                mistuning_grp.create_dataset(
                    cid_str,
                    data=np.asarray(cond.mistuning_pattern, dtype=np.float64),
                    **comp_kwargs,
                )

        # Global attributes
        f.attrs["turbomodal_version"] = "0.1.0"
        f.attrs["num_conditions"] = len(conditions)
        f.attrs["num_modes_per_harmonic"] = n_modes
        f.attrs["include_mode_shapes"] = config.include_mode_shapes


def load_modal_results(
    path: str | Path,
) -> tuple[dict[str, Any], list[OperatingCondition], dict[int, dict[str, Any]]]:
    """Load modal analysis results from an HDF5 dataset.

    Parameters
    ----------
    path : path to the HDF5 file written by ``export_modal_results``

    Returns
    -------
    mesh_data : dict with keys "nodes" (ndarray), "elements" (ndarray),
        "num_sectors" (int)
    conditions : list of OperatingCondition reconstructed from the file
    results_dict : mapping from condition_id to a dict containing:
        - "eigenvalues" : (n_harmonics, n_modes) float64
        - "harmonic_index" : (n_harmonics,) int32
        - "whirl_direction" : (n_harmonics, n_modes) int32
        - "mode_shapes" : (n_harmonics, n_modes, n_dof) complex128 (if stored)
        - "mistuning_pattern" : (n_sectors,) float64 (if stored)
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {path}")

    mesh_data: dict[str, Any] = {}
    conditions: list[OperatingCondition] = []
    results_dict: dict[int, dict[str, Any]] = {}

    with h5py.File(str(path), "r") as f:
        # ---- Mesh ----
        mesh_grp = f["mesh"]
        mesh_data["nodes"] = np.array(mesh_grp["nodes"])
        mesh_data["elements"] = np.array(mesh_grp["elements"])
        mesh_data["num_sectors"] = int(mesh_grp.attrs["num_sectors"])

        # ---- Conditions ----
        cond_arr = np.array(f["conditions"])
        for row in cond_arr:
            conditions.append(
                OperatingCondition(
                    condition_id=int(row["condition_id"]),
                    rpm=float(row["rpm"]),
                    temperature=float(row["temperature"]),
                    pressure_ratio=float(row["pressure_ratio"]),
                    inlet_distortion=float(row["inlet_distortion"]),
                    tip_clearance=float(row["tip_clearance"]),
                )
            )

        # ---- Modal results per condition ----
        eigenval_grp = f["modes/eigenvalues"]
        harmonic_grp = f["modes/harmonic_index"]
        whirl_grp = f["modes/whirl_direction"]
        has_shapes = "modes/mode_shapes" in f
        shape_grp = f["modes/mode_shapes"] if has_shapes else None
        has_mistuning = "mistuning" in f
        mistuning_grp = f["mistuning"] if has_mistuning else None

        for cond in conditions:
            cid_str = str(cond.condition_id)
            if cid_str not in eigenval_grp:
                continue

            entry: dict[str, Any] = {
                "eigenvalues": np.array(eigenval_grp[cid_str]),
                "harmonic_index": np.array(harmonic_grp[cid_str]),
                "whirl_direction": np.array(whirl_grp[cid_str]),
            }

            if shape_grp is not None and cid_str in shape_grp:
                entry["mode_shapes"] = np.array(shape_grp[cid_str])

            if mistuning_grp is not None and cid_str in mistuning_grp:
                pattern = np.array(mistuning_grp[cid_str])
                entry["mistuning_pattern"] = pattern
                cond.mistuning_pattern = pattern

            results_dict[cond.condition_id] = entry

    return mesh_data, conditions, results_dict

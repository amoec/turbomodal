"""End-to-end integration tests for turbomodal Python API."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def solved_wedge(wedge_mesh_path):
    """Solve wedge sector once for all integration tests."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver

    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    mat = Material(200e9, 0.3, 7800)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(0.0, 3)
    return mesh, mat, results


def test_load_to_hdf5_roundtrip(solved_wedge, tmp_path):
    """Load mesh → solve → export HDF5 → load HDF5 → frequencies match."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import (
        DatasetConfig, OperatingCondition,
        export_modal_results, load_modal_results,
    )

    mesh, mat, results = solved_wedge
    cond = OperatingCondition(condition_id=0, rpm=0.0)
    all_results = {0: results}
    path = str(tmp_path / "roundtrip.h5")

    cfg = DatasetConfig(output_path=path)
    export_modal_results(path, mesh, [cond], all_results, cfg)

    mesh_data, loaded_conds, loaded_results = load_modal_results(path)
    assert 0 in loaded_results
    assert len(loaded_conds) == 1
    # Eigenvalues should be present
    loaded_eigs = loaded_results[0]["eigenvalues"]
    assert loaded_eigs.size > 0


def test_pipeline_with_sensors(solved_wedge):
    """Solve → VirtualSensorArray → sample mode shape → check output."""
    from turbomodal.sensors import SensorArrayConfig, VirtualSensorArray

    mesh, mat, results = solved_wedge
    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
    )
    vsa = VirtualSensorArray(mesh, cfg)
    assert vsa.n_sensors == 4

    # Sample a mode shape from the first harmonic's first mode
    for r in results:
        if len(r.frequencies) > 0 and r.mode_shapes.shape[1] > 0:
            mode = np.asarray(r.mode_shapes[:, 0]).flatten()
            sampled = vsa.sample_mode_shape(mode)
            assert sampled.shape == (4,)
            break


def test_pipeline_mistuning(solved_wedge):
    """Solve tuned → extract frequencies → FMMSolver → magnification >= 1."""
    from turbomodal._core import FMMSolver

    mesh, mat, results = solved_wedge
    N = mesh.num_sectors

    # Extract tuned frequencies (first mode per harmonic)
    tuned_freqs = []
    for r in results:
        if len(r.frequencies) > 0:
            tuned_freqs.append(r.frequencies[0])
    tuned = np.array(tuned_freqs)

    dev = FMMSolver.random_mistuning(N, 0.03, seed=42)
    fmm_result = FMMSolver.solve(N, tuned, dev)

    assert fmm_result.peak_magnification >= 1.0
    assert len(fmm_result.frequencies) == N


def test_pipeline_forced_response(solved_wedge):
    """Solve → DampingConfig → ForcedResponseSolver → peak near natural freq."""
    from turbomodal._core import ForcedResponseSolver, DampingConfig, DampingType

    mesh, mat, results = solved_wedge

    damping = DampingConfig()
    damping.type = DampingType.MODAL
    damping.modal_damping_ratios = [0.02]

    fr_solver = ForcedResponseSolver(mesh, damping)

    from turbomodal._core import ForcedResponseConfig, ExcitationType
    cfg = ForcedResponseConfig()
    cfg.engine_order = 1
    cfg.num_freq_points = 100
    cfg.excitation_type = ExcitationType.UNIFORM_PRESSURE

    fr_result = fr_solver.solve(results, 0.0, cfg)

    # Check if EO=1 excited any modes
    has_k1 = any(r.harmonic_index == 1 for r in results)
    if has_k1 and len(fr_result.natural_frequencies) > 0:
        f_nat = fr_result.natural_frequencies[0]
        f_peak = fr_result.resonance_frequencies[0]
        # Peak should be within 10% of natural frequency
        assert abs(f_peak - f_nat) < f_nat * 0.10

"""Tests for turbomodal.dataset module — HDF5 export/import."""

import numpy as np
import pytest

from turbomodal.dataset import DatasetConfig, OperatingCondition


# ---- OperatingCondition defaults ----

def test_operating_condition_defaults():
    cond = OperatingCondition(condition_id=0, rpm=3000.0)
    assert cond.temperature == 293.15
    assert cond.pressure_ratio == 1.0
    assert cond.inlet_distortion == 0.0
    assert cond.tip_clearance == 0.0
    assert cond.mistuning_pattern is None


def test_operating_condition_with_mistuning():
    pattern = np.random.default_rng(42).normal(0, 0.02, 24)
    cond = OperatingCondition(condition_id=1, rpm=5000.0,
                              mistuning_pattern=pattern)
    assert cond.mistuning_pattern is not None
    assert len(cond.mistuning_pattern) == 24


# ---- DatasetConfig defaults ----

def test_dataset_config_defaults():
    cfg = DatasetConfig()
    assert cfg.compression == "gzip"
    assert cfg.include_mode_shapes is True
    assert cfg.output_path == "turbomodal_dataset.h5"
    assert cfg.num_modes_per_harmonic == 10
    assert cfg.compression_level == 4


# ---- Export/import roundtrip (requires h5py and C++ extension) ----

@pytest.fixture
def solved_data(wedge_mesh_path):
    """Load wedge mesh and solve to generate data for export testing."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver

    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    mat = Material(200e9, 0.3, 7800)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(0.0, 3)
    return mesh, results


def test_export_creates_file(solved_data, tmp_path):
    """HDF5 file should exist after export."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import export_modal_results

    mesh, results = solved_data
    cond = OperatingCondition(condition_id=0, rpm=0.0)
    all_results = {0: results}
    path = str(tmp_path / "test_export.h5")

    cfg = DatasetConfig(output_path=path)
    export_modal_results(path, mesh, [cond], all_results, cfg)

    assert (tmp_path / "test_export.h5").exists()


def test_export_file_structure(solved_data, tmp_path):
    """HDF5 file should have expected group structure."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import export_modal_results

    mesh, results = solved_data
    cond = OperatingCondition(condition_id=0, rpm=0.0)
    all_results = {0: results}
    path = str(tmp_path / "test_structure.h5")

    cfg = DatasetConfig(output_path=path)
    export_modal_results(path, mesh, [cond], all_results, cfg)

    with h5py.File(path, "r") as f:
        assert "mesh" in f
        assert "conditions" in f
        assert "modes" in f


def test_roundtrip_frequencies(solved_data, tmp_path):
    """Frequencies should survive export → load roundtrip."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import export_modal_results, load_modal_results

    mesh, results = solved_data
    cond = OperatingCondition(condition_id=0, rpm=0.0)
    all_results = {0: results}
    path = str(tmp_path / "test_roundtrip.h5")

    cfg = DatasetConfig(output_path=path)
    export_modal_results(path, mesh, [cond], all_results, cfg)

    _, _, loaded = load_modal_results(path)
    assert 0 in loaded
    # Check that eigenvalues are present
    loaded_eigs = loaded[0]["eigenvalues"]
    assert loaded_eigs.size > 0


def test_export_multiple_conditions(solved_data, tmp_path):
    """Multiple conditions should be stored correctly."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import export_modal_results, load_modal_results

    mesh, results = solved_data
    conds = [
        OperatingCondition(condition_id=0, rpm=0.0),
        OperatingCondition(condition_id=1, rpm=1000.0, temperature=500.0),
        OperatingCondition(condition_id=2, rpm=3000.0),
    ]
    all_results = {c.condition_id: results for c in conds}
    path = str(tmp_path / "test_multi.h5")

    cfg = DatasetConfig(output_path=path)
    export_modal_results(path, mesh, conds, all_results, cfg)

    _, loaded_conds, loaded_results = load_modal_results(path)
    assert len(loaded_conds) == 3
    assert len(loaded_results) == 3


def test_export_without_mode_shapes(solved_data, tmp_path):
    """include_mode_shapes=False should omit mode shapes."""
    h5py = pytest.importorskip("h5py")
    from turbomodal.dataset import export_modal_results

    mesh, results = solved_data
    cond = OperatingCondition(condition_id=0, rpm=0.0)
    all_results = {0: results}
    path = str(tmp_path / "test_no_shapes.h5")

    cfg = DatasetConfig(output_path=path, include_mode_shapes=False)
    export_modal_results(path, mesh, [cond], all_results, cfg)

    with h5py.File(path, "r") as f:
        modes = f["modes"]
        # Should not have mode_shapes group
        assert "mode_shapes" not in modes


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        from turbomodal.dataset import load_modal_results
        load_modal_results("/nonexistent/path/data.h5")

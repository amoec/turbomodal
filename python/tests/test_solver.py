"""Tests for turbomodal.solver module — Python solver wrapper."""

import numpy as np
import pytest

from turbomodal.solver import solve, rpm_sweep, campbell_data


@pytest.fixture(scope="module")
def wedge_mesh(wedge_mesh_path):
    """Load wedge mesh once for all tests in this module."""
    from turbomodal._core import Mesh
    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()
    return mesh


@pytest.fixture(scope="module")
def steel_material():
    from turbomodal._core import Material
    return Material(200e9, 0.3, 7800)


# ---- solve ----

def test_solve_returns_modal_results(wedge_mesh, steel_material):
    results = solve(wedge_mesh, steel_material, rpm=0.0, num_modes=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_solve_num_modes_respected(wedge_mesh, steel_material):
    results = solve(wedge_mesh, steel_material, rpm=0.0, num_modes=3)
    for r in results:
        assert len(r.frequencies) <= 3


def test_solve_zero_rpm_standing_waves(wedge_mesh, steel_material):
    results = solve(wedge_mesh, steel_material, rpm=0.0, num_modes=3)
    for r in results:
        if r.harmonic_index == 0:
            for m in range(len(r.whirl_direction)):
                assert r.whirl_direction[m] == 0


def test_solve_nonzero_rpm_whirl_split(wedge_mesh, steel_material):
    results = solve(wedge_mesh, steel_material, rpm=3000.0, num_modes=3)
    found_fw = False
    found_bw = False
    for r in results:
        if r.harmonic_index > 0 and r.harmonic_index < wedge_mesh.num_sectors // 2:
            for m in range(len(r.whirl_direction)):
                if r.whirl_direction[m] > 0:
                    found_fw = True
                if r.whirl_direction[m] < 0:
                    found_bw = True
    if any(r.harmonic_index > 0 for r in results):
        assert found_fw, "Expected FW modes at non-zero RPM"
        assert found_bw, "Expected BW modes at non-zero RPM"


def test_solve_verbose_levels(wedge_mesh, steel_material):
    for v in [0, 1, 2]:
        results = solve(wedge_mesh, steel_material, rpm=0.0, num_modes=2, verbose=v)
        assert len(results) > 0


def test_solve_config_override(wedge_mesh, steel_material):
    from turbomodal._core import SolverConfig
    cfg = SolverConfig()
    cfg.nev = 3
    results = solve(wedge_mesh, steel_material, rpm=0.0, num_modes=3, config=cfg)
    assert len(results) > 0


# ---- rpm_sweep ----

def test_rpm_sweep_shape(wedge_mesh, steel_material):
    rpm_vals = [0.0, 1000.0, 3000.0]
    all_results = rpm_sweep(wedge_mesh, steel_material, rpm_vals, num_modes=2)
    assert len(all_results) == 3
    for results in all_results:
        assert isinstance(results, list)


def test_rpm_sweep_stiffening(wedge_mesh, steel_material):
    """ND=0 frequency should increase (or stay same) with RPM."""
    rpm_vals = [0.0, 5000.0]
    all_results = rpm_sweep(wedge_mesh, steel_material, rpm_vals, num_modes=2)

    # Find ND=0 first frequency at each RPM
    f_0rpm = None
    f_5krpm = None
    for r in all_results[0]:
        if r.harmonic_index == 0 and len(r.frequencies) > 0:
            f_0rpm = r.frequencies[0]
            break
    for r in all_results[1]:
        if r.harmonic_index == 0 and len(r.frequencies) > 0:
            f_5krpm = r.frequencies[0]
            break

    if f_0rpm is not None and f_5krpm is not None:
        assert f_5krpm >= f_0rpm * 0.95  # allow small numerical noise


# ---- campbell_data ----

def test_campbell_data_structure(wedge_mesh, steel_material):
    rpm_vals = [0.0, 1000.0]
    all_results = rpm_sweep(wedge_mesh, steel_material, rpm_vals, num_modes=2)
    cd = campbell_data(all_results)
    assert isinstance(cd, dict)
    assert "rpm" in cd
    assert "frequencies" in cd


def test_campbell_data_rpm_values(wedge_mesh, steel_material):
    rpm_vals = [0.0, 1500.0, 3000.0]
    all_results = rpm_sweep(wedge_mesh, steel_material, rpm_vals, num_modes=2)
    cd = campbell_data(all_results)
    np.testing.assert_array_almost_equal(cd["rpm"], rpm_vals)


def test_campbell_data_empty():
    cd = campbell_data([])
    assert len(cd["rpm"]) == 0

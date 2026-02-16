"""Shared fixtures for turbomodal Python tests."""

import os
from pathlib import Path

import pytest

# Path to C++ test data directory
TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "test_data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "validation: slow validation tests against analytical benchmarks")
    config.addinivalue_line("markers", "slow: tests taking more than a few seconds")


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def wedge_mesh_path(test_data_dir):
    """Path to the wedge sector mesh file."""
    p = test_data_dir / "wedge_sector.msh"
    if not p.exists():
        pytest.skip(f"Test mesh not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def leissa_mesh_path(test_data_dir):
    """Path to the Leissa disk sector mesh file."""
    p = test_data_dir / "leissa_disk_sector.msh"
    if not p.exists():
        pytest.skip(f"Test mesh not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def test_step_path(test_data_dir):
    """Path to a test STEP CAD file."""
    p = test_data_dir / "test_sector.step"
    if not p.exists():
        pytest.skip(f"Test CAD file not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def solved_wedge(wedge_mesh_path):
    """Solve wedge sector once for entire test session."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver

    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()
    mat = Material(200e9, 0.3, 7800)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(0, 5)
    return mesh, mat, results

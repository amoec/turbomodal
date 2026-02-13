"""Shared fixtures for turbomodal Python tests."""

import os
from pathlib import Path

import pytest

# Path to C++ test data directory
TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "test_data"


@pytest.fixture
def test_data_dir():
    """Return path to the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def wedge_mesh_path(test_data_dir):
    """Path to the wedge sector mesh file."""
    p = test_data_dir / "wedge_sector.msh"
    if not p.exists():
        pytest.skip(f"Test mesh not found: {p}")
    return str(p)


@pytest.fixture
def leissa_mesh_path(test_data_dir):
    """Path to the Leissa disk sector mesh file."""
    p = test_data_dir / "leissa_disk_sector.msh"
    if not p.exists():
        pytest.skip(f"Test mesh not found: {p}")
    return str(p)

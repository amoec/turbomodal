"""Tests for turbomodal C++ Python bindings."""

import numpy as np
import pytest


class TestMaterial:
    def test_basic_construction(self):
        from turbomodal._core import Material
        mat = Material(200e9, 0.3, 7800)
        assert mat.E == 200e9
        assert mat.nu == 0.3
        assert mat.rho == 7800

    def test_temperature_dependent(self):
        from turbomodal._core import Material
        mat = Material(200e9, 0.3, 7800, 293.15, -50e6)
        assert mat.T_ref == 293.15
        assert mat.E_slope == -50e6

    def test_at_temperature(self):
        from turbomodal._core import Material
        mat = Material(200e9, 0.3, 7800, 293.15, -50e6)
        mat_hot = mat.at_temperature(500)
        # E should decrease at higher temperature
        assert mat_hot.E < mat.E

    def test_validate(self):
        from turbomodal._core import Material
        mat = Material(200e9, 0.3, 7800)
        mat.validate()  # should not raise

    def test_repr(self):
        from turbomodal._core import Material
        mat = Material(200e9, 0.3, 7800)
        assert "Material" in repr(mat)


class TestNodeSet:
    def test_construction(self):
        from turbomodal._core import NodeSet
        ns = NodeSet()
        ns.name = "test"
        ns.node_ids = [0, 1, 2]
        assert ns.name == "test"
        assert ns.node_ids == [0, 1, 2]


class TestMesh:
    def test_load_from_gmsh(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0
        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 10

    def test_load_from_arrays(self, wedge_mesh_path):
        from turbomodal._core import Mesh, NodeSet
        # First load via gmsh to get reference data
        ref = Mesh()
        ref.num_sectors = 24
        ref.load_from_gmsh(wedge_mesh_path)
        ref.identify_cyclic_boundaries()
        ref.match_boundary_nodes()

        # Now load via arrays
        mesh = Mesh()
        mesh.load_from_arrays(
            np.asarray(ref.nodes),
            np.asarray(ref.elements),
            ref.node_sets,
            24,
        )
        assert mesh.num_nodes() == ref.num_nodes()
        assert mesh.num_elements() == ref.num_elements()
        assert len(mesh.left_boundary) > 0
        assert len(mesh.right_boundary) > 0

    def test_num_dof(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        assert mesh.num_dof() == 3 * mesh.num_nodes()

    def test_repr(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        assert "Mesh" in repr(mesh)


class TestSolverConfig:
    def test_defaults(self):
        from turbomodal._core import SolverConfig
        cfg = SolverConfig()
        assert cfg.nev == 20
        assert cfg.tolerance == 1e-8

    def test_modify(self):
        from turbomodal._core import SolverConfig
        cfg = SolverConfig()
        cfg.nev = 10
        cfg.shift = 100.0
        assert cfg.nev == 10
        assert cfg.shift == 100.0


class TestFluidConfig:
    def test_defaults(self):
        from turbomodal._core import FluidConfig, FluidType
        fc = FluidConfig()
        assert fc.type == FluidType.NONE
        assert fc.fluid_density == 0.0


class TestCyclicSymmetrySolver:
    def test_solve_at_rpm(self, wedge_mesh_path):
        from turbomodal._core import Mesh, Material, CyclicSymmetrySolver
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        mat = Material(200e9, 0.3, 7800)
        solver = CyclicSymmetrySolver(mesh, mat)
        results = solver.solve_at_rpm(0, 5)

        # Should get one result per harmonic index (0 to N/2)
        assert len(results) > 0
        for r in results:
            assert len(r.frequencies) > 0
            assert r.frequencies[0] >= 0
            assert r.mode_shapes.shape[0] > 0  # reduced DOF after cyclic BCs
            assert isinstance(r.whirl_direction[0], (int, np.integer))


class TestModalResult:
    def test_fields(self, wedge_mesh_path):
        from turbomodal._core import Mesh, Material, CyclicSymmetrySolver
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        mat = Material(200e9, 0.3, 7800)
        solver = CyclicSymmetrySolver(mesh, mat)
        results = solver.solve_at_rpm(0, 3)

        r = results[0]
        assert r.harmonic_index == 0
        assert r.rpm == 0.0
        assert isinstance(r.frequencies, np.ndarray)
        assert r.frequencies.dtype == np.float64
        assert r.mode_shapes.dtype == np.complex128
        assert "ModalResult" in repr(r)

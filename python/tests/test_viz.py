"""Tests for turbomodal.viz visualization module."""

import numpy as np
import pytest


@pytest.fixture
def solved_result(wedge_mesh_path):
    """Solve a small problem to get ModalResult for viz testing."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver
    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    mat = Material(200e9, 0.3, 7800)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(0, 3)
    return mesh, results


class TestPlotMesh:
    def test_returns_plotter(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        from turbomodal.viz import plot_mesh
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        plotter = plot_mesh(mesh, off_screen=True)
        assert plotter is not None
        plotter.close()


class TestPlotMode:
    def test_returns_plotter(self, solved_result):
        from turbomodal.viz import plot_mode
        mesh, results = solved_result
        plotter = plot_mode(mesh, results[0], mode_index=0, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_components(self, solved_result):
        from turbomodal.viz import plot_mode
        mesh, results = solved_result
        for comp in ("magnitude", "x", "y", "z"):
            plotter = plot_mode(mesh, results[0], mode_index=0,
                                component=comp, off_screen=True)
            plotter.close()


class TestPlotFullAnnulus:
    def test_returns_plotter(self, solved_result):
        from turbomodal.viz import plot_full_annulus
        mesh, results = solved_result
        plotter = plot_full_annulus(mesh, results[0], mode_index=0, off_screen=True)
        assert plotter is not None
        plotter.close()


class TestPlotCad:
    def test_single_sector(self, test_step_path):
        from turbomodal.viz import plot_cad
        plotter = plot_cad(test_step_path, num_sectors=24, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_full_disk(self, test_step_path):
        from turbomodal.viz import plot_cad
        plotter = plot_cad(
            test_step_path, num_sectors=24,
            show_full_disk=True, off_screen=True,
        )
        assert plotter is not None
        plotter.close()

    def test_no_dimensions(self, test_step_path):
        from turbomodal.viz import plot_cad
        plotter = plot_cad(
            test_step_path, num_sectors=24,
            show_dimensions=False, off_screen=True,
        )
        assert plotter is not None
        plotter.close()


class TestPlotFullMesh:
    def test_returns_plotter(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        from turbomodal.viz import plot_full_mesh
        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        plotter = plot_full_mesh(mesh, off_screen=True)
        assert plotter is not None
        plotter.close()


class TestPlotCampbell:
    def test_with_mock_data(self):
        from turbomodal._core import ModalResult
        from turbomodal.viz import plot_campbell

        # Create mock results for 3 RPM points, 2 harmonics
        results = []
        for rpm in [0, 5000, 10000]:
            row = []
            for nd in [0, 1]:
                r = ModalResult()
                r.harmonic_index = nd
                r.rpm = float(rpm)
                r.frequencies = np.array([100.0 + rpm * 0.001, 200.0 + rpm * 0.002])
                r.whirl_direction = np.array([0, 1] if nd == 0 else [-1, 1])
                r.mode_shapes = np.zeros((6, 2), dtype=complex)
                row.append(r)
            results.append(row)

        fig = plot_campbell(results, engine_orders=[1, 2])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotZzenf:
    def test_with_mock_data(self):
        from turbomodal._core import ModalResult
        from turbomodal.viz import plot_zzenf

        results = []
        for nd in range(4):
            r = ModalResult()
            r.harmonic_index = nd
            r.rpm = 5000.0
            r.frequencies = np.array([100.0 * (nd + 1), 200.0 * (nd + 1)])
            r.whirl_direction = np.array([-1, 1] if nd > 0 else [0, 0])
            r.mode_shapes = np.zeros((6, 2), dtype=complex)
            results.append(r)

        fig = plot_zzenf(results, num_sectors=24)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

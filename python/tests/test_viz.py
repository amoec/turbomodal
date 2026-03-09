"""Tests for turbomodal.viz visualization module."""

import os

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

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="GIF rendering requires a GPU/display context unavailable on CI",
    )
    def test_animate_saves_gif(self, solved_result, tmp_path):
        from turbomodal.viz import plot_mode
        mesh, results = solved_result
        gif_path = str(tmp_path / "mode.gif")
        plotter = plot_mode(mesh, results[0], mode_index=0, scale=0.002,
                            animate=True, n_frames=4, filename=gif_path,
                            off_screen=True)
        assert (tmp_path / "mode.gif").exists()
        assert (tmp_path / "mode.gif").stat().st_size > 0


class TestPlotFullAnnulus:
    def test_returns_plotter(self, solved_result):
        from turbomodal.viz import plot_mode
        mesh, results = solved_result
        plotter = plot_mode(mesh, results[0], mode_index=0, off_screen=True,
                            full_annulus=True)
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

    def test_rotation_axis_override(self, test_step_path):
        from turbomodal.viz import plot_cad
        plotter = plot_cad(
            test_step_path, num_sectors=24,
            rotation_axis=2, off_screen=True,
        )
        assert plotter is not None
        plotter.close()

    def test_units_override(self, test_step_path):
        from turbomodal.viz import plot_cad
        plotter = plot_cad(
            test_step_path, num_sectors=24,
            units="mm", off_screen=True,
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


class TestPlotSensors:
    @pytest.fixture
    def mesh_and_sensors(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        from turbomodal.sensors import SensorArrayConfig, VirtualSensorArray

        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        btt_cfg = SensorArrayConfig.default_btt_array(
            num_probes=4, casing_radius=0.15,
            axial_positions=[0.005], sample_rate=100_000.0, duration=0.01,
        )
        btt_vsa = VirtualSensorArray(mesh, btt_cfg)
        return mesh, btt_vsa

    def test_returns_plotter(self, mesh_and_sensors):
        from turbomodal.viz import plot_sensors
        mesh, vsa = mesh_and_sensors
        plotter = plot_sensors(mesh, vsa, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_no_mesh_background(self, mesh_and_sensors):
        from turbomodal.viz import plot_sensors
        mesh, vsa = mesh_and_sensors
        plotter = plot_sensors(mesh, vsa, show_mesh=False, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_no_directions(self, mesh_and_sensors):
        from turbomodal.viz import plot_sensors
        mesh, vsa = mesh_and_sensors
        plotter = plot_sensors(mesh, vsa, show_directions=False, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_full_annulus(self, mesh_and_sensors):
        from turbomodal.viz import plot_sensors
        mesh, vsa = mesh_and_sensors
        plotter = plot_sensors(mesh, vsa, full_annulus=True, off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_uniform_color(self, mesh_and_sensors):
        from turbomodal.viz import plot_sensors
        mesh, vsa = mesh_and_sensors
        plotter = plot_sensors(mesh, vsa, color_by="uniform", off_screen=True)
        assert plotter is not None
        plotter.close()

    def test_mixed_sensor_types(self, wedge_mesh_path):
        from turbomodal._core import Mesh
        from turbomodal.sensors import (
            SensorArrayConfig, SensorLocation, SensorType, VirtualSensorArray,
        )
        from turbomodal.viz import plot_sensors

        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        cfg = SensorArrayConfig(sensors=[
            SensorLocation(
                sensor_type=SensorType.BTT_PROBE,
                position=np.array([0.15, 0.0, 0.005]),
                direction=np.array([-1.0, 0.0, 0.0]),
                label="BTT_0",
            ),
            SensorLocation(
                sensor_type=SensorType.DISPLACEMENT,
                position=np.array([0.08, 0.0, 0.005]),
                direction=np.array([0.0, 0.0, -1.0]),
                label="Disp_0",
            ),
        ])
        vsa = VirtualSensorArray(mesh, cfg)
        plotter = plot_sensors(mesh, vsa, off_screen=True)
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


class TestDiagramStyle:
    def test_default_construction(self):
        from turbomodal.viz import DiagramStyle
        s = DiagramStyle()
        assert s.eo_linewidth == 1.5
        assert s.eo_alpha == 0.55
        assert s.crossing_color == "red"
        assert s.colormap is None

    def test_custom_values(self):
        from turbomodal.viz import DiagramStyle
        s = DiagramStyle(eo_linewidth=3.0, eo_color="blue", colormap="viridis")
        assert s.eo_linewidth == 3.0
        assert s.eo_color == "blue"
        assert s.colormap == "viridis"


class TestPlotZzenfStyled:
    def _make_results(self, num_sectors=24):
        from turbomodal._core import ModalResult
        results = []
        for nd in range(num_sectors // 2 + 1):
            r = ModalResult()
            r.harmonic_index = nd
            r.rpm = 5000.0
            r.frequencies = np.array([100.0 * (nd + 1), 200.0 * (nd + 1)])
            r.whirl_direction = np.array([-1, 1] if nd > 0 else [0, 0])
            r.mode_shapes = np.zeros((6, 2), dtype=complex)
            results.append(r)
        return results

    def test_custom_style(self):
        from turbomodal.viz import plot_zzenf, DiagramStyle
        style = DiagramStyle(eo_linewidth=3.0, family_linewidth=2.5)
        results = self._make_results()
        fig = plot_zzenf(results, num_sectors=24, style=style)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_engine_orders(self):
        from turbomodal.viz import plot_zzenf
        results = self._make_results()
        fig = plot_zzenf(results, num_sectors=24, engine_orders=[1, 2, 24])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_stator_vanes(self):
        from turbomodal.viz import plot_zzenf
        results = self._make_results()
        fig = plot_zzenf(results, num_sectors=24, stator_vanes=36)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_eo_lines_deprecation(self):
        from turbomodal.viz import plot_zzenf
        results = self._make_results()
        with pytest.warns(DeprecationWarning, match="eo_lines is deprecated"):
            fig = plot_zzenf(results, num_sectors=24, eo_lines=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_crossing_markers_with_engine_orders(self):
        from turbomodal.viz import plot_zzenf
        results = self._make_results()
        fig = plot_zzenf(results, num_sectors=24, engine_orders=[1, 2],
                         crossing_markers=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotCampbellStyled:
    def _make_results(self):
        from turbomodal._core import ModalResult
        results = []
        for rpm in [0, 5000, 10000]:
            row = []
            for nd in [0, 1]:
                r = ModalResult()
                r.harmonic_index = nd
                r.rpm = float(rpm)
                r.frequencies = np.array([100.0 + rpm * 0.001,
                                          200.0 + rpm * 0.002])
                r.whirl_direction = np.array([0, 1] if nd == 0 else [-1, 1])
                r.mode_shapes = np.zeros((6, 2), dtype=complex)
                row.append(r)
            results.append(row)
        return results

    def test_custom_style(self):
        from turbomodal.viz import plot_campbell, DiagramStyle
        style = DiagramStyle(eo_linewidth=3.0, mode_linewidth_fw=2.5)
        results = self._make_results()
        fig = plot_campbell(results, engine_orders=[1, 2], style=style)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_stator_vanes(self):
        from turbomodal.viz import plot_campbell
        results = self._make_results()
        fig = plot_campbell(results, stator_vanes=36)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDiagnoseFrequencies:
    def _make_results(self, num_sectors=24, n_modes=3):
        from turbomodal._core import ModalResult
        results = []
        for nd in range(num_sectors // 2 + 1):
            r = ModalResult()
            r.harmonic_index = nd
            r.rpm = 5000.0
            freqs = [100.0 * (nd + 1) * (m + 1) for m in range(n_modes)]
            r.frequencies = np.array(freqs)
            r.whirl_direction = np.zeros(n_modes)
            r.mode_shapes = np.zeros((6, n_modes), dtype=complex)
            results.append(r)
        return results

    def test_zero_error_at_perfect_match(self):
        from turbomodal.viz import diagnose_frequencies
        results = self._make_results(num_sectors=8, n_modes=2)
        # Build GT from the same frequencies
        N = 8
        half_N = N // 2
        gt = np.full((half_N + 1, 2), np.nan)
        for r in results:
            nd = r.harmonic_index
            if nd <= half_N:
                gt[nd] = sorted(r.frequencies)
        diag = diagnose_frequencies(results, gt, num_sectors=N)
        assert diag["worst_mode"][2] == pytest.approx(0.0, abs=1e-10)
        assert np.nanmax(diag["error_matrix"]) == pytest.approx(0.0, abs=1e-10)
        assert "figures" in diag
        assert len(diag["figures"]) == 3
        import matplotlib.pyplot as plt
        for f in diag["figures"]:
            plt.close(f)

    def test_nan_gt_handling(self):
        from turbomodal.viz import diagnose_frequencies
        results = self._make_results(num_sectors=8, n_modes=2)
        gt = np.full((5, 2), np.nan)
        gt[0, 0] = results[0].frequencies[0]
        diag = diagnose_frequencies(results, gt, num_sectors=8)
        # Only one valid comparison (ND=0, mode=0) with 0% error
        assert diag["worst_mode"][2] == pytest.approx(0.0, abs=1e-10)
        import matplotlib.pyplot as plt
        for f in diag.get("figures", []):
            plt.close(f)

    def test_rpm_sweep_input(self):
        from turbomodal.viz import diagnose_frequencies
        # Wrap single-RPM results in a sweep list
        results = self._make_results(num_sectors=8, n_modes=2)
        sweep = [results, results]  # two RPM points, same data
        gt = np.full((5, 2), np.nan)
        gt[0] = sorted(results[0].frequencies)
        diag = diagnose_frequencies(sweep, gt, num_sectors=8, rpm_index=0)
        assert diag["worst_mode"][2] == pytest.approx(0.0, abs=1e-10)
        import matplotlib.pyplot as plt
        for f in diag.get("figures", []):
            plt.close(f)

    def test_return_figures_false(self):
        from turbomodal.viz import diagnose_frequencies
        results = self._make_results(num_sectors=8, n_modes=2)
        gt = np.full((5, 2), np.nan)
        gt[0] = sorted(results[0].frequencies)
        diag = diagnose_frequencies(results, gt, num_sectors=8,
                                    return_figures=False)
        assert "figures" not in diag
        assert "summary" in diag
        assert "computed_matrix" in diag


class TestPlotSensorSignals:
    @pytest.fixture
    def signal_setup(self, wedge_mesh_path):
        from turbomodal._core import Mesh, Material, CyclicSymmetrySolver
        from turbomodal.sensors import SensorArrayConfig, VirtualSensorArray

        mesh = Mesh()
        mesh.num_sectors = 24
        mesh.load_from_gmsh(wedge_mesh_path)
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()

        mat = Material(200e9, 0.3, 7800)
        solver = CyclicSymmetrySolver(mesh, mat)
        results = solver.solve_at_rpm(3000.0, 3)

        cfg = SensorArrayConfig.default_btt_array(
            num_probes=4, casing_radius=0.15,
            axial_positions=[0.005], sample_rate=100_000.0, duration=0.01,
        )
        vsa = VirtualSensorArray(mesh, cfg)
        return mesh, vsa, results

    def test_returns_figure(self, signal_setup):
        import matplotlib
        import matplotlib.pyplot as plt
        from turbomodal.viz import plot_sensor_signals

        mesh, vsa, results = signal_setup
        fig = plot_sensor_signals(vsa, results, 3000.0)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_filter_by_nd(self, signal_setup):
        import matplotlib.pyplot as plt
        from turbomodal.viz import plot_sensor_signals

        mesh, vsa, results = signal_setup
        fig = plot_sensor_signals(vsa, results, 3000.0, nd=1)
        axes = fig.get_axes()
        assert len(axes) == vsa.n_sensors
        plt.close(fig)

    def test_filter_by_nd_and_nc(self, signal_setup):
        import matplotlib.pyplot as plt
        from turbomodal.viz import plot_sensor_signals

        mesh, vsa, results = signal_setup
        fig = plot_sensor_signals(vsa, results, 3000.0, mesh=mesh, nd=1, nc=0)
        assert "ND=1" in fig.texts[0].get_text()
        assert "NC=0" in fig.texts[0].get_text()
        plt.close(fig)

    def test_sensor_subset(self, signal_setup):
        import matplotlib.pyplot as plt
        from turbomodal.viz import plot_sensor_signals

        mesh, vsa, results = signal_setup
        fig = plot_sensor_signals(vsa, results, 3000.0, sensors=[0, 2])
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

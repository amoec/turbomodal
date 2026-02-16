"""Tests for turbomodal.io mesh import module."""

import numpy as np
import pytest


class TestLoadMesh:
    def test_load_gmsh_msh(self, wedge_mesh_path):
        from turbomodal.io import load_mesh
        mesh = load_mesh(wedge_mesh_path, num_sectors=24)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0
        assert len(mesh.left_boundary) > 0
        assert len(mesh.right_boundary) > 0
        assert mesh.num_sectors == 24

    def test_load_leissa_msh(self, leissa_mesh_path):
        from turbomodal.io import load_mesh
        mesh = load_mesh(leissa_mesh_path, num_sectors=24)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0

    def test_file_not_found(self):
        from turbomodal.io import load_mesh
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/file.msh", num_sectors=24)


class TestLoadCad:
    def test_unsupported_format(self, tmp_path):
        from turbomodal.io import load_cad
        # Create a dummy file with unsupported extension
        dummy = tmp_path / "model.obj"
        dummy.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported CAD format"):
            load_cad(str(dummy), num_sectors=24)

    def test_stl_rejected(self, tmp_path):
        from turbomodal.io import load_cad
        dummy = tmp_path / "model.stl"
        dummy.write_text("dummy")
        with pytest.raises(ValueError, match="STL files are surface meshes"):
            load_cad(str(dummy), num_sectors=24)

    def test_file_not_found(self):
        from turbomodal.io import load_cad
        with pytest.raises(FileNotFoundError):
            load_cad("/nonexistent/file.step", num_sectors=24)


class TestInspectCad:
    def test_returns_cad_info(self, test_step_path):
        from turbomodal.io import inspect_cad, CadInfo
        info = inspect_cad(test_step_path, num_sectors=24)
        assert isinstance(info, CadInfo)
        assert info.num_sectors == 24
        assert info.sector_angle_deg == pytest.approx(15.0)
        assert info.outer_radius > 0
        assert info.axial_length > 0
        assert info.recommended_mesh_size > 0
        assert info.num_surfaces > 0
        assert info.num_volumes > 0
        # Axis and unit detection
        assert info.rotation_axis in (0, 1, 2)
        assert info.detected_unit == "mm"  # test_sector.step declares MILLI METRE

    def test_rotation_axis_override(self, test_step_path):
        from turbomodal.io import inspect_cad
        info = inspect_cad(test_step_path, num_sectors=24, rotation_axis=0)
        assert info.rotation_axis == 0

    def test_units_override(self, test_step_path):
        from turbomodal.io import inspect_cad
        info = inspect_cad(test_step_path, num_sectors=24, units="m")
        assert info.detected_unit == "m"

    def test_file_not_found(self):
        from turbomodal.io import inspect_cad
        with pytest.raises(FileNotFoundError):
            inspect_cad("/nonexistent/file.step", num_sectors=24)

    def test_unsupported_format(self, tmp_path):
        from turbomodal.io import inspect_cad
        dummy = tmp_path / "model.obj"
        dummy.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported CAD format"):
            inspect_cad(str(dummy), num_sectors=24)


class TestDetectStepUnits:
    def test_milli_metre(self, test_step_path):
        from pathlib import Path
        from turbomodal.io import _detect_step_units
        assert _detect_step_units(Path(test_step_path)) == "mm"

    def test_unknown_for_non_step(self, tmp_path):
        from turbomodal.io import _detect_step_units
        dummy = tmp_path / "model.brep"
        dummy.write_text("dummy")
        assert _detect_step_units(dummy) == "unknown"

    def test_metre(self, tmp_path):
        from turbomodal.io import _detect_step_units
        step = tmp_path / "model.step"
        step.write_text(
            "DATA;\n"
            "#10 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );\n"
            "ENDSEC;"
        )
        assert _detect_step_units(step) == "m"


class TestAlignAxisToZ:
    def test_z_is_noop(self):
        from turbomodal.io import _align_axis_to_z
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _align_axis_to_z(pts, 2)
        np.testing.assert_array_equal(result, pts)

    def test_x_to_z(self):
        from turbomodal.io import _align_axis_to_z
        pts = np.array([[1.0, 2.0, 3.0]])
        result = _align_axis_to_z(pts, 0)
        # X->Z means (x,y,z) -> (y,z,x)
        np.testing.assert_array_equal(result, [[2.0, 3.0, 1.0]])

    def test_y_to_z(self):
        from turbomodal.io import _align_axis_to_z
        pts = np.array([[1.0, 2.0, 3.0]])
        result = _align_axis_to_z(pts, 1)
        # Y->Z means (x,y,z) -> (z,x,y)
        np.testing.assert_array_equal(result, [[3.0, 1.0, 2.0]])


class TestExtractSurfaceTessellation:
    def test_returns_nodes_and_triangles(self, test_step_path):
        from turbomodal.io import _extract_surface_tessellation
        nodes, triangles, info = _extract_surface_tessellation(
            test_step_path, num_sectors=24
        )
        assert nodes.ndim == 2 and nodes.shape[1] == 3
        assert triangles.ndim == 2 and triangles.shape[1] == 3
        assert nodes.shape[0] > 0
        assert triangles.shape[0] > 0
        # All triangle indices should reference valid nodes
        assert triangles.max() < nodes.shape[0]
        assert triangles.min() >= 0


class TestLoadMeshValidation:
    def test_unsupported_format(self, tmp_path):
        from turbomodal.io import load_mesh
        dummy = tmp_path / "model.stl"
        dummy.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported mesh format"):
            load_mesh(str(dummy), num_sectors=24)

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


class TestDetectCadUnits:
    def test_step_milli_metre(self, test_step_path):
        from pathlib import Path
        from turbomodal.io import _detect_cad_units
        assert _detect_cad_units(Path(test_step_path)) == "mm"

    def test_unknown_for_brep(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        dummy = tmp_path / "model.brep"
        dummy.write_text("dummy")
        assert _detect_cad_units(dummy) == "unknown"

    def test_step_metre(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        step = tmp_path / "model.step"
        step.write_text(
            "DATA;\n"
            "#10 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );\n"
            "ENDSEC;"
        )
        assert _detect_cad_units(step) == "m"

    def test_iges_mm(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        # IGES Global Section: param 14 (Units Flag) = 2 (mm)
        # Params: 1H, (delim), 1H; (record delim), then 12 filler params, then 2
        filler = ",".join([""] * 11)
        global_line = f"1H,,1H;,{filler},2;".ljust(72) + "G      1"
        iges = tmp_path / "model.iges"
        iges.write_text(
            "test file".ljust(72) + "S      1\n"
            + global_line + "\n"
        )
        assert _detect_cad_units(iges) == "mm"

    def test_iges_inch(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        filler = ",".join([""] * 11)
        global_line = f"1H,,1H;,{filler},1;".ljust(72) + "G      1"
        iges = tmp_path / "model.igs"
        iges.write_text(
            "test file".ljust(72) + "S      1\n"
            + global_line + "\n"
        )
        assert _detect_cad_units(iges) == "inch"

    def test_iges_metre(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        filler = ",".join([""] * 11)
        global_line = f"1H,,1H;,{filler},6;".ljust(72) + "G      1"
        iges = tmp_path / "model.igs"
        iges.write_text(
            "test file".ljust(72) + "S      1\n"
            + global_line + "\n"
        )
        assert _detect_cad_units(iges) == "m"

    def test_iges_custom_unit(self, tmp_path):
        from turbomodal.io import _detect_cad_units
        # Units Flag=3 means check parameter 15 for the unit name
        filler = ",".join([""] * 11)
        global_line = f"1H,,1H;,{filler},3,4HINCH;".ljust(72) + "G      1"
        iges = tmp_path / "model.igs"
        iges.write_text(
            "test file".ljust(72) + "S      1\n"
            + global_line + "\n"
        )
        assert _detect_cad_units(iges) == "inch"


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


class TestLoadCadFullPipeline:
    def test_produces_tet10_mesh(self, test_step_path):
        from turbomodal.io import load_cad
        mesh = load_cad(test_step_path, num_sectors=24)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0
        assert len(mesh.left_boundary) > 0
        assert len(mesh.right_boundary) > 0

    def test_rotation_axis_override(self, test_step_path):
        from turbomodal.io import load_cad
        mesh = load_cad(test_step_path, num_sectors=24, rotation_axis=2)
        assert mesh.num_elements() > 0

    def test_open_surface_raises(self, tmp_path):
        """An open surface (not a closed shell) cannot become a solid."""
        import gmsh
        from turbomodal.io import load_cad
        brep = tmp_path / "open_surface.brep"
        gmsh.initialize()
        try:
            gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
            gmsh.model.occ.synchronize()
            gmsh.write(str(brep))
        finally:
            gmsh.finalize()
        with pytest.raises(RuntimeError, match="no solid volumes"):
            load_cad(str(brep), num_sectors=24)

    def test_heals_surface_only_model(self, tmp_path):
        """A closed shell (surfaces forming a watertight boundary) should
        be auto-healed into a solid volume and produce TET10 elements."""
        import gmsh
        brep = tmp_path / "surface_only_box.brep"
        gmsh.initialize()
        try:
            gmsh.model.occ.addBox(0, 0, 0, 0.1, 0.05, 0.025)
            gmsh.model.occ.synchronize()
            # Remove volume, keep surfaces
            vols = gmsh.model.getEntities(dim=3)
            gmsh.model.occ.remove(vols, recursive=False)
            gmsh.model.occ.synchronize()
            gmsh.write(str(brep))
        finally:
            gmsh.finalize()
        # Verify the healing pipeline directly: import surface-only BREP,
        # heal into a solid, mesh, and check for TET10 elements.
        # We can't use load_cad end-to-end because a box isn't a cyclic
        # sector and load_from_arrays requires cyclic boundary groups.
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.occ.importShapes(str(brep))
            gmsh.model.occ.synchronize()
            assert len(gmsh.model.getEntities(dim=3)) == 0, "Should start with no volumes"
            gmsh.model.occ.healShapes(sewFaces=True, makeSolids=True)
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(dim=3)
            assert len(volumes) > 0, "healShapes should create a solid volume"
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(2)
            # Check TET10 elements (type 11) exist
            elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=3)
            tet10_count = 0
            for etype, enodes in zip(elem_types, elem_nodes):
                if etype == 11:
                    tet10_count = len(enodes) // 10
            assert tet10_count > 0, "Should produce TET10 elements after healing"
        finally:
            gmsh.finalize()


    def test_offset_sector_boundaries_detected(self, tmp_path):
        """A sector NOT starting at theta=0 should still have its cyclic
        boundaries detected.  This tests the orientation-agnostic logic."""
        import gmsh
        from turbomodal.io import load_cad

        brep = tmp_path / "offset_sector.brep"
        num_sectors = 24
        sector_angle = 2 * np.pi / num_sectors  # 15 degrees
        offset_rad = np.radians(45.0)

        # Build an annular sector, then rotate it by 45 degrees
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            outer = gmsh.model.occ.addCylinder(
                0, 0, 0, 0, 0, 0.02, 0.05, angle=sector_angle)
            inner = gmsh.model.occ.addCylinder(
                0, 0, 0, 0, 0, 0.02, 0.03, angle=sector_angle)
            result = gmsh.model.occ.cut([(3, outer)], [(3, inner)])[0]
            # Rotate the entire sector by 45 degrees about Z
            gmsh.model.occ.rotate(result, 0, 0, 0, 0, 0, 1, offset_rad)
            gmsh.model.occ.synchronize()
            gmsh.write(str(brep))
        finally:
            gmsh.finalize()

        mesh = load_cad(str(brep), num_sectors=num_sectors,
                        auto_detect_boundaries=True)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0
        assert len(mesh.left_boundary) > 0, "Left boundary not detected for offset sector"
        assert len(mesh.right_boundary) > 0, "Right boundary not detected for offset sector"

    def test_healed_model_with_boundaries(self, tmp_path):
        """Surface-only annular sector should be healed AND have its cyclic
        boundaries detected when auto_detect_boundaries=True."""
        import gmsh
        from turbomodal.io import load_cad

        brep = tmp_path / "surface_sector.brep"
        num_sectors = 24
        sector_angle = 2 * np.pi / num_sectors

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            # Create a solid annular sector at theta=0
            outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 0.02,
                                                0.05, angle=sector_angle)
            inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 0.02,
                                                0.03, angle=sector_angle)
            result = gmsh.model.occ.cut([(3, outer)], [(3, inner)])[0]
            gmsh.model.occ.synchronize()

            # Remove volume, keep only surfaces
            vols = gmsh.model.getEntities(dim=3)
            gmsh.model.occ.remove(vols, recursive=False)
            gmsh.model.occ.synchronize()
            assert len(gmsh.model.getEntities(dim=3)) == 0
            gmsh.write(str(brep))
        finally:
            gmsh.finalize()

        # load_cad should heal (create volume) AND detect boundaries
        mesh = load_cad(str(brep), num_sectors=num_sectors,
                        auto_detect_boundaries=True)
        assert mesh.num_nodes() > 0
        assert mesh.num_elements() > 0
        assert len(mesh.left_boundary) > 0, "Left boundary not detected after healing"
        assert len(mesh.right_boundary) > 0, "Right boundary not detected after healing"


class TestRotationMatrix4x4:
    def test_z_axis(self):
        from turbomodal.io import _rotation_matrix_4x4
        angle = np.pi / 12  # 15 degrees
        m = _rotation_matrix_4x4(2, angle)
        assert len(m) == 16
        assert m[0] == pytest.approx(np.cos(angle))
        assert m[10] == pytest.approx(1.0)  # z-z = 1

    def test_x_axis(self):
        from turbomodal.io import _rotation_matrix_4x4
        m = _rotation_matrix_4x4(0, np.pi / 6)
        assert m[0] == pytest.approx(1.0)  # x-x = 1
        assert m[5] == pytest.approx(np.cos(np.pi / 6))  # y-y

    def test_y_axis(self):
        from turbomodal.io import _rotation_matrix_4x4
        m = _rotation_matrix_4x4(1, np.pi / 4)
        assert m[5] == pytest.approx(1.0)  # y-y = 1
        assert m[0] == pytest.approx(np.cos(np.pi / 4))  # x-x


class TestLoadMeshValidation:
    def test_unsupported_format(self, tmp_path):
        from turbomodal.io import load_mesh
        dummy = tmp_path / "model.stl"
        dummy.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported mesh format"):
            load_mesh(str(dummy), num_sectors=24)

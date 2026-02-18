"""Geometry and mesh import for turbomodal.

Supports CAD files (STEP, IGES, BREP) via gmsh OpenCASCADE and
pre-meshed files (NASTRAN, Abaqus, VTK, gmsh MSH) via meshio.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from turbomodal._core import Mesh, NodeSet


# File extensions handled by gmsh OpenCASCADE kernel (solid geometry only)
_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".brep"}

# File extensions handled by meshio
_MESHIO_EXTENSIONS = {
    ".bdf",
    ".nas",  # NASTRAN
    ".inp",  # Abaqus
    ".vtk",
    ".vtu",  # VTK
    ".cgns",  # CGNS
    ".med",  # Salome MED
    ".xdmf",  # XDMF
}

# All mesh extensions (meshio + native .msh)
_MESH_EXTENSIONS = _MESHIO_EXTENSIONS | {".msh"}


@dataclass
class CadInfo:
    """Metadata extracted from a CAD geometry file.

    Returned by :func:`inspect_cad` to let the user inspect geometry
    dimensions and the recommended mesh size before committing to a
    full volumetric mesh via :func:`load_cad`.

    All length values (radii, axial_length, etc.) are in **metres**.
    ``Geometry.OCCTargetUnit = "M"`` is set before CAD import so that
    gmsh normalises all coordinates to metres regardless of the source
    file's declared unit.  ``detected_unit`` records the source file's
    declared unit for display formatting only.
    """

    filepath: str
    num_sectors: int
    sector_angle_deg: float
    bounding_box: dict = field(default_factory=dict)
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    axial_length: float = 0.0
    radial_span: float = 0.0
    volume: float = 0.0
    surface_area: float = 0.0
    characteristic_length: float = 0.0
    recommended_mesh_size: float = 0.0
    num_surfaces: int = 0
    num_volumes: int = 0
    rotation_axis: int = 2  # 0=X, 1=Y, 2=Z
    detected_unit: str = "unknown"  # "mm", "m", "inch", etc.


def _validate_cad_path(filepath: str | Path) -> Path:
    """Validate a CAD file path and return a resolved Path."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CAD file not found: {filepath}")
    ext = filepath.suffix.lower()
    if ext == ".stl":
        raise ValueError(
            "STL files are surface meshes and cannot be volumetrically meshed. "
            "Convert to STEP or BREP in your CAD tool, or use load_mesh() "
            "with a pre-meshed volumetric file (.bdf, .inp, .msh, etc.)."
        )
    if ext not in _CAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported CAD format '{ext}'. Supported: {sorted(_CAD_EXTENSIONS)}"
        )
    return filepath


# Mapping from rotation axis to the two perpendicular coordinate indices.
# E.g. if the rotation axis is Z (2), radii are computed from X (0) and Y (1).
_PERP_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _detect_rotation_axis(num_sectors: int) -> int:
    """Detect which axis (0=X, 1=Y, 2=Z) is the rotation/symmetry axis.

    For each candidate axis, projects per-entity bounding-box *centres*
    onto the perpendicular plane and computes the angular span.  The axis
    whose angular span is closest to 360/num_sectors degrees is the
    rotation axis.

    Uses the centre of ``getBoundingBox(dim, tag)`` (one point per
    surface entity) rather than all 8 bounding-box corners, because
    combining min/max coordinates across different axes creates phantom
    angular positions that inflate the measured span.

    Must be called within an active gmsh session with a loaded model.
    """
    import gmsh

    expected_span = 2 * np.pi / num_sectors

    surfaces = gmsh.model.getEntities(dim=2)
    points = []
    for dim, tag in surfaces:
        try:
            sb = gmsh.model.getBoundingBox(dim, tag)
            # Use the centre of the entity bounding box
            points.append(
                [
                    (sb[0] + sb[3]) / 2,
                    (sb[1] + sb[4]) / 2,
                    (sb[2] + sb[5]) / 2,
                ]
            )
        except Exception:
            pass

    if not points:
        return 2  # fallback to Z

    pts = np.array(points)

    best_axis = 2
    best_score = float("inf")

    for axis, (c1, c2) in _PERP_AXES.items():
        angles = np.arctan2(pts[:, c2], pts[:, c1])
        angles = angles % (2 * np.pi)
        if len(angles) < 2:
            continue
        sorted_angles = np.sort(angles)
        gaps = np.diff(sorted_angles)
        wraparound_gap = (2 * np.pi - sorted_angles[-1]) + sorted_angles[0]
        max_gap = max(float(np.max(gaps)), float(wraparound_gap))
        angular_span = 2 * np.pi - max_gap

        score = abs(angular_span - expected_span)
        if score < best_score:
            best_score = score
            best_axis = axis

    return best_axis


def _detect_cad_units(filepath: Path) -> str:
    """Detect length units from a CAD file's metadata.

    Dispatches to format-specific parsers based on file extension.
    Returns a unit name: "mm", "m", "cm", "um", "inch", "ft", or "unknown".

    Supported formats with unit metadata:
    - STEP / STP : SI_UNIT or CONVERSION_BASED_UNIT in the data section
    - IGES / IGS : Units Flag (parameter 14) in the Global Section
    - BREP       : no unit metadata (always returns "unknown")
    """
    ext = filepath.suffix.lower()
    if ext in (".step", ".stp"):
        return _parse_step_units(filepath)
    if ext in (".iges", ".igs"):
        return _parse_iges_units(filepath)
    return "unknown"


def _parse_step_units(filepath: Path) -> str:
    """Parse a STEP file for SI_UNIT length declarations."""
    import re

    try:
        text = filepath.read_text(errors="ignore")
    except Exception:
        return "unknown"

    # Match: LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(<prefix>, .METRE.)
    si_pattern = re.compile(
        r"LENGTH_UNIT\(\)\s+NAMED_UNIT\(\*\)\s+SI_UNIT\(\s*"
        r"([.$\w]*)\s*,\s*\.METRE\.\s*\)",
        re.IGNORECASE,
    )
    match = si_pattern.search(text)
    if match:
        prefix_raw = match.group(1).strip().strip(".")
        prefix_map = {
            "$": "m",
            "": "m",
            "MILLI": "mm",
            "CENTI": "cm",
            "MICRO": "um",
        }
        return prefix_map.get(prefix_raw.upper(), "m")

    # Check for CONVERSION_BASED_UNIT (imperial)
    conv_pattern = re.compile(
        r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'\s*,",
        re.IGNORECASE,
    )
    conv_match = conv_pattern.search(text)
    if conv_match:
        unit_str = conv_match.group(1).upper()
        conv_units = {"INCH": "inch", "FOOT": "ft"}
        return conv_units.get(unit_str, "unknown")

    return "unknown"


def _parse_iges_units(filepath: Path) -> str:
    """Parse an IGES file for the Units Flag in the Global Section.

    The IGES Global Section (lines ending with 'G' in column 73) contains
    comma-separated parameters.  Parameter 14 is the Units Flag:

        1=inch, 2=mm, 3=see param 15, 4=ft, 5=mile,
        6=m, 7=km, 8=mil, 9=um, 10=cm, 11=uin
    """
    # Map IGES Units Flag integer to our unit string
    _IGES_UNIT_FLAG = {
        1: "inch",
        2: "mm",
        # 3 = custom (handled below via param 15)
        4: "ft",
        5: "mile",
        6: "m",
        7: "km",
        8: "mil",
        9: "um",
        10: "cm",
        11: "uin",
    }

    try:
        text = filepath.read_text(errors="ignore")
    except Exception:
        return "unknown"

    # Collect Global Section lines (column 73 == 'G')
    global_chars = []
    for line in text.splitlines():
        if len(line) >= 73 and line[72] == "G":
            # Columns 1-72 contain the data
            global_chars.append(line[:72])

    if not global_chars:
        return "unknown"

    global_text = "".join(global_chars)

    # The Global Section uses comma as the parameter delimiter and semicolon
    # as the record delimiter.  The first two parameters are Hollerith strings
    # defining these delimiters (e.g. "1H," and "1H;").
    #
    # Splitting by ";" is unsafe because "1H;" is a Hollerith string containing
    # the semicolon character.  Instead, split by commas and strip trailing
    # semicolons and whitespace from each parameter.
    params = [p.strip().rstrip(";").strip() for p in global_text.split(",")]

    # After splitting "1H,,1H;,...,<unit_flag>;":
    #   params[0] = "1H"  (first half of the Hollerith "1H,")
    #   params[1] = ""    (the literal comma from "1H," consumes a split slot)
    #   params[2] = "1H"  (record delimiter "1H;" with ; stripped)
    #   params[3..13] = IGES params 3-13
    #   params[14] = IGES param 14 = Units Flag
    # The extra split slot from "1H," means IGES param N is at index N for N >= 2.
    if len(params) < 15:
        return "unknown"

    try:
        unit_flag = int(params[14])
    except (ValueError, IndexError):
        return "unknown"

    if unit_flag == 3:
        # Custom unit — check IGES param 15 (split index 15) for a unit name
        # Hollerith string like "2HMM" or "4HINCH"; strip the count+H prefix.
        if len(params) > 15:
            import re

            cleaned = re.sub(r"^\d+H", "", params[15]).upper()
            name_map = {
                "MM": "mm",
                "MILLIMETER": "mm",
                "MILLIMETRE": "mm",
                "CM": "cm",
                "CENTIMETER": "cm",
                "CENTIMETRE": "cm",
                "M": "m",
                "METER": "m",
                "METRE": "m",
                "IN": "inch",
                "INCH": "inch",
                "FT": "ft",
                "FOOT": "ft",
                "UM": "um",
                "MICRON": "um",
                "MICROMETER": "um",
            }
            return name_map.get(cleaned, "unknown")
        return "unknown"

    return _IGES_UNIT_FLAG.get(unit_flag, "unknown")


def _align_axis_to_z(nodes: np.ndarray, rotation_axis: int) -> np.ndarray:
    """Rotate node coordinates so that *rotation_axis* maps to Z.

    Uses cyclic column permutations which are proper rotations
    (determinant +1, preserve handedness).
    """
    if rotation_axis == 2:
        return nodes
    if rotation_axis == 0:
        # X -> Z: (x,y,z) -> (y,z,x)
        return nodes[:, [1, 2, 0]]
    if rotation_axis == 1:
        # Y -> Z: (x,y,z) -> (z,x,y)
        return nodes[:, [2, 0, 1]]
    return nodes


def inspect_cad(
    filepath: str | Path,
    num_sectors: int,
    rotation_axis: int | None = None,
    units: str | None = None,
    verbosity: int = 0,
) -> CadInfo:
    """Inspect a CAD file and return geometry metadata without meshing.

    This is a lightweight operation that imports the CAD geometry,
    computes bounding box, dimensions, and a recommended mesh size,
    but does NOT generate any mesh.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep)
    num_sectors : number of sectors in the full annulus
    rotation_axis : override rotation axis detection (0=X, 1=Y, 2=Z, None=auto)
    units : override unit detection ("mm", "m", "cm", "inch", None=auto)
    verbosity : gmsh verbosity level (0=silent)

    Returns
    -------
    CadInfo
        Geometry metadata including recommended mesh size.
    """
    import gmsh

    filepath = _validate_cad_path(filepath)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        info = _build_cad_info(
            filepath, num_sectors, rotation_axis=rotation_axis, units=units
        )
    finally:
        gmsh.finalize()

    return info


def _extract_surface_tessellation(
    filepath: str | Path,
    num_sectors: int,
    surface_mesh_size: float | None = None,
    rotation_axis: int | None = None,
    units: str | None = None,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, CadInfo]:
    """Import CAD file and extract a lightweight surface triangulation.

    Generates a 2D surface mesh (triangles only) via gmsh, which is
    orders of magnitude faster than a full 3D tetrahedral mesh. Used
    by :func:`~turbomodal.viz.plot_cad` for preview rendering.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep)
    num_sectors : number of sectors in the full annulus
    surface_mesh_size : element size for the surface mesh (None = auto)
    rotation_axis : override rotation axis detection (0=X, 1=Y, 2=Z, None=auto)
    units : override unit detection ("mm", "m", "cm", "inch", None=auto)
    verbosity : gmsh verbosity level (0=silent)

    Returns
    -------
    nodes : (N, 3) float64 array of node coordinates (aligned so rotation axis = Z)
    triangles : (M, 3) int32 array of triangle connectivity (0-based)
    info : CadInfo with geometry metadata
    """
    import gmsh

    filepath = _validate_cad_path(filepath)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        info = _build_cad_info(
            filepath, num_sectors, rotation_axis=rotation_axis, units=units
        )

        # Set surface mesh size
        mesh_sz = surface_mesh_size or info.recommended_mesh_size * 2
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_sz)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_sz * 0.1)

        # Generate 2D surface mesh only (fast)
        gmsh.model.mesh.generate(2)

        # Extract surface nodes and triangles
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)

        tag_to_idx = {}
        for i, tag in enumerate(node_tags):
            tag_to_idx[int(tag)] = i

        # Get triangle elements (type 2 = 3-node triangle)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_nodes = None
        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # 3-node triangle
                n_elems = len(etags)
                raw = np.array(enodes, dtype=np.int64).reshape(n_elems, 3)
                triangles = np.zeros_like(raw, dtype=np.int32)
                for i in range(n_elems):
                    for j in range(3):
                        triangles[i, j] = tag_to_idx[int(raw[i, j])]
                tri_nodes = triangles
                break

        if tri_nodes is None:
            raise RuntimeError("No triangles found in surface mesh.")
    finally:
        gmsh.finalize()

    # Align nodes so the detected rotation axis maps to Z for sector replication
    aligned = _align_axis_to_z(coords.astype(np.float64), info.rotation_axis)
    return aligned, tri_nodes, info


def _build_cad_info(
    filepath: Path,
    num_sectors: int,
    rotation_axis: int | None = None,
    units: str | None = None,
) -> CadInfo:
    """Build CadInfo from an active gmsh model (must be called within gmsh session)."""
    import gmsh

    sector_angle_deg = 360.0 / num_sectors

    # Detect rotation axis
    if rotation_axis is None:
        rotation_axis = _detect_rotation_axis(num_sectors)

    # Detect units
    if units is not None:
        detected_unit = units
    else:
        detected_unit = _detect_cad_units(filepath)

    # Perpendicular coordinate indices for radius computation
    c1, c2 = _PERP_AXES[rotation_axis]

    # Bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    bbox = dict(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)
    bbox_vals = [xmin, ymin, zmin, xmax, ymax, zmax]

    # Count entities
    surfaces = gmsh.model.getEntities(dim=2)
    volumes = gmsh.model.getEntities(dim=3)

    # Volume and surface area via getMass
    total_volume = 0.0
    for _, tag in volumes:
        total_volume += gmsh.model.occ.getMass(3, tag)
    total_area = 0.0
    for _, tag in surfaces:
        total_area += gmsh.model.occ.getMass(2, tag)

    # Compute inner/outer radius from per-entity bounding boxes.
    # We use getBoundingBox(dim, tag) which returns the tight spatial bounds
    # of each trimmed entity.  This is much more reliable than parametric
    # sampling via getParametrizationBounds/getValue, because the parametric
    # domain of untrimmed surfaces can extend far beyond the actual geometry.
    radii = []
    for dim, tag in surfaces:
        try:
            sb = gmsh.model.getBoundingBox(dim, tag)
            # sb = (xmin, ymin, zmin, xmax, ymax, zmax)
            for v1 in (sb[c1], sb[c1 + 3]):
                for v2 in (sb[c2], sb[c2 + 3]):
                    radii.append(np.sqrt(v1**2 + v2**2))
        except Exception:
            pass

    if radii:
        inner_radius = float(min(radii))
        outer_radius = float(max(radii))
    else:
        # Fallback: estimate from overall bounding box corners
        corners_r = []
        for v1 in (bbox_vals[c1], bbox_vals[c1 + 3]):
            for v2 in (bbox_vals[c2], bbox_vals[c2 + 3]):
                corners_r.append(np.sqrt(v1**2 + v2**2))
        inner_radius = float(min(corners_r))
        outer_radius = float(max(corners_r))

    # Axial length is the extent along the rotation axis
    axial_length = float(bbox_vals[rotation_axis + 3] - bbox_vals[rotation_axis])
    radial_span = outer_radius - inner_radius

    # Recommended mesh size heuristic
    characteristic_length = min(radial_span, axial_length)
    if characteristic_length < 1e-10:
        characteristic_length = max(radial_span, axial_length)
    recommended_mesh_size = characteristic_length / 20.0

    return CadInfo(
        filepath=str(filepath),
        num_sectors=num_sectors,
        sector_angle_deg=sector_angle_deg,
        bounding_box=bbox,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        axial_length=axial_length,
        radial_span=radial_span,
        volume=total_volume,
        surface_area=total_area,
        characteristic_length=characteristic_length,
        recommended_mesh_size=recommended_mesh_size,
        num_surfaces=len(surfaces),
        num_volumes=len(volumes),
        rotation_axis=rotation_axis,
        detected_unit=detected_unit,
    )


def load_cad(
    filepath: str | Path,
    num_sectors: int,
    mesh_size: float | None = None,
    order: int = 2,
    left_boundary_name: str = "left_boundary",
    right_boundary_name: str = "right_boundary",
    hub_name: str | None = "hub_constraint",
    auto_detect_boundaries: bool = True,
    rotation_axis: int | None = None,
    units: str | None = None,
    optimize: bool = True,
    verbosity: int = 0,
) -> Mesh:
    """Load a CAD file and generate a TET10 mesh for cyclic symmetry analysis.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep)
    num_sectors : number of sectors in the full annulus
    mesh_size : characteristic mesh element size (None = auto from geometry)
    order : element order (2 = quadratic TET10)
    left_boundary_name : physical group name for the left cyclic boundary
    right_boundary_name : physical group name for the right cyclic boundary
    hub_name : physical group name for the hub constraint (None = no hub)
    auto_detect_boundaries : automatically identify cyclic boundary surfaces
    rotation_axis : override rotation axis (0=X, 1=Y, 2=Z, None=auto-detect)
    units : override unit detection ("mm", "m", "inch", None=auto-detect)
    optimize : run mesh optimization passes to improve element quality
    verbosity : gmsh verbosity level (0=silent, 1=errors, 5=debug)

    Returns
    -------
    Mesh object ready for cyclic symmetry analysis

    Notes
    -----
    STL files are not supported — they are surface meshes without solid
    geometry and cannot be volumetrically meshed. Convert to STEP or BREP
    first, or use ``load_mesh`` with a pre-meshed volumetric file.
    """
    import gmsh

    filepath = _validate_cad_path(filepath)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")

        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        # Ensure the model contains solid volumes.  If the CAD file only
        # has surfaces (a common export artefact), attempt to heal/sew
        # them into a closed solid automatically.
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            surfaces = gmsh.model.getEntities(dim=2)
            if not surfaces:
                raise RuntimeError(
                    "CAD file contains no geometry entities (0 surfaces, "
                    "0 volumes). The file may be empty or corrupt."
                )
            gmsh.model.occ.healShapes(
                sewFaces=True,
                makeSolids=True,
            )
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(dim=3)
            if not volumes:
                raise RuntimeError(
                    f"CAD file contains no solid volumes (found "
                    f"{len(surfaces)} surfaces) and automatic repair "
                    f"via healShapes failed — the surfaces likely do not "
                    f"form a closed shell. Repair the geometry in your "
                    f"CAD tool to create a watertight solid, or use "
                    f"load_mesh() with a pre-meshed volumetric file."
                )

        # Inspect geometry for mesh size and rotation axis
        info = _build_cad_info(
            filepath, num_sectors, rotation_axis=rotation_axis, units=units
        )

        # Set mesh size & auto-compute from geometry when not specified
        if mesh_size is None:
            mesh_size = info.recommended_mesh_size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)

        # Always create volume physical group (needed for gmsh to include
        # volume elements in the output).  This is independent of boundary
        # detection so that meshing succeeds even when no boundaries are found.
        volumes_for_pg = gmsh.model.getEntities(dim=3)
        if volumes_for_pg:
            vol_tags = [v[1] for v in volumes_for_pg]
            pg = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, pg, "volume")

        # --- Pass 1: Best-effort gmsh detection + periodic meshing ---
        # Use gmsh surface-normal heuristics to identify boundary surfaces.
        # Feed them to setPeriodic for compatible mesh topology, then mesh.
        left_tags: list[int] = []
        right_tags: list[int] = []
        left_cands: list = []
        right_cands: list = []
        if auto_detect_boundaries:
            left_tags, right_tags, left_cands, right_cands = (
                _auto_detect_cyclic_boundaries(
                    num_sectors,
                    left_boundary_name,
                    right_boundary_name,
                    hub_name,
                    rotation_axis=info.rotation_axis,
                )
            )

        if auto_detect_boundaries and (left_cands or right_cands):
            _validate_boundary_detection(
                left_cands, right_cands, num_sectors, info.rotation_axis
            )

        _apply_periodic_and_mesh(
            left_tags,
            right_tags,
            left_cands,
            right_cands,
            num_sectors,
            info.rotation_axis,
            order,
            optimize,
        )

        # Extract mesh and run authoritative geometric boundary detection.
        coords, connectivity, node_sets, tag_to_idx = _extract_mesh_arrays()
        n_free = _detect_boundaries_geometric(
            coords, connectivity, node_sets,
            num_sectors, info.rotation_axis,
        )

        # Check if boundary detection found any matched pairs at all.
        # Zero matched pairs indicates a fundamental problem (wrong
        # setPeriodic surface pairs); free boundary nodes are expected
        # for complex blade geometries and do NOT trigger remeshing.
        left_ns = next(
            (ns for ns in node_sets if ns.name == "left_boundary"), None,
        )
        n_matched = len(left_ns.node_ids) if left_ns else 0

        if n_matched == 0:
            import logging
            logging.getLogger("turbomodal.io").info(
                "Pass 1 found zero matched boundary pairs; remeshing with "
                "geometrically-detected surface pairs.",
            )

            detected_left, detected_right = _map_boundaries_to_gmsh_surfaces(
                coords, node_sets, tag_to_idx,
            )

            if detected_left and detected_right:
                # Build candidate dicts for the periodic matcher
                sector_angle = 2 * np.pi / num_sectors
                c1, c2 = _PERP_AXES[info.rotation_axis]
                det_left_cands = []
                det_right_cands = []
                for tag in detected_left:
                    try:
                        sb = gmsh.model.getBoundingBox(2, tag)
                        center = np.array([
                            (sb[0] + sb[3]) / 2,
                            (sb[1] + sb[4]) / 2,
                            (sb[2] + sb[5]) / 2,
                        ])
                        r = np.sqrt(center[c1] ** 2 + center[c2] ** 2)
                        det_left_cands.append({
                            "tag": tag, "center": center, "radius": r,
                        })
                    except Exception:
                        pass
                for tag in detected_right:
                    try:
                        sb = gmsh.model.getBoundingBox(2, tag)
                        center = np.array([
                            (sb[0] + sb[3]) / 2,
                            (sb[1] + sb[4]) / 2,
                            (sb[2] + sb[5]) / 2,
                        ])
                        r = np.sqrt(center[c1] ** 2 + center[c2] ** 2)
                        det_right_cands.append({
                            "tag": tag, "center": center, "radius": r,
                        })
                    except Exception:
                        pass

                # Clear mesh and remesh with correct periodic surfaces
                gmsh.model.mesh.clear()
                _apply_periodic_and_mesh(
                    detected_left,
                    detected_right,
                    det_left_cands,
                    det_right_cands,
                    num_sectors,
                    info.rotation_axis,
                    order,
                    optimize,
                )

                # Final extraction + geometric detection
                coords, connectivity, node_sets, tag_to_idx = (
                    _extract_mesh_arrays()
                )
                _detect_boundaries_geometric(
                    coords, connectivity, node_sets,
                    num_sectors, info.rotation_axis,
                )

        # Build C++ Mesh from corrected arrays
        mesh = Mesh()
        mesh.load_from_arrays(
            coords, connectivity, node_sets, num_sectors, info.rotation_axis
        )
    finally:
        gmsh.finalize()

    return mesh


def _count_boundary_nodes(
    left_name: str = "left_boundary",
    right_name: str = "right_boundary",
) -> tuple[int, int]:
    """Count mesh nodes on left/right boundary physical groups."""
    import gmsh

    n_left = n_right = 0
    for dim, tag in gmsh.model.getPhysicalGroups(dim=2):
        name = gmsh.model.getPhysicalName(dim, tag)
        if name not in (left_name, right_name):
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        nodes = set()
        for ent in entities:
            n_tags, _, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
            nodes.update(int(t) for t in n_tags)
        if name == left_name:
            n_left = len(nodes)
        else:
            n_right = len(nodes)
    return n_left, n_right


def _validate_boundary_detection(
    left_candidates: list,
    right_candidates: list,
    num_sectors: int,
    rotation_axis: int,
) -> None:
    """Log diagnostics about detected cyclic boundaries."""
    import logging

    logger = logging.getLogger("turbomodal.io")
    sector_angle = 2 * np.pi / num_sectors

    n_left = len(left_candidates)
    n_right = len(right_candidates)
    logger.info(
        "Boundary detection: %d left, %d right surfaces "
        "(sector_angle=%.1f deg, axis=%s)",
        n_left, n_right, np.degrees(sector_angle),
        ["X", "Y", "Z"][rotation_axis],
    )

    if n_left != n_right:
        logger.warning(
            "Asymmetric boundary detection: %d left vs %d right surfaces. "
            "Non-periodic surfaces may have been included, or genuine "
            "patches were missed.",
            n_left, n_right,
        )

    for label, cands in [("left", left_candidates), ("right", right_candidates)]:
        if len(cands) < 2:
            continue
        thetas = [s["theta"] for s in cands]
        spread = max(thetas) - min(thetas)
        spread = min(spread, 2 * np.pi - spread)
        if spread > sector_angle * 0.15:
            logger.warning(
                "%s boundary cluster has angular spread %.2f deg "
                "(> 15%% of sector angle %.2f deg). Some surfaces "
                "may not be true periodic boundaries.",
                label, np.degrees(spread), np.degrees(sector_angle),
            )

    for label, cands in [("left", left_candidates), ("right", right_candidates)]:
        for s in cands:
            logger.debug(
                "  %s surface tag=%d: theta=%.2f deg, r=%.4f, "
                "center=(%.4f, %.4f, %.4f)",
                label, s["tag"], np.degrees(s["theta"]), s["radius"],
                s["center"][0], s["center"][1], s["center"][2],
            )


def _set_periodic_centroid_matched(
    left_tags: list[int],
    right_tags: list[int],
    left_candidates: list,
    right_candidates: list,
    rotation_axis: int,
    sector_angle: float,
    max_pair_distance: float | None = None,
) -> int:
    """Apply setPeriodic with centroid-based surface pairing.

    Returns the number of successfully applied periodic constraints.
    """
    import gmsh
    import logging

    logger = logging.getLogger("turbomodal.io")

    rot = _rotation_matrix_4x4(rotation_axis, sector_angle)
    rot_np = np.array(rot).reshape(4, 4)

    left_centers = np.array([s["center"] for s in left_candidates])
    right_centers = np.array([s["center"] for s in right_candidates])

    # Rotate left centroids by sector_angle to find matching right surfaces
    left_h = np.hstack([left_centers, np.ones((len(left_centers), 1))])
    left_rotated = (rot_np @ left_h.T).T[:, :3]

    n_applied = 0
    n_skipped = 0
    used_right: set[int] = set()
    for i, lc_rot in enumerate(left_rotated):
        dists = np.linalg.norm(right_centers - lc_rot, axis=1)
        for u in used_right:
            dists[u] = np.inf
        best_j = int(np.argmin(dists))
        if dists[best_j] == np.inf:
            continue
        if max_pair_distance is not None and dists[best_j] > max_pair_distance:
            logger.warning(
                "Skipping surface pair: left tag %d -> right tag %d, "
                "distance %.4e exceeds threshold %.4e",
                left_tags[i], right_tags[best_j],
                dists[best_j], max_pair_distance,
            )
            n_skipped += 1
            continue
        used_right.add(best_j)
        try:
            gmsh.model.mesh.setPeriodic(
                2,
                [right_tags[best_j]],
                [left_tags[i]],
                rot,
            )
            n_applied += 1
        except Exception as exc:
            logger.debug(
                "setPeriodic failed for left surface %d -> right surface %d: %s",
                left_tags[i], right_tags[best_j], exc,
            )

    n_total = len(left_tags)
    n_failed = n_total - n_applied - n_skipped
    if n_failed > 0 or n_skipped > 0:
        logger.info(
            "setPeriodic: %d/%d surface pairs succeeded, %d failed, %d skipped (distance).",
            n_applied, n_total, n_failed, n_skipped,
        )
    return n_applied


def _map_boundaries_to_gmsh_surfaces(
    coords: np.ndarray,
    node_sets: list[NodeSet],
    tag_to_idx: dict[int, int],
) -> tuple[list[int], list[int]]:
    """Map detected left/right boundary nodes back to gmsh surface entities.

    After geometric boundary detection has identified which mesh nodes are
    left/right boundaries, this function finds which gmsh 2D surface entities
    those nodes belong to — needed for setPeriodic in a remesh pass.

    Parameters
    ----------
    coords : (N_nodes, 3) node coordinates
    node_sets : must contain ``left_boundary`` and ``right_boundary``
    tag_to_idx : gmsh node tag → 0-based local index mapping

    Returns
    -------
    (left_surface_tags, right_surface_tags) — gmsh 2D entity tags
    """
    import gmsh
    from scipy.spatial import cKDTree

    left_ns = right_ns = None
    for ns in node_sets:
        if ns.name == "left_boundary":
            left_ns = ns
        elif ns.name == "right_boundary":
            right_ns = ns
    if left_ns is None or right_ns is None:
        return [], []

    left_set = set(left_ns.node_ids)
    right_set = set(right_ns.node_ids)

    # Reverse mapping: local index → gmsh tag (only needed for node lookup)
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}

    # For each gmsh 2D surface entity, check what fraction of its mesh nodes
    # overlap with the detected left or right boundary nodes.
    surfaces = gmsh.model.getEntities(dim=2)
    left_tags: list[int] = []
    right_tags: list[int] = []

    for dim, tag in surfaces:
        n_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
        if len(n_tags) == 0:
            continue
        # Convert gmsh node tags to local indices
        local_ids = set()
        for nt in n_tags:
            idx = tag_to_idx.get(int(nt))
            if idx is not None:
                local_ids.add(idx)
        if not local_ids:
            continue

        n_on_left = len(local_ids & left_set)
        n_on_right = len(local_ids & right_set)
        frac_left = n_on_left / len(local_ids)
        frac_right = n_on_right / len(local_ids)

        # A surface is a boundary if a majority of its nodes are on that boundary
        if frac_left > 0.5:
            left_tags.append(tag)
        elif frac_right > 0.5:
            right_tags.append(tag)

    return left_tags, right_tags


def _apply_periodic_and_mesh(
    left_tags: list[int],
    right_tags: list[int],
    left_candidates: list,
    right_candidates: list,
    num_sectors: int,
    rotation_axis: int,
    order: int,
    optimize: bool = True,
) -> bool:
    """Apply periodic constraints, generate mesh, verify boundary match.

    Returns True if boundary nodes matched, False if post-mesh node
    snapping is needed.
    """
    import gmsh

    has_boundaries = bool(left_tags and right_tags)
    sector_angle = 2 * np.pi / num_sectors

    if has_boundaries:
        # Compute a distance threshold for centroid pairing from the
        # radial span of all detected boundary surfaces.
        all_radii = [s["radius"] for s in left_candidates + right_candidates]
        if len(all_radii) >= 2:
            max_pair_distance = (max(all_radii) - min(all_radii)) * 0.5
            # Ensure a reasonable minimum so tightly-spaced patches still pair
            max_pair_distance = max(max_pair_distance, 0.01 * max(all_radii))
        else:
            max_pair_distance = None

        _set_periodic_centroid_matched(
            left_tags,
            right_tags,
            left_candidates,
            right_candidates,
            rotation_axis,
            sector_angle,
            max_pair_distance,
        )

    gmsh.model.mesh.generate(3)

    if optimize:
        _optimize_mesh(order)
    else:
        gmsh.model.mesh.setOrder(order)

    if not has_boundaries:
        return True

    n_left, n_right = _count_boundary_nodes()
    if n_left == n_right and n_left > 0:
        return True  # periodic meshing worked

    import logging

    logging.getLogger("turbomodal.io").info(
        "setPeriodic produced mismatched boundary nodes "
        "(left=%d, right=%d); will apply node snapping.",
        n_left,
        n_right,
    )
    return False


def _optimize_mesh(order: int) -> None:
    """Run mesh optimization passes to improve element quality.

    For linear meshes: Netgen optimization + Laplace smoothing.
    For high-order meshes: additionally runs HighOrder and
    HighOrderElastic optimizers to fix mid-node placement and
    eliminate negative Jacobians.
    """
    import gmsh

    # Optimize the linear mesh first (better starting point for
    # high-order conversion).
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.optimize("Relocate3D")

    if order >= 2:
        gmsh.model.mesh.setOrder(order)
        # HighOrder optimizer repositions mid-edge nodes to improve
        # the scaled Jacobian of curved elements.
        gmsh.model.mesh.optimize("HighOrder")
        # HighOrderElastic uses an elasticity-based smoother — more
        # aggressive at eliminating negative Jacobians.
        gmsh.model.mesh.optimize("HighOrderElastic")


def _rotation_matrix_4x4(axis: int, angle: float) -> list[float]:
    """Build a flat row-major 4x4 affine rotation matrix for gmsh.

    Parameters
    ----------
    axis : 0=X, 1=Y, 2=Z
    angle : rotation angle in radians
    """
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 0:  # rotation about X
        return [
            1,
            0,
            0,
            0,
            0,
            c,
            -s,
            0,
            0,
            s,
            c,
            0,
            0,
            0,
            0,
            1,
        ]
    if axis == 1:  # rotation about Y
        return [
            c,
            0,
            s,
            0,
            0,
            1,
            0,
            0,
            -s,
            0,
            c,
            0,
            0,
            0,
            0,
            1,
        ]
    # axis == 2: rotation about Z
    return [
        c,
        -s,
        0,
        0,
        s,
        c,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    ]


def _auto_detect_cyclic_boundaries(
    num_sectors: int,
    left_name: str,
    right_name: str,
    hub_name: str | None,
    rotation_axis: int = 2,
) -> tuple[list[int], list[int], list, list]:
    """Auto-detect cyclic boundary surfaces and hub from geometry.

    Identifies surfaces whose outward normals are approximately tangential
    (perpendicular to the rotation axis) and splits them into two groups
    separated by approximately ``sector_angle = 2*pi/num_sectors`` radians.
    The lower-theta group becomes ``left_boundary``, the higher-theta group
    becomes ``right_boundary``.  The innermost radial surface is the hub.

    The detection is **orientation-agnostic**: the sector does not need to
    start at theta=0.

    Parameters
    ----------
    rotation_axis : 0=X, 1=Y, 2=Z — which axis the sector revolves around.
    """
    import gmsh

    sector_angle = 2 * np.pi / num_sectors
    c1, c2 = _PERP_AXES[rotation_axis]
    surfaces = gmsh.model.getEntities(dim=2)

    # Compute mean normal and mean position of each surface.
    # Position is derived from the per-entity bounding box (getBoundingBox)
    # which returns the tight spatial bounds of the *trimmed* entity.
    # This avoids the untrimmed-parametric-domain issue where
    # getParametrizationBounds/getValue can produce points far outside
    # the actual geometry.
    # Normals are still evaluated via getNormal at the parametric centre;
    # the *direction* is correct even if the parametric domain is untrimmed.
    surface_data = []
    for dim, tag in surfaces:
        try:
            sb = gmsh.model.getBoundingBox(dim, tag)
        except Exception:
            continue
        # Bounding-box centre as a reliable surface position estimate
        center = np.array(
            [
                (sb[0] + sb[3]) / 2,
                (sb[1] + sb[4]) / 2,
                (sb[2] + sb[5]) / 2,
            ]
        )
        # Normal at parametric centre (direction only — still valid)
        try:
            bounds_min, bounds_max = gmsh.model.getParametrizationBounds(dim, tag)
            u_mid = (bounds_min[0] + bounds_max[0]) / 2
            v_mid = (bounds_min[1] + bounds_max[1]) / 2
            normal = gmsh.model.getNormal(tag, [u_mid, v_mid])
        except Exception:
            continue  # skip — cannot determine surface orientation
        # theta and radius in the perpendicular plane
        theta = np.arctan2(center[c2], center[c1])
        if theta < -0.01:
            theta += 2 * np.pi
        r = np.sqrt(center[c1] ** 2 + center[c2] ** 2)
        surface_data.append(
            {
                "tag": tag,
                "normal": np.array(normal[:3]),
                "center": center,
                "theta": theta,
                "radius": r,
            }
        )

    if not surface_data:
        return [], [], [], []

    # --- Hub detection (unchanged — uses radius/radial-normal checks) ---
    hub_candidates = []
    median_radius = np.median([s["radius"] for s in surface_data])
    for sd in surface_data:
        axial_component = abs(sd["normal"][rotation_axis])
        if axial_component < 0.3 and sd["radius"] < median_radius:
            radial_dir = np.zeros(3)
            radial_dir[c1] = sd["center"][c1]
            radial_dir[c2] = sd["center"][c2]
            if np.linalg.norm(radial_dir) > 1e-10:
                radial_dir /= np.linalg.norm(radial_dir)
                dot = np.dot(sd["normal"], radial_dir)
                if abs(dot) > 0.7:
                    hub_candidates.append(sd)

    # --- Cyclic boundary detection (orientation-agnostic) ---
    # Cyclic boundary surfaces have normals that are approximately
    # *tangential* — perpendicular to both the rotation axis and the
    # radial direction.  This distinguishes them from:
    #   - hub/shroud surfaces (radial normals)
    #   - inlet/outlet surfaces (axial normals)
    #   - cylindrical walls (also radial normals)
    boundary_candidates = []
    for sd in surface_data:
        axial_component = abs(sd["normal"][rotation_axis])
        if axial_component > 0.5:
            continue  # axial surface (inlet/outlet)
        # Compute tangential direction at this surface's angular position
        r = sd["radius"]
        if r < 1e-10:
            continue  # on-axis surface, can't compute tangential
        tangential = np.zeros(3)
        tangential[c1] = -sd["center"][c2] / r
        tangential[c2] = sd["center"][c1] / r
        tangential_component = abs(np.dot(sd["normal"], tangential))
        if tangential_component <= 0.5:
            continue

        # --- Wedge-proximity check ---
        # True sector boundaries lie along a constant-theta wedge plane
        # (flat or curved).  Blade/shroud surfaces may have tangential
        # normals at their centroid but span a significant angular range.
        # Sample points along the surface's boundary curves (which are
        # trimmed, unlike the parametric domain) and reject if they
        # spread too much in theta.
        try:
            boundary_edges = gmsh.model.getBoundary(
                [(2, sd["tag"])], oriented=False, combined=False,
            )
            thetas_edge: list[float] = []
            for dim_b, tag_b in boundary_edges:
                if dim_b != 1:
                    continue
                try:
                    b_bounds = gmsh.model.getParametrizationBounds(
                        1, abs(tag_b),
                    )
                    for t_param in np.linspace(
                        b_bounds[0][0], b_bounds[1][0], 5,
                    ):
                        pt = gmsh.model.getValue(1, abs(tag_b), [t_param])
                        thetas_edge.append(np.arctan2(pt[c2], pt[c1]))
                except Exception:
                    continue  # skip this edge, use others
            if len(thetas_edge) >= 4:
                ref = sd["theta"]
                deltas = np.array(thetas_edge) - ref
                # Wrap to [-pi, pi]
                deltas = (deltas + np.pi) % (2 * np.pi) - np.pi
                # Robust spread: 10th-to-90th percentile range
                spread = float(
                    np.percentile(deltas, 90) - np.percentile(deltas, 10)
                )
                if spread > sector_angle * 0.10:
                    continue  # surface spans too much theta
        except Exception:
            pass  # conservative: accept if edge sampling fails

        boundary_candidates.append(sd)

    left_candidates = []
    right_candidates = []

    if len(boundary_candidates) >= 2:
        # Find the pair of surfaces whose angular separation is closest
        # to sector_angle.  These anchor the two boundary clusters.
        # Then assign remaining candidates to the nearest cluster.
        n = len(boundary_candidates)
        best_pair = None
        best_pair_score = float("inf")

        for i in range(n):
            for j in range(i + 1, n):
                ti = boundary_candidates[i]["theta"]
                tj = boundary_candidates[j]["theta"]
                sep = tj - ti
                if sep < 0:
                    sep += 2 * np.pi
                score = abs(sep - sector_angle)
                # Also check reverse direction
                rev_sep = 2 * np.pi - sep
                rev_score = abs(rev_sep - sector_angle)
                if rev_score < score:
                    # j is actually the "left" (lower) boundary
                    if rev_score < best_pair_score:
                        best_pair_score = rev_score
                        best_pair = (j, i)
                else:
                    if score < best_pair_score:
                        best_pair_score = score
                        best_pair = (i, j)

        # Accept if the best pair separation is within 10% of sector_angle
        if best_pair is not None and best_pair_score < sector_angle * 0.10:
            left_theta = boundary_candidates[best_pair[0]]["theta"]
            right_theta = boundary_candidates[best_pair[1]]["theta"]
            cluster_tol = sector_angle * 0.15

            for sd in boundary_candidates:
                # Angular distance to each cluster centre (handles wraparound)
                dt_left = abs(sd["theta"] - left_theta)
                dt_left = min(dt_left, 2 * np.pi - dt_left)
                dt_right = abs(sd["theta"] - right_theta)
                dt_right = min(dt_right, 2 * np.pi - dt_right)
                if dt_left < cluster_tol:
                    left_candidates.append(sd)
                elif dt_right < cluster_tol:
                    right_candidates.append(sd)

    # --- Warn if boundaries not found ---
    if not left_candidates or not right_candidates:
        import warnings

        n_surf = len(surface_data)
        n_cand = len(boundary_candidates) if "boundary_candidates" in dir() else 0
        theta_range = ""
        if surface_data:
            all_thetas = [sd["theta"] for sd in surface_data]
            theta_range = (
                f" Theta range: [{min(all_thetas):.2f}, {max(all_thetas):.2f}] rad."
            )
        axial_info = ""
        if surface_data:
            axials = [abs(sd["normal"][rotation_axis]) for sd in surface_data]
            axial_info = f" Axial components: [{min(axials):.2f}, {max(axials):.2f}]."
        warnings.warn(
            f"Cyclic boundary auto-detection found {len(left_candidates)} left "
            f"and {len(right_candidates)} right candidate surfaces out of "
            f"{n_surf} total ({n_cand} with small axial normal component)."
            f"{theta_range}{axial_info} "
            f"The sector boundaries may not have been correctly identified. "
            f"Consider providing boundary surfaces manually via physical groups."
        )

    # --- Create physical groups ---
    left_tags: list[int] = []
    right_tags: list[int] = []
    if left_candidates:
        left_tags = [s["tag"] for s in left_candidates]
        pg = gmsh.model.addPhysicalGroup(2, left_tags)
        gmsh.model.setPhysicalName(2, pg, left_name)

    if right_candidates:
        right_tags = [s["tag"] for s in right_candidates]
        pg = gmsh.model.addPhysicalGroup(2, right_tags)
        gmsh.model.setPhysicalName(2, pg, right_name)

    if hub_name and hub_candidates:
        hub_tags = [s["tag"] for s in hub_candidates]
        pg = gmsh.model.addPhysicalGroup(2, hub_tags)
        gmsh.model.setPhysicalName(2, pg, hub_name)

    return left_tags, right_tags, left_candidates, right_candidates


def _extract_mesh_arrays(
) -> tuple[np.ndarray, np.ndarray, list[NodeSet], dict[int, int]]:
    """Extract mesh arrays from active gmsh model.

    Returns (coords, connectivity, node_sets, tag_to_idx) for post-processing
    before passing to the C++ Mesh.  tag_to_idx maps gmsh node tags to
    0-based local indices.
    """
    import gmsh

    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(node_coords).reshape(-1, 3)

    # Build gmsh tag -> 0-based index mapping
    tag_to_idx = {}
    for i, tag in enumerate(node_tags):
        tag_to_idx[int(tag)] = i

    # Get TET10 elements (type 11 in gmsh)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

    tet10_connectivity = None
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 11:  # TET10
            n_elems = len(etags)
            raw = np.array(enodes, dtype=np.int64).reshape(n_elems, 10)
            connectivity = np.zeros_like(raw, dtype=np.int32)
            for i in range(n_elems):
                for j in range(10):
                    connectivity[i, j] = tag_to_idx[int(raw[i, j])]
            # gmsh Python API returns TET10 nodes in a different order than
            # the .msh file format: nodes 8 and 9 are swapped.  The C++ solver
            # expects the .msh convention, so swap them back.
            connectivity[:, [8, 9]] = connectivity[:, [9, 8]]
            tet10_connectivity = connectivity
            break

    if tet10_connectivity is None:
        volumes = gmsh.model.getEntities(dim=3)
        found_types = {}
        for d in range(4):
            et, etg, _ = gmsh.model.mesh.getElements(dim=d)
            for etype, tags in zip(et, etg):
                found_types[int(etype)] = (d, len(tags))

        _GMSH_TYPE_NAMES = {
            1: "Line2",
            2: "Tri3",
            3: "Quad4",
            4: "Tet4",
            5: "Hex8",
            6: "Prism6",
            8: "Line3",
            9: "Tri6",
            11: "Tet10",
            15: "Point",
        }
        type_summary = ", ".join(
            f"{_GMSH_TYPE_NAMES.get(t, f'type{t}')}(dim{d})×{n}"
            for t, (d, n) in sorted(found_types.items())
        )

        hints = []
        if not volumes:
            hints.append(
                "The model has no solid volumes — only surfaces/shells. "
                "Repair the CAD file to create a closed solid body."
            )
        if 4 in found_types and 11 not in found_types:
            hints.append(
                "Linear tetrahedra (Tet4) were generated but quadratic "
                "conversion failed. Try calling gmsh.model.mesh.setOrder(2) "
                "explicitly, or check for geometry issues that prevent "
                "high-order meshing."
            )
        if not found_types:
            hints.append(
                "No mesh elements were generated at all. The 3D mesher "
                "may have failed silently — try increasing verbosity or "
                "providing an explicit mesh_size."
            )

        msg = (
            f"No TET10 (quadratic tetrahedra) elements found in mesh. "
            f"Volumes in model: {len(volumes)}. "
            f"Elements found: [{type_summary or 'none'}]."
        )
        if hints:
            msg += " " + " ".join(hints)

        raise RuntimeError(msg)

    # Extract physical group node sets
    node_sets: list[NodeSet] = []
    phys_groups = gmsh.model.getPhysicalGroups()
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if not name:
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        group_nodes: set[int] = set()
        for ent in entities:
            n_tags, _, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
            for nt in n_tags:
                if int(nt) in tag_to_idx:
                    group_nodes.add(tag_to_idx[int(nt)])
        ns = NodeSet()
        ns.name = name
        ns.node_ids = sorted(group_nodes)
        node_sets.append(ns)

    return coords.astype(np.float64), tet10_connectivity.astype(np.int32), node_sets, tag_to_idx


def _extract_surface_node_ids(connectivity: np.ndarray) -> np.ndarray:
    """Extract node IDs that lie on the surface (boundary faces) of TET10 mesh.

    A face is on the surface if it belongs to exactly one element.

    Parameters
    ----------
    connectivity : (N_elements, 10) TET10 element connectivity (0-based)

    Returns
    -------
    Sorted array of unique surface node IDs.
    """
    # TET10 face definitions: (corner_indices, all_6_node_indices)
    # Gmsh convention: corners 0-3, mid-edge 4(0-1) 5(1-2) 6(0-2) 7(0-3) 8(1-3) 9(2-3)
    _FACES = [
        ([0, 1, 2], [0, 1, 2, 4, 5, 6]),  # opposite node 3
        ([0, 1, 3], [0, 1, 3, 4, 8, 7]),  # opposite node 2
        ([1, 2, 3], [1, 2, 3, 5, 9, 8]),  # opposite node 0
        ([0, 2, 3], [0, 2, 3, 6, 9, 7]),  # opposite node 1
    ]

    # Count how many elements share each face (key = sorted corner node tuple)
    face_count: dict[tuple[int, ...], list[int]] = {}
    for elem_idx in range(connectivity.shape[0]):
        elem = connectivity[elem_idx]
        for corner_idx, all_idx in _FACES:
            key = tuple(sorted(int(elem[c]) for c in corner_idx))
            if key not in face_count:
                face_count[key] = [int(elem[n]) for n in all_idx]
            else:
                # Mark as shared (interior) by setting to empty
                face_count[key] = []

    # Collect unique node IDs from faces appearing exactly once
    surface_nodes: set[int] = set()
    for nodes_list in face_count.values():
        if nodes_list:  # non-empty = surface face
            surface_nodes.update(nodes_list)

    return np.array(sorted(surface_nodes), dtype=np.int64)


def _detect_boundaries_geometric(
    coords: np.ndarray,
    connectivity: np.ndarray,
    node_sets: list[NodeSet],
    num_sectors: int,
    rotation_axis: int,
    tolerance: float | None = None,
) -> int:
    """Detect cyclic boundaries by rotating the surface shell and finding contact.

    Extracts surface nodes, rotates them by the sector angle, and finds
    coincident pairs.  Sets ``left_boundary`` and ``right_boundary`` node
    sets on *node_sets* in place, and snaps right boundary node coordinates
    in *coords*.

    Parameters
    ----------
    coords : (N_nodes, 3) mutable node coordinates
    connectivity : (N_elements, 10) TET10 element connectivity
    node_sets : list of NodeSet — modified in place
    num_sectors : number of sectors in the full annulus
    rotation_axis : 0=X, 1=Y, 2=Z
    tolerance : matching distance threshold (None = auto from mesh)

    Returns
    -------
    Number of orphaned surface nodes on boundary faces that have no partner.
    Zero means perfect matching.
    """
    from scipy.spatial import cKDTree

    surface_ids = _extract_surface_node_ids(connectivity)
    if len(surface_ids) == 0:
        raise RuntimeError("No surface nodes found — mesh may be empty.")

    # Auto-compute tolerance from average edge length if not provided
    if tolerance is None:
        # Sample edge lengths from first ~100 elements
        n_sample = min(100, connectivity.shape[0])
        edge_lengths = []
        for i in range(n_sample):
            elem = connectivity[i]
            # Check edges between corner nodes (0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
            for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                d = np.linalg.norm(coords[elem[a]] - coords[elem[b]])
                edge_lengths.append(d)
        avg_edge = np.mean(edge_lengths)
        tolerance = avg_edge * 0.3

    # Rotation setup
    alpha = 2 * np.pi / num_sectors
    c, s = np.cos(alpha), np.sin(alpha)
    _PERP = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    c1, c2 = _PERP[rotation_axis]

    # Rotate surface nodes by +alpha (simulates the next sector CCW)
    surface_coords = coords[surface_ids].copy()
    rotated = surface_coords.copy()
    rotated[:, c1] = c * surface_coords[:, c1] - s * surface_coords[:, c2]
    rotated[:, c2] = s * surface_coords[:, c1] + c * surface_coords[:, c2]

    # Build KD-tree on original surface node coordinates
    tree = cKDTree(surface_coords)

    # For each rotated node, find the nearest original surface node
    dists, idxs = tree.query(rotated)

    # Matched pairs: rotated node i matches original node idxs[i]
    # - The original node at position idxs[i] is a RIGHT boundary node
    #   (it coincides with the rotated position of node i from the next sector)
    # - Node i (pre-rotation) is a LEFT boundary node
    #   (its rotated position touches the original sector)
    matched_left = []
    matched_right = []
    used_right: set[int] = set()

    # Cross-patch validation: check radial and axial consistency
    surface_r = np.sqrt(surface_coords[:, c1] ** 2 + surface_coords[:, c2] ** 2)
    surface_ax = surface_coords[:, rotation_axis]
    rotated_r = np.sqrt(rotated[:, c1] ** 2 + rotated[:, c2] ** 2)
    rotated_ax = rotated[:, rotation_axis]
    radial_tol = tolerance
    axial_tol = tolerance

    for i in range(len(surface_ids)):
        if dists[i] > tolerance:
            continue
        j = idxs[i]
        right_global = int(surface_ids[j])
        if right_global in used_right:
            continue
        # Cross-patch validation
        dr = abs(rotated_r[i] - surface_r[j])
        dax = abs(rotated_ax[i] - surface_ax[j])
        if dr > radial_tol or dax > axial_tol:
            continue
        left_global = int(surface_ids[i])
        used_right.add(right_global)
        matched_left.append(left_global)
        matched_right.append(right_global)

    if not matched_left:
        raise RuntimeError(
            "Geometric boundary detection failed: no surface node pairs "
            "found within tolerance after rotating by the sector angle. "
            "Check that num_sectors is correct and the mesh represents "
            "a single sector."
        )

    # Snap right boundary nodes to exact rotated left positions
    for i in range(len(matched_left)):
        left_id = matched_left[i]
        right_id = matched_right[i]
        lc = coords[left_id]
        coords[right_id, c1] = c * lc[c1] - s * lc[c2]
        coords[right_id, c2] = s * lc[c1] + c * lc[c2]
        coords[right_id, rotation_axis] = lc[rotation_axis]

    # Update or create node sets
    left_ns = right_ns = None
    for ns in node_sets:
        if ns.name == "left_boundary":
            left_ns = ns
        elif ns.name == "right_boundary":
            right_ns = ns

    if left_ns is None:
        left_ns = NodeSet()
        left_ns.name = "left_boundary"
        node_sets.append(left_ns)
    if right_ns is None:
        right_ns = NodeSet()
        right_ns.name = "right_boundary"
        node_sets.append(right_ns)

    left_ns.node_ids = sorted(matched_left)
    right_ns.node_ids = sorted(matched_right)

    # Classify unmatched sector-face nodes as free boundary.
    # These are nodes near the matched boundary but without a rotational
    # partner — e.g. inset blade edges that don't connect to adjacent sectors.
    free_nodes = []
    matched_set = set(matched_left) | set(matched_right)
    if matched_left:
        matched_left_coords = coords[np.array(matched_left)]
        matched_right_coords = coords[np.array(matched_right)]
        boundary_tree = cKDTree(
            np.vstack([matched_left_coords, matched_right_coords])
        )
        proximity_tol = tolerance * 3.0  # slightly wider to catch near-misses
        for sid in surface_ids:
            if int(sid) in matched_set:
                continue
            dist, _ = boundary_tree.query(coords[sid])
            if dist < proximity_tol:
                free_nodes.append(int(sid))

    # Create or update free_boundary node set
    free_ns = None
    for ns in node_sets:
        if ns.name == "free_boundary":
            free_ns = ns
            break
    if free_ns is None and free_nodes:
        free_ns = NodeSet()
        free_ns.name = "free_boundary"
        node_sets.append(free_ns)
    if free_ns is not None:
        free_ns.node_ids = sorted(free_nodes)

    import logging
    logger = logging.getLogger("turbomodal.io")
    logger.info(
        "Geometric boundary detection: matched %d node pairs, "
        "%d free boundary nodes (tolerance=%.2e, %d surface nodes total).",
        len(matched_left), len(free_nodes), tolerance, len(surface_ids),
    )

    return len(free_nodes)


def _snap_boundary_nodes(
    coords: np.ndarray,
    node_sets: list[NodeSet],
    num_sectors: int,
    rotation_axis: int,
    mesh_size: float,
) -> None:
    """Snap right boundary nodes to match rotated left boundary positions.

    Modifies *coords* and the ``left_boundary`` / ``right_boundary``
    NodeSets **in place** so that every left boundary node has an exact
    partner on the right boundary at the rotated position.

    This is the fallback when gmsh's ``setPeriodic`` fails to produce
    matching meshes on both sector boundaries.
    """
    from scipy.spatial import cKDTree

    # Find the left/right node sets
    left_ns = right_ns = None
    for ns in node_sets:
        if ns.name == "left_boundary":
            left_ns = ns
        elif ns.name == "right_boundary":
            right_ns = ns
    if left_ns is None or right_ns is None:
        return

    left_ids = np.array(left_ns.node_ids, dtype=int)
    right_ids = np.array(right_ns.node_ids, dtype=int)
    if len(left_ids) == 0 or len(right_ids) == 0:
        return

    # Build 3D rotation matrix for sector_angle around rotation_axis
    alpha = 2 * np.pi / num_sectors
    c, s = np.cos(alpha), np.sin(alpha)
    _PERP = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    c1, c2 = _PERP[rotation_axis]

    # Rotate left boundary nodes by +sector_angle → target right positions
    left_coords = coords[left_ids].copy()
    target = left_coords.copy()
    target[:, c1] = c * left_coords[:, c1] - s * left_coords[:, c2]
    target[:, c2] = s * left_coords[:, c1] + c * left_coords[:, c2]

    # KD-tree on right boundary nodes
    right_coords = coords[right_ids]
    tree = cKDTree(right_coords)

    tol = mesh_size * 0.5
    dists, idxs = tree.query(target)

    # Pre-compute cylindrical coordinates for cross-patch validation.
    # Target and right nodes on the same physical patch should have
    # similar radial and axial positions; cross-patch matches (e.g.
    # hub ↔ shroud) will diverge in radius or axial position.
    target_r = np.sqrt(target[:, c1] ** 2 + target[:, c2] ** 2)
    target_ax = target[:, rotation_axis]
    right_r = np.sqrt(right_coords[:, c1] ** 2 + right_coords[:, c2] ** 2)
    right_ax = right_coords[:, rotation_axis]
    radial_tol = mesh_size * 0.3
    axial_tol = mesh_size * 0.3

    # Build matched pairs: left_id → right_id (with snapping)
    matched_left = []
    matched_right = []
    used_right: set[int] = set()
    n_rejected_cross_patch = 0
    for i in range(len(left_ids)):
        if dists[i] > tol:
            continue  # gap region or no nearby match
        right_local_idx = idxs[i]
        right_global_idx = right_ids[right_local_idx]
        if right_global_idx in used_right:
            continue  # already matched to another left node
        # Cross-patch validation: reject if radial or axial mismatch
        dr = abs(target_r[i] - right_r[right_local_idx])
        dax = abs(target_ax[i] - right_ax[right_local_idx])
        if dr > radial_tol or dax > axial_tol:
            n_rejected_cross_patch += 1
            continue
        used_right.add(right_global_idx)
        # Snap the right node to the exact target position
        coords[right_global_idx] = target[i]
        matched_left.append(int(left_ids[i]))
        matched_right.append(int(right_global_idx))

    if not matched_left:
        raise RuntimeError(
            "Node snapping failed: no left boundary node could be matched "
            "to a right boundary node within tolerance. The mesh may not "
            "have periodic boundaries."
        )

    import logging
    logger = logging.getLogger("turbomodal.io")
    n_dropped = len(left_ids) + len(right_ids) - 2 * len(matched_left)
    if n_dropped > 0 or n_rejected_cross_patch > 0:
        logger.info(
            "Boundary node snapping: matched %d pairs, dropped %d nodes, "
            "rejected %d cross-patch matches.",
            len(matched_left), n_dropped, n_rejected_cross_patch,
        )

    # Update node sets in place
    left_ns.node_ids = sorted(matched_left)
    right_ns.node_ids = sorted(matched_right)


def _meshio_to_gmsh_tet10(connectivity: np.ndarray) -> np.ndarray:
    """Reorder TET10 nodes from meshio convention to gmsh convention.

    meshio swaps mid-edge nodes 8 and 9 relative to gmsh:
      gmsh:   [0,1,2,3, 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)]
      meshio: [0,1,2,3, 4(0-1), 5(1-2), 6(0-2), 7(0-3), 9(2-3), 8(1-3)]

    The C++ solver expects gmsh ordering.
    """
    reordered = connectivity.copy()
    reordered[:, 8] = connectivity[:, 9]
    reordered[:, 9] = connectivity[:, 8]
    return reordered


def load_mesh(
    filepath: str | Path,
    num_sectors: int,
    file_format: str | None = None,
) -> Mesh:
    """Load a pre-meshed file into a turbomodal Mesh.

    Parameters
    ----------
    filepath : path to mesh file
    num_sectors : number of sectors in the full annulus
    file_format : override automatic format detection (e.g. 'nastran', 'abaqus')

    Supported formats:
    - .msh (gmsh MSH 2.x — uses native C++ loader)
    - .bdf/.nas (NASTRAN)
    - .inp (Abaqus)
    - .vtk/.vtu (VTK)
    - .cgns (CGNS)
    - .med (Salome MED)
    - .xdmf (XDMF)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    ext = filepath.suffix.lower()

    # For .msh files, use the native C++ gmsh loader
    if ext == ".msh" and file_format is None:
        mesh = Mesh()
        mesh.num_sectors = num_sectors
        mesh.load_from_gmsh(str(filepath))
        mesh.identify_cyclic_boundaries()
        mesh.match_boundary_nodes()
        return mesh

    # Validate extension when no explicit format override
    if file_format is None and ext not in _MESHIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported mesh format '{ext}'. Supported: "
            f"{sorted(_MESH_EXTENSIONS)}. "
            f"For CAD files (.step, .iges, .brep) use load_cad() instead."
        )

    # For all other formats, use meshio
    import meshio

    mio = meshio.read(str(filepath), file_format=file_format)

    # Find TET10 cells
    tet10_blocks = []
    for block in mio.cells:
        if block.type == "tetra10":
            tet10_blocks.append(block.data)

    if not tet10_blocks:
        # Fall back to linear tets if no quadratic
        for block in mio.cells:
            if block.type == "tetra":
                raise RuntimeError(
                    "Mesh contains only linear tetrahedra (tetra). "
                    "Turbomodal requires quadratic tetrahedra (tetra10/TET10). "
                    "Re-mesh with second-order elements."
                )
        raise RuntimeError(
            f"No tetrahedra found in mesh file. Cell types: "
            f"{[b.type for b in mio.cells]}"
        )

    connectivity = np.vstack(tet10_blocks).astype(np.int32)

    # meshio uses a different TET10 node ordering than gmsh — swap nodes 8 & 9
    connectivity = _meshio_to_gmsh_tet10(connectivity)

    coords = mio.points.astype(np.float64)

    # Extract node sets from meshio point_sets
    node_sets = []
    if hasattr(mio, "point_sets") and mio.point_sets:
        for name, ids in mio.point_sets.items():
            ns = NodeSet()
            ns.name = name
            ns.node_ids = sorted(int(i) for i in ids)
            node_sets.append(ns)

    # Also check point_data for node set markers
    if hasattr(mio, "point_data"):
        for key, data in mio.point_data.items():
            if key.startswith("nset_") or key in (
                "left_boundary",
                "right_boundary",
                "hub",
            ):
                ns = NodeSet()
                ns.name = key.replace("nset_", "")
                ns.node_ids = sorted(int(i) for i in np.where(data > 0)[0])
                if ns.node_ids:
                    node_sets.append(ns)

    mesh = Mesh()
    mesh.load_from_arrays(coords, connectivity, node_sets, num_sectors)
    return mesh

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
    ".bdf", ".nas",   # NASTRAN
    ".inp",           # Abaqus
    ".vtk", ".vtu",   # VTK
    ".cgns",          # CGNS
    ".med",           # Salome MED
    ".xdmf",          # XDMF
}

# All mesh extensions (meshio + native .msh)
_MESH_EXTENSIONS = _MESHIO_EXTENSIONS | {".msh"}


@dataclass
class CadInfo:
    """Metadata extracted from a CAD geometry file.

    Returned by :func:`inspect_cad` to let the user inspect geometry
    dimensions and the recommended mesh size before committing to a
    full volumetric mesh via :func:`load_cad`.
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

    For each candidate axis, projects per-entity bounding-box corners onto the
    perpendicular plane and computes the angular span.  The axis whose
    angular span is closest to 360/num_sectors degrees is the rotation axis.

    Uses ``getBoundingBox(dim, tag)`` which returns the tight spatial bounds
    of each trimmed entity, avoiding the untrimmed-parametric-domain issue
    that plagues ``getParametrizationBounds``/``getValue``.

    Must be called within an active gmsh session with a loaded model.
    """
    import gmsh

    expected_span = 2 * np.pi / num_sectors

    surfaces = gmsh.model.getEntities(dim=2)
    points = []
    for dim, tag in surfaces:
        try:
            sb = gmsh.model.getBoundingBox(dim, tag)
            # sb = (xmin, ymin, zmin, xmax, ymax, zmax)
            for x in (sb[0], sb[3]):
                for y in (sb[1], sb[4]):
                    for z in (sb[2], sb[5]):
                        points.append([x, y, z])
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
                "MM": "mm", "MILLIMETER": "mm", "MILLIMETRE": "mm",
                "CM": "cm", "CENTIMETER": "cm", "CENTIMETRE": "cm",
                "M": "m", "METER": "m", "METRE": "m",
                "IN": "inch", "INCH": "inch",
                "FT": "ft", "FOOT": "ft",
                "UM": "um", "MICRON": "um", "MICROMETER": "um",
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
        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        info = _build_cad_info(filepath, num_sectors,
                               rotation_axis=rotation_axis, units=units)
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
        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        info = _build_cad_info(filepath, num_sectors,
                               rotation_axis=rotation_axis, units=units)

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
    verbosity: int = 0,
) -> Mesh:
    """Load a CAD file and generate a TET10 mesh for cyclic symmetry analysis.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep)
    num_sectors : number of sectors in the full annulus
    mesh_size : characteristic mesh element size (None = auto)
    order : element order (2 = quadratic TET10)
    left_boundary_name : physical group name for the left cyclic boundary
    right_boundary_name : physical group name for the right cyclic boundary
    hub_name : physical group name for the hub constraint (None = no hub)
    auto_detect_boundaries : automatically identify cyclic boundary surfaces
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

        gmsh.model.occ.importShapes(str(filepath))
        gmsh.model.occ.synchronize()

        # Set mesh size
        if mesh_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)

        # Auto-detect cyclic boundaries if requested
        if auto_detect_boundaries:
            _auto_detect_cyclic_boundaries(
                num_sectors, left_boundary_name, right_boundary_name, hub_name
            )

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)

        # Extract mesh data
        mesh = _gmsh_to_mesh(num_sectors)
    finally:
        gmsh.finalize()

    return mesh


def _auto_detect_cyclic_boundaries(
    num_sectors: int,
    left_name: str,
    right_name: str,
    hub_name: str | None,
) -> None:
    """Auto-detect cyclic boundary surfaces and hub from geometry.

    Identifies surfaces whose outward normals are approximately tangential
    to the sector angle boundaries. The surface at theta=0 is 'left_boundary',
    the surface at theta=2*pi/N is 'right_boundary'. The innermost surface
    (smallest mean radius) is the hub.
    """
    import gmsh

    sector_angle = 2 * np.pi / num_sectors
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
        x = (sb[0] + sb[3]) / 2
        y = (sb[1] + sb[4]) / 2
        z = (sb[2] + sb[5]) / 2
        # Normal at parametric centre (direction only — still valid)
        try:
            bounds_min, bounds_max = gmsh.model.getParametrizationBounds(dim, tag)
            u_mid = (bounds_min[0] + bounds_max[0]) / 2
            v_mid = (bounds_min[1] + bounds_max[1]) / 2
            normal = gmsh.model.getNormal(tag, [u_mid, v_mid])
        except Exception:
            normal = [0.0, 0.0, 1.0]  # fallback
        theta = np.arctan2(y, x)
        if theta < -0.01:
            theta += 2 * np.pi
        r = np.sqrt(x**2 + y**2)
        surface_data.append({
            "tag": tag,
            "normal": np.array(normal[:3]),
            "center": np.array([x, y, z]),
            "theta": theta,
            "radius": r,
        })

    if not surface_data:
        return

    # Find surfaces near theta=0 and theta=sector_angle
    left_candidates = []
    right_candidates = []
    hub_candidates = []

    for sd in surface_data:
        nx, ny, nz = sd["normal"]
        # Normal of a cyclic cut surface is roughly tangential (perpendicular to radial)
        # Check if normal is mostly in the tangential direction
        theta = sd["theta"]
        r = sd["radius"]

        # For the left boundary (theta ~ 0): normal is ~ (0, -1, 0) rotated
        # For the right boundary (theta ~ sector_angle): normal rotated by sector_angle
        # Simple heuristic: check if the surface center theta is near 0 or sector_angle
        center_theta = sd["theta"]
        mean_z_component = abs(nz)

        # Hub: surface with normal pointing inward (radially) and low z-component of normal
        if mean_z_component < 0.3 and r < np.median([s["radius"] for s in surface_data]):
            # Check if normal is roughly radially inward
            radial_dir = np.array([sd["center"][0], sd["center"][1], 0])
            if np.linalg.norm(radial_dir) > 1e-10:
                radial_dir /= np.linalg.norm(radial_dir)
                dot = np.dot(sd["normal"], radial_dir)
                if abs(dot) > 0.7:
                    hub_candidates.append(sd)

        # Left/right: surfaces with normals mostly in the xy-plane and tangential
        if mean_z_component < 0.5:
            if abs(center_theta) < sector_angle * 0.15 or abs(center_theta - 2 * np.pi) < sector_angle * 0.15:
                left_candidates.append(sd)
            elif abs(center_theta - sector_angle) < sector_angle * 0.15:
                right_candidates.append(sd)

    # Create physical groups
    if left_candidates:
        left_tags = [s["tag"] for s in left_candidates]
        pg = gmsh.model.addPhysicalGroup(2, left_tags)
        gmsh.model.setPhysicalName(2, pg, left_name)

    if right_candidates:
        right_tags = [s["tag"] for s in right_candidates]
        pg = gmsh.model.addPhysicalGroup(2, right_tags)
        gmsh.model.setPhysicalName(2, pg, right_name)

    # Enforce periodic (node-matched) meshing on cyclic boundary pairs.
    # Without this, gmsh meshes each face independently and the cyclic
    # symmetry solver cannot pair boundary nodes.
    if left_candidates and right_candidates:
        cos_a = np.cos(sector_angle)
        sin_a = np.sin(sector_angle)
        # 4x4 affine rotation matrix about Z (flat row-major for gmsh)
        rot = [
            cos_a, -sin_a, 0, 0,
            sin_a,  cos_a, 0, 0,
            0,      0,     1, 0,
            0,      0,     0, 1,
        ]
        for lt, rt in zip(left_tags, right_tags):
            gmsh.model.mesh.setPeriodic(2, [rt], [lt], rot)

    if hub_name and hub_candidates:
        hub_tags = [s["tag"] for s in hub_candidates]
        pg = gmsh.model.addPhysicalGroup(2, hub_tags)
        gmsh.model.setPhysicalName(2, pg, hub_name)

    # Add volume physical group (needed for gmsh to include volume elements)
    volumes = gmsh.model.getEntities(dim=3)
    if volumes:
        vol_tags = [v[1] for v in volumes]
        pg = gmsh.model.addPhysicalGroup(3, vol_tags)
        gmsh.model.setPhysicalName(3, pg, "volume")


def _gmsh_to_mesh(num_sectors: int) -> Mesh:
    """Extract mesh data from gmsh and build a turbomodal Mesh."""
    import gmsh

    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
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
            # Convert gmsh tags to 0-based indices
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
        raise RuntimeError("No TET10 elements found in mesh. Ensure 3D meshing with order=2.")

    # Extract physical group node sets
    node_sets = []
    phys_groups = gmsh.model.getPhysicalGroups()
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if not name:
            continue
        # Get entities in this physical group
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        group_nodes = set()
        for ent in entities:
            n_tags, _, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
            for nt in n_tags:
                if int(nt) in tag_to_idx:
                    group_nodes.add(tag_to_idx[int(nt)])
        ns = NodeSet()
        ns.name = name
        ns.node_ids = sorted(group_nodes)
        node_sets.append(ns)

    # Build Mesh
    mesh = Mesh()
    mesh.load_from_arrays(
        coords.astype(np.float64),
        tet10_connectivity.astype(np.int32),
        node_sets,
        num_sectors,
    )
    return mesh


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
            if key.startswith("nset_") or key in ("left_boundary", "right_boundary", "hub"):
                ns = NodeSet()
                ns.name = key.replace("nset_", "")
                ns.node_ids = sorted(int(i) for i in np.where(data > 0)[0])
                if ns.node_ids:
                    node_sets.append(ns)

    mesh = Mesh()
    mesh.load_from_arrays(coords, connectivity, node_sets, num_sectors)
    return mesh

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


def inspect_cad(
    filepath: str | Path,
    num_sectors: int,
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

        info = _build_cad_info(filepath, num_sectors)
    finally:
        gmsh.finalize()

    return info


def _extract_surface_tessellation(
    filepath: str | Path,
    num_sectors: int,
    surface_mesh_size: float | None = None,
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
    verbosity : gmsh verbosity level (0=silent)

    Returns
    -------
    nodes : (N, 3) float64 array of node coordinates
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

        info = _build_cad_info(filepath, num_sectors)

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

    return coords.astype(np.float64), tri_nodes, info


def _build_cad_info(filepath: Path, num_sectors: int) -> CadInfo:
    """Build CadInfo from an active gmsh model (must be called within gmsh session)."""
    import gmsh

    sector_angle_deg = 360.0 / num_sectors

    # Bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    bbox = dict(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

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

    # Compute inner/outer radius by sampling surface parametric points
    radii = []
    for dim, tag in surfaces:
        try:
            bounds_min, bounds_max = gmsh.model.getParametrizationBounds(dim, tag)
            # Sample a grid of points on each surface
            for u_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                for v_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    u = bounds_min[0] + u_frac * (bounds_max[0] - bounds_min[0])
                    v = bounds_min[1] + v_frac * (bounds_max[1] - bounds_min[1])
                    pt = gmsh.model.getValue(dim, tag, [u, v])
                    r = np.sqrt(pt[0] ** 2 + pt[1] ** 2)
                    radii.append(r)
        except Exception:
            pass

    if radii:
        inner_radius = float(min(radii))
        outer_radius = float(max(radii))
    else:
        # Fallback: estimate from bounding box
        corners_r = [
            np.sqrt(xmin**2 + ymin**2),
            np.sqrt(xmax**2 + ymax**2),
            np.sqrt(xmin**2 + ymax**2),
            np.sqrt(xmax**2 + ymin**2),
        ]
        inner_radius = float(min(corners_r))
        outer_radius = float(max(corners_r))

    axial_length = float(zmax - zmin)
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

    # Compute mean normal and mean position of each surface
    surface_data = []
    for dim, tag in surfaces:
        # Get parametric bounds — returns ([umin, vmin], [umax, vmax])
        bounds_min, bounds_max = gmsh.model.getParametrizationBounds(dim, tag)
        u_mid = (bounds_min[0] + bounds_max[0]) / 2
        v_mid = (bounds_min[1] + bounds_max[1]) / 2
        # Evaluate point and normal at center
        pt = gmsh.model.getValue(dim, tag, [u_mid, v_mid])
        normal = gmsh.model.getNormal(tag, [u_mid, v_mid])
        x, y, z = pt[0], pt[1], pt[2]
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

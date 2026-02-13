"""Geometry and mesh import for turbomodal.

Supports CAD files (STEP, IGES, BREP, STL) via gmsh and
pre-meshed files (NASTRAN, Abaqus, VTK, gmsh MSH) via meshio.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from turbomodal._core import Mesh, NodeSet


# File extensions handled by gmsh OpenCASCADE kernel
_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".brep", ".stl"}

# File extensions handled by meshio
_MESHIO_EXTENSIONS = {
    ".bdf", ".nas",   # NASTRAN
    ".inp",           # Abaqus
    ".vtk", ".vtu",   # VTK
    ".cgns",          # CGNS
    ".med",           # Salome MED
    ".xdmf",          # XDMF
}


def load_cad(
    filepath: str | Path,
    num_sectors: int,
    mesh_size: float | None = None,
    order: int = 2,
    left_boundary_name: str = "left_boundary",
    right_boundary_name: str = "right_boundary",
    hub_name: str | None = "hub",
    auto_detect_boundaries: bool = True,
    verbosity: int = 0,
) -> Mesh:
    """Load a CAD file and generate a TET10 mesh for cyclic symmetry analysis.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep, .stl)
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
    """
    import gmsh

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CAD file not found: {filepath}")

    ext = filepath.suffix.lower()
    if ext not in _CAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported CAD format '{ext}'. Supported: {sorted(_CAD_EXTENSIONS)}"
        )

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)

        # Import geometry via OpenCASCADE
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
        # Get parametric bounds
        umin, vmin, umax, vmax = gmsh.model.getParametrizationBounds(dim, tag)
        u_mid = (umin + umax) / 2
        v_mid = (vmin + vmax) / 2
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

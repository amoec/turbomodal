"""Visualization for turbomodal: mesh, mode shapes, Campbell and ZZENF diagrams."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from turbomodal._core import Mesh, ModalResult


# VTK cell type for quadratic tetrahedron (TET10)
_VTK_QUADRATIC_TETRA = 24


def _save_gif(filename: str, frames: list, duration: int = 33) -> None:
    """Save frames as an animated GIF with a shared global palette."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    # Quantize all frames to a single palette derived from the first frame
    # so the colorbar gradient doesn't flicker between frames.
    palette = imgs[0].quantize(colors=256)
    quantized = [img.quantize(palette=palette, dither=0) for img in imgs]
    quantized[0].save(
        filename, save_all=True, append_images=quantized[1:],
        duration=duration, loop=0,
    )


def _apply_camera(plotter, camera_position, rotation_axis: int = 2) -> None:
    """Set the camera on *plotter* according to *camera_position*.

    Parameters
    ----------
    plotter : pyvista.Plotter
    camera_position : 'auto', 'xy', 'xz', 'yz', 'iso', tuple, or None
    rotation_axis : 0 (X), 1 (Y), or 2 (Z)
    """
    if camera_position is None:
        return

    if isinstance(camera_position, tuple):
        plotter.camera_position = camera_position
        return

    if camera_position == "auto":
        # Look from an elevated angle with the rotation axis pointing up.
        up = [0, 0, 0]
        up[rotation_axis] = 1
        # Camera position: offset in the two non-axis directions
        axes = [i for i in range(3) if i != rotation_axis]
        pos = [0, 0, 0]
        pos[axes[0]] = 1
        pos[axes[1]] = -0.5
        pos[rotation_axis] = 0.6
        plotter.camera_position = [pos, [0, 0, 0], up]
        plotter.reset_camera()
        return

    view_map = {"xy": "xy", "xz": "xz", "yz": "yz", "iso": "iso"}
    if camera_position in view_map:
        plotter.view_vector(
            *{
                "xy": ([0, 0, 1], [0, 1, 0]),
                "xz": ([0, -1, 0], [0, 0, 1]),
                "yz": ([1, 0, 0], [0, 0, 1]),
                "iso": ([1, -1, 0.5], [0, 0, 1]),
            }[camera_position]
        )
        plotter.reset_camera()
        return


def _mesh_to_pyvista(mesh: Mesh):
    """Convert a turbomodal Mesh to a PyVista UnstructuredGrid."""
    import pyvista as pv

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    n_elem = elements.shape[0]

    # PyVista cell array format: [n_points, p0, p1, ..., p9, ...]
    cells = np.empty((n_elem, 11), dtype=np.int64)
    cells[:, 0] = 10
    cells[:, 1:] = elements

    celltypes = np.full(n_elem, _VTK_QUADRATIC_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, nodes)
    return grid


from turbomodal._utils import rotation_matrix_3x3 as _rotation_matrix


def format_condition_label(condition) -> str:
    """Format an OperatingCondition as a compact plot annotation string.

    Only non-default fields are included. Returns an empty string when
    all fields are at their defaults.

    Parameters
    ----------
    condition : OperatingCondition

    Returns
    -------
    str
        e.g. ``"T=500K"`` or ``""`` when everything is default.
    """
    parts: list[str] = []
    if condition.temperature != 293.15:
        parts.append(f"T={condition.temperature:.0f}K")
    return ", ".join(parts)


def _replicate_sectors(
    nodes: np.ndarray,
    cells_per_sector: np.ndarray,
    n_sectors: int,
    celltypes: np.ndarray,
    rotation_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a sector mesh around the rotation axis.

    Parameters
    ----------
    nodes : (N, 3) node coordinates for one sector
    cells_per_sector : (E, K+1) cell array in PyVista format (first col = n_pts)
    n_sectors : number of sectors
    celltypes : (E,) VTK cell type array for one sector
    rotation_axis : 0=X, 1=Y, 2=Z

    Returns
    -------
    all_nodes : (N*n_sectors, 3) replicated nodes
    all_cells : (E*n_sectors, K+1) replicated cell array
    all_celltypes : (E*n_sectors,) replicated cell types
    """
    n_nodes = len(nodes)
    sector_angle = 2 * np.pi / n_sectors

    all_nodes_list = []
    all_cells_list = []
    all_celltypes_list = []

    for s in range(n_sectors):
        theta = s * sector_angle
        R = _rotation_matrix(theta, rotation_axis)
        rotated = nodes @ R.T

        offset_cells = cells_per_sector.copy()
        offset_cells[:, 1:] += s * n_nodes

        all_nodes_list.append(rotated)
        all_cells_list.append(offset_cells)
        all_celltypes_list.append(celltypes)

    return (
        np.vstack(all_nodes_list),
        np.vstack(all_cells_list),
        np.concatenate(all_celltypes_list),
    )


def _format_dimension(value: float, unit: str = "unknown") -> str:
    """Format a dimension value with appropriate units.

    *value* is always in **metres** (guaranteed by setting
    ``Geometry.OCCTargetUnit = "M"`` before CAD import).
    *unit* is the declared source unit from the CAD file and
    controls which display unit is used.
    """
    if unit == "mm":
        mm = value * 1000
        if abs(mm) < 0.1:
            return f"{mm * 1000:.1f} um"
        return f"{mm:.2f} mm"
    if unit == "cm":
        return f"{value * 100:.3f} cm"
    if unit == "m":
        if abs(value) < 0.01:
            return f"{value * 1000:.3f} mm"
        return f"{value:.4f} m"
    if unit == "inch":
        return f'{value / 0.0254:.4f}"'
    # Unknown: auto-scale assuming metres
    abs_v = abs(value)
    if abs_v < 0.01:
        return f"{value * 1000:.3f} mm"
    if abs_v < 1.0:
        return f"{value * 1000:.1f} mm"
    return f"{value:.4f} m"


def plot_cad(
    filepath,
    num_sectors: int,
    show_full_disk: bool = False,
    show_dimensions: bool = True,
    surface_mesh_size: float | None = None,
    rotation_axis: int | None = None,
    units: str | None = None,
    off_screen: bool = False,
):
    """Visualize CAD geometry before volumetric meshing.

    Imports the CAD file, generates a lightweight surface triangulation,
    and renders it in PyVista. Optionally shows the full 360-degree
    assembly and dimension annotations including recommended mesh size.

    Parameters
    ----------
    filepath : path to CAD file (.step, .stp, .iges, .igs, .brep)
    num_sectors : number of sectors in the full annulus
    show_full_disk : replicate sector around Z-axis for full 360-degree view
    show_dimensions : add annotations with bounding box, radii, mesh size
    surface_mesh_size : override surface mesh element size (None = auto)
    rotation_axis : override rotation axis detection (0=X, 1=Y, 2=Z, None=auto)
    units : override unit detection ("mm", "m", "cm", "inch", None=auto)
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv
    from pathlib import Path
    from turbomodal.io import _extract_surface_tessellation

    nodes, triangles, info = _extract_surface_tessellation(
        filepath, num_sectors, surface_mesh_size,
        rotation_axis=rotation_axis, units=units,
    )

    # Build PyVista PolyData from surface triangles
    n_tri = len(triangles)
    faces = np.empty((n_tri, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles

    plotter = pv.Plotter(off_screen=off_screen)

    if show_full_disk:
        # VTK_TRIANGLE = 5
        celltypes = np.full(n_tri, 5, dtype=np.uint8)
        all_nodes, all_cells, all_celltypes = _replicate_sectors(
            nodes, faces, num_sectors, celltypes,
        )
        # Alternate colors per sector for visual distinction
        sector_ids = np.repeat(np.arange(num_sectors), len(nodes))
        grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, all_nodes)
        grid.point_data["sector"] = sector_ids.astype(np.float32)
        plotter.add_mesh(
            grid, scalars="sector", cmap="Set3", show_edges=True,
            edge_color="gray", line_width=0.5, opacity=0.9,
            show_scalar_bar=False,
        )
        title = f"Full Disk Preview ({num_sectors} sectors)"
    else:
        surface = pv.PolyData(nodes, faces.ravel())
        plotter.add_mesh(
            surface, color="lightblue", show_edges=True,
            edge_color="gray", line_width=0.5, opacity=0.9,
        )
        title = f"Sector Preview (1/{num_sectors})"

    plotter.add_text(title, font_size=12, position="upper_edge")

    if show_dimensions:
        u = info.detected_unit
        fmt = lambda v: _format_dimension(v, u)
        axis_label = {0: "X", 1: "Y", 2: "Z"}.get(info.rotation_axis, "?")
        unit_note = f" [{u}]" if u != "unknown" else ""
        info_text = (
            f"File: {Path(filepath).name}{unit_note}\n"
            f"Sectors: {info.num_sectors} "
            f"({info.sector_angle_deg:.1f} deg)\n"
            f"Rotation axis: {axis_label}\n"
            f"Inner radius: {fmt(info.inner_radius)}\n"
            f"Outer radius: {fmt(info.outer_radius)}\n"
            f"Radial span: {fmt(info.radial_span)}\n"
            f"Axial length: {fmt(info.axial_length)}\n"
            f"Recommended mesh size: {fmt(info.recommended_mesh_size)}"
        )
        plotter.add_text(info_text, position="upper_left", font_size=9)
        plotter.add_bounding_box(color="gray", opacity=0.3)

    plotter.add_axes()
    return plotter


def plot_full_mesh(
    mesh: Mesh,
    off_screen: bool = False,
):
    """Plot the full 360-degree mesh without requiring a ModalResult.

    Replicates the single sector mesh around the rotation axis to show the
    complete annular geometry.

    Parameters
    ----------
    mesh : Mesh object (single sector)
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    n_elem = elements.shape[0]
    n_sectors = mesh.num_sectors

    # Build per-sector cell array
    cells = np.empty((n_elem, 11), dtype=np.int64)
    cells[:, 0] = 10
    cells[:, 1:] = elements
    celltypes = np.full(n_elem, _VTK_QUADRATIC_TETRA, dtype=np.uint8)

    all_nodes, all_cells, all_celltypes = _replicate_sectors(
        nodes, cells, n_sectors, celltypes,
        rotation_axis=mesh.rotation_axis,
    )

    grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, all_nodes)

    # Color by sector for visual distinction
    n_nodes_per = len(nodes)
    sector_ids = np.repeat(np.arange(n_sectors), n_nodes_per)
    grid.point_data["sector"] = sector_ids.astype(np.float32)

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(
        grid, scalars="sector", cmap="Set3", show_edges=True,
        edge_color="gray", line_width=0.5, opacity=0.7,
        show_scalar_bar=False,
    )
    axis_label = {0: "X", 1: "Y", 2: "Z"}.get(mesh.rotation_axis, "Z")
    plotter.add_text(
        f"Full Mesh ({n_sectors} sectors, "
        f"{mesh.num_elements() * n_sectors} elements, axis={axis_label})",
        font_size=12,
    )
    plotter.add_axes()
    return plotter


def plot_mesh(
    mesh: Mesh,
    show_boundaries: bool = True,
    show_node_sets: bool = True,
    show_node_classes: bool = True,
    off_screen: bool = False,
):
    """Plot the sector mesh with boundary highlights.

    Parameters
    ----------
    mesh : Mesh object
    show_boundaries : highlight left/right cyclic boundary nodes as spheres
    show_node_sets : highlight hub and other node sets as spheres
    show_node_classes : color every surface node by classification
        (interior=gray, left=blue, right=red, hub=green)
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    grid = _mesh_to_pyvista(mesh)
    nodes = np.asarray(mesh.nodes)

    # Extract outer surface so interior tet faces don't occlude boundary points
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(surface, color="lightgray", opacity=0.3, show_edges=True,
                     edge_color="gray", line_width=0.5)

    if show_node_classes:
        # Classify every node and render as colored point spheres per class.
        # This avoids VTK scalar interpolation artifacts on triangle faces.
        n_nodes = mesh.num_nodes()
        left_set = set(mesh.left_boundary)
        right_set = set(mesh.right_boundary)
        free_set = (
            set(mesh.free_boundary)
            if hasattr(mesh, "free_boundary")
            else set()
        )
        hub_ns = next(
            (ns for ns in mesh.node_sets if ns.name == "hub_constraint"), None
        )
        hub_set = set(hub_ns.node_ids) if hub_ns else set()

        interior_ids, left_ids, right_ids, hub_ids, free_ids = [], [], [], [], []
        for i in range(n_nodes):
            if i in free_set:
                free_ids.append(i)
            elif i in left_set:
                left_ids.append(i)
            elif i in right_set:
                right_ids.append(i)
            elif i in hub_set:
                hub_ids.append(i)
            else:
                interior_ids.append(i)

        # Compute a sphere radius from the mesh bounding box so that glyphs
        # have a consistent physical size across platforms (GPU-based
        # render_points_as_spheres is unreliable on Windows).
        bbox = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        diag = np.sqrt(
            (bbox[1] - bbox[0]) ** 2
            + (bbox[3] - bbox[2]) ** 2
            + (bbox[5] - bbox[4]) ** 2
        )
        r_small = diag * 0.003
        r_large = diag * 0.006

        class_groups = [
            (interior_ids, "#aaaaaa", r_small, "Interior"),
            (left_ids, "#2166ac", r_large, "Left boundary"),
            (right_ids, "#d62728", r_large, "Right boundary"),
            (hub_ids, "#2ca02c", r_large, "Hub"),
            (free_ids, "#ff7f0e", r_large, "Free boundary"),
        ]
        for ids, color, radius, label in class_groups:
            if ids:
                pts = pv.PolyData(nodes[ids])
                glyphs = pts.glyph(
                    geom=pv.Sphere(radius=radius), scale=False, orient=False,
                )
                plotter.add_mesh(glyphs, color=color, label=label)

    if show_boundaries and not show_node_classes:
        bbox = grid.bounds
        diag = np.sqrt(
            (bbox[1] - bbox[0]) ** 2
            + (bbox[3] - bbox[2]) ** 2
            + (bbox[5] - bbox[4]) ** 2
        )
        r = diag * 0.006
        left = mesh.left_boundary
        right = mesh.right_boundary
        if left:
            pts = pv.PolyData(nodes[left])
            glyphs = pts.glyph(
                geom=pv.Sphere(radius=r), scale=False, orient=False,
            )
            plotter.add_mesh(glyphs, color="blue", label="Left boundary")
        if right:
            pts = pv.PolyData(nodes[right])
            glyphs = pts.glyph(
                geom=pv.Sphere(radius=r), scale=False, orient=False,
            )
            plotter.add_mesh(glyphs, color="red", label="Right boundary")

    if show_node_sets and not show_node_classes:
        bbox = grid.bounds
        diag = np.sqrt(
            (bbox[1] - bbox[0]) ** 2
            + (bbox[3] - bbox[2]) ** 2
            + (bbox[5] - bbox[4]) ** 2
        )
        r = diag * 0.005
        colors = ["green", "orange", "cyan", "magenta", "yellow"]
        ci = 0
        for ns in mesh.node_sets:
            if ns.name in ("left_boundary", "right_boundary"):
                continue
            if ns.node_ids:
                pts = pv.PolyData(nodes[ns.node_ids])
                glyphs = pts.glyph(
                    geom=pv.Sphere(radius=r), scale=False, orient=False,
                )
                plotter.add_mesh(glyphs, color=colors[ci % len(colors)],
                                 label=ns.name)
                ci += 1

    axis_label = {0: "X", 1: "Y", 2: "Z"}.get(mesh.rotation_axis, "Z")
    sector_angle = 360.0 / mesh.num_sectors if mesh.num_sectors > 0 else 0
    plotter.add_text(
        f"Sector Mesh ({mesh.num_sectors} sectors, "
        f"{sector_angle:.1f} deg, axis={axis_label})",
        font_size=12, position="upper_edge",
    )
    plotter.add_legend()
    plotter.add_axes()
    return plotter


def plot_mode(
    mesh: Mesh,
    result: ModalResult,
    mode_index: int = 0,
    scale: float = 1.0,
    component: str = "magnitude",
    off_screen: bool = False,
    full_annulus: bool = False,
    animate: bool = False,
    n_frames: int = 60,
    filename: str | None = None,
    camera_position: str | tuple | None = "auto",
):
    """Plot or animate a mode shape on the sector or full-annulus mesh.

    Parameters
    ----------
    mesh : Mesh object (single sector)
    result : ModalResult from solver
    mode_index : which mode to display (0-based)
    scale : displacement amplification factor
    component : 'magnitude', 'x', 'y', 'z', 'real', 'imag'
        (only used for single-sector static view)
    off_screen : render off-screen (for testing)
    full_annulus : if True, reconstruct all sectors and display the full
        360-degree mode shape.  For harmonic index k and sector s, the
        displacement is: ``u_s = Re(u_sector * exp(i * k * s * 2*pi/N))``
    animate : if True, animate the mode shape oscillation:
        ``u(t) = Re(mode * exp(i * 2*pi*t/T))``
    n_frames : number of animation frames (only used when ``animate=True``)
    filename : if given with ``animate=True``, save the animation as a GIF
    camera_position : camera view specification.
        ``'auto'`` (default) orients the view so the rotation axis points up.
        ``'xy'``, ``'xz'``, ``'yz'``, ``'iso'`` for standard views.
        A tuple ``((cx,cy,cz), (fx,fy,fz), (ux,uy,uz))`` for full control
        (camera position, focal point, view-up vector).
        ``None`` to use PyVista's default.

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    nodes = np.asarray(mesh.nodes)
    n_nodes = mesh.num_nodes()

    # Extract mode shape (complex, ndof x 1)
    mode = np.asarray(result.mode_shapes[:, mode_index])
    mode_3d = mode.reshape(n_nodes, 3)

    freq = result.frequencies[mode_index]
    nd = result.harmonic_index
    whirl = result.whirl_direction[mode_index]
    whirl_str = {1: "FW", -1: "BW", 0: ""}

    # --- Full annulus ---
    if full_annulus:
        elements = np.asarray(mesh.elements)
        n_sectors = mesh.num_sectors
        k = nd
        sector_angle = 2 * np.pi / n_sectors
        n_elem = elements.shape[0]

        # Pre-compute per-sector rotations and rotated rest positions
        rotations = []
        rotated_rest = []
        for s in range(n_sectors):
            theta = s * sector_angle
            R = _rotation_matrix(theta, mesh.rotation_axis)
            rotations.append((theta, R))
            rotated_rest.append(nodes @ R.T)

        # Build connectivity (constant across frames)
        all_cells = np.empty((n_sectors * n_elem, 11), dtype=np.int64)
        all_celltypes = np.full(n_sectors * n_elem, _VTK_QUADRATIC_TETRA,
                                dtype=np.uint8)
        for s in range(n_sectors):
            offset = s * n_elem
            cells = all_cells[offset:offset + n_elem]
            cells[:, 0] = 10
            cells[:, 1:] = elements + s * n_nodes

        def _build_annulus(time_phase=1.0):
            """Return (points, scalars) for all sectors at a given time phase."""
            all_pts = np.empty((n_sectors * n_nodes, 3))
            all_scl = np.empty(n_sectors * n_nodes)
            for s, (theta, R) in enumerate(rotations):
                spatial_phase = np.exp(1j * k * theta)
                u_real = np.real(mode_3d * spatial_phase * time_phase) @ R.T
                i = s * n_nodes
                all_pts[i:i + n_nodes] = rotated_rest[s] + scale * u_real
                all_scl[i:i + n_nodes] = np.sqrt(np.sum(u_real ** 2, axis=1)) * scale
            return all_pts, all_scl

        pts, scl = _build_annulus()
        grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, pts)
        grid.point_data["displacement"] = scl

        title = (f"Full Annulus: ND={nd} Mode {mode_index} "
                 f"f={freq:.2f} Hz {whirl_str.get(whirl, '')}")
        clim = [0, np.max(np.abs(mode_3d)) * scale]

        if not animate:
            plotter = pv.Plotter(off_screen=off_screen)
            plotter.add_mesh(grid, scalars="displacement", cmap="turbo",
                             show_edges=False)
            plotter.add_text(title, font_size=12)
            plotter.add_axes()
            _apply_camera(plotter, camera_position, mesh.rotation_axis)
            return plotter

        # Full-annulus animation
        save_only = filename is not None
        plotter = pv.Plotter(off_screen=off_screen or save_only)
        actor = plotter.add_mesh(grid, scalars="displacement", cmap="turbo",
                                 show_edges=False, clim=clim)
        mapper = actor.GetMapper()
        mapper.SetUseLookupTableScalarRange(True)
        lut = mapper.GetLookupTable()
        lut.SetTableRange(*clim)
        plotter.add_text(title, font_size=12)
        plotter.add_axes()

        def _update_annulus(frame_idx):
            phase = np.exp(2j * np.pi * frame_idx / n_frames)
            pts, scl = _build_annulus(phase)
            grid.points = pts
            grid.point_data["displacement"][:] = scl
            grid.GetPointData().Modified()

        _apply_camera(plotter, camera_position, mesh.rotation_axis)

        if save_only:
            frames = []
            for f in range(n_frames):
                _update_annulus(f)
                plotter.render()
                frames.append(plotter.screenshot(return_img=True))
            plotter.close()
            _save_gif(filename, frames)
            return plotter

        def _cb(step):
            _update_annulus(step % n_frames)
            plotter.render()

        plotter.add_timer_event(max_steps=n_frames * 100, duration=33,
                                callback=_cb)
        plotter.show()
        return plotter

    # --- Single-sector animation ---
    if animate:
        grid = _mesh_to_pyvista(mesh)
        title = f"ND={nd} Mode {mode_index} f={freq:.2f} Hz {whirl_str.get(whirl, '')}"
        clim = [0, np.max(np.abs(mode_3d)) * scale]

        save_only = filename is not None
        plotter = pv.Plotter(off_screen=off_screen or save_only)
        plotter.add_mesh(grid, color="lightgray", style="wireframe",
                         opacity=0.3, line_width=0.5)

        deformed_grid = grid.copy()
        disp = np.real(mode_3d) * scale
        deformed_grid.points = nodes + disp
        deformed_grid.point_data["displacement"] = np.sqrt(
            np.sum(disp ** 2, axis=1))
        actor = plotter.add_mesh(deformed_grid, scalars="displacement",
                                 cmap="turbo", show_edges=False, clim=clim)
        mapper = actor.GetMapper()
        mapper.SetUseLookupTableScalarRange(True)
        lut = mapper.GetLookupTable()
        lut.SetTableRange(*clim)
        plotter.add_text(title, font_size=12)
        plotter.add_axes()

        def _update_sector(frame_idx):
            phase = np.exp(2j * np.pi * frame_idx / n_frames)
            disp = np.real(mode_3d * phase) * scale
            deformed_grid.points = nodes + disp
            deformed_grid.point_data["displacement"][:] = np.sqrt(
                np.sum(disp ** 2, axis=1))
            deformed_grid.GetPointData().Modified()

        _apply_camera(plotter, camera_position, mesh.rotation_axis)

        if save_only:
            frames = []
            for f in range(n_frames):
                _update_sector(f)
                plotter.render()
                frames.append(plotter.screenshot(return_img=True))
            plotter.close()
            _save_gif(filename, frames)
            return plotter

        def _cb(step):
            _update_sector(step % n_frames)
            plotter.render()

        plotter.add_timer_event(max_steps=n_frames * 100, duration=33,
                                callback=_cb)
        plotter.show()
        return plotter

    # --- Single-sector static view ---
    grid = _mesh_to_pyvista(mesh)

    if component == "magnitude":
        scalars = np.sqrt(np.sum(np.abs(mode_3d) ** 2, axis=1))
    elif component == "x":
        scalars = np.real(mode_3d[:, 0])
    elif component == "y":
        scalars = np.real(mode_3d[:, 1])
    elif component == "z":
        scalars = np.real(mode_3d[:, 2])
    elif component == "real":
        scalars = np.sqrt(np.sum(np.real(mode_3d) ** 2, axis=1))
    elif component == "imag":
        scalars = np.sqrt(np.sum(np.imag(mode_3d) ** 2, axis=1))
    else:
        raise ValueError(f"Unknown component '{component}'")

    disp = np.real(mode_3d) * scale
    deformed_nodes = nodes + disp
    deformed_grid = grid.copy()
    deformed_grid.points = deformed_nodes

    title = f"ND={nd} Mode {mode_index} f={freq:.2f} Hz {whirl_str.get(whirl, '')}"

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid, color="lightgray", style="wireframe",
                     opacity=0.3, line_width=0.5)
    deformed_grid.point_data["displacement"] = scalars
    plotter.add_mesh(deformed_grid, scalars="displacement", cmap="turbo",
                     show_edges=False)
    plotter.add_text(title, font_size=12)
    plotter.add_axes()
    _apply_camera(plotter, camera_position, mesh.rotation_axis)
    return plotter



# ---------------------------------------------------------------------------
# Mode tracking for Campbell diagrams
# ---------------------------------------------------------------------------

def _mac(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """Modal Assurance Criterion between two complex mode shape vectors."""
    num = np.abs(np.vdot(phi_a, phi_b)) ** 2
    den = np.real(np.vdot(phi_a, phi_a)) * np.real(np.vdot(phi_b, phi_b))
    if den < 1e-30:
        return 0.0
    return float(num / den)


def _find_harmonic(results_at_rpm: list[ModalResult], nd: int) -> ModalResult | None:
    """Find the ModalResult with the given harmonic_index in a list."""
    for r in results_at_rpm:
        if r.harmonic_index == nd:
            return r
    return None


def _track_modes_for_harmonic(
    results: list[list[ModalResult]],
    nd: int,
) -> list[list[int]]:
    """Track modes across RPM points for one nodal diameter using MAC.

    Parameters
    ----------
    results : results[rpm_idx] = list of ModalResult at that RPM
    nd : nodal diameter (harmonic_index) to track

    Returns
    -------
    perm : list of lists, perm[rpm_idx][track_id] = mode_idx at that RPM
           (-1 if that harmonic is missing at that RPM)
    """
    n_rpm = len(results)
    if n_rpm == 0:
        return []

    # Find first RPM that has this harmonic
    first_r = None
    first_i = 0
    for i in range(n_rpm):
        first_r = _find_harmonic(results[i], nd)
        if first_r is not None:
            first_i = i
            break
    if first_r is None:
        return [[] for _ in range(n_rpm)]

    n_modes = len(first_r.frequencies)
    if n_modes == 0:
        return [[] for _ in range(n_rpm)]

    # Initialize perm: fill with [-1]*n_modes for RPMs before first_i
    perm = [[-1] * n_modes for _ in range(first_i)]
    perm.append(list(range(n_modes)))  # identity at first valid RPM

    prev_r = first_r
    prev_perm = perm[-1]

    for i in range(first_i + 1, n_rpm):
        curr_r = _find_harmonic(results[i], nd)
        if curr_r is None:
            perm.append([-1] * n_modes)
            continue

        n_curr = len(curr_r.frequencies)
        n = min(len(prev_r.frequencies), n_curr, n_modes)

        # Build MAC matrix between previous (tracked) modes and current modes
        mac_matrix = np.zeros((n, n_curr))
        prev_shapes = np.asarray(prev_r.mode_shapes)
        curr_shapes = np.asarray(curr_r.mode_shapes)

        for t in range(n):
            prev_m = prev_perm[t] if t < len(prev_perm) else t
            if prev_m < 0 or prev_m >= prev_shapes.shape[1]:
                continue
            phi_prev = prev_shapes[:, prev_m]
            for j in range(n_curr):
                phi_curr = curr_shapes[:, j]
                mac_matrix[t, j] = _mac(phi_prev, phi_curr)

        # Greedy assignment: for each track, pick the best-matching current mode
        curr_perm = [-1] * n_modes
        used = set()

        pairs = []
        for t in range(n):
            for j in range(n_curr):
                pairs.append((mac_matrix[t, j], t, j))
        pairs.sort(reverse=True)

        for mac_val, t, j in pairs:
            if t >= n_modes:
                continue
            if curr_perm[t] != -1 or j in used:
                continue
            curr_perm[t] = j
            used.add(j)

        perm.append(curr_perm)
        prev_r = curr_r
        prev_perm = curr_perm

    return perm


def plot_campbell(
    results: list[list[ModalResult]],
    engine_orders: Sequence[int] | None = None,
    max_freq: float | None = None,
    figsize: tuple[float, float] = (12, 8),
    confidence_bands: dict | None = None,
    crossing_markers: bool = False,
    num_sectors: int = 0,
    condition_label: str = "",
):
    """Plot Campbell diagram from RPM sweep results.

    Uses MAC (Modal Assurance Criterion) to track modes across RPM points,
    ensuring smooth frequency lines even through mode crossings/veerings.
    Rotating-frame eigenfrequencies are converted to stationary-frame
    FW (solid) and BW (dashed) lines.

    Parameters
    ----------
    results : output from rpm_sweep() — results[rpm_idx][harmonic_idx]
    engine_orders : engine order lines to overlay (e.g. [1, 2, 24])
    max_freq : maximum frequency for y-axis (None = auto)
    figsize : matplotlib figure size
    num_sectors : number of sectors (0 = infer from max harmonic index)

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)

    n_rpm = len(results)
    if n_rpm == 0:
        return fig

    # Infer num_sectors if not provided
    if num_sectors <= 0:
        max_nd = max(r.harmonic_index for row in results for r in row)
        num_sectors = 2 * max_nd  # best guess: max ND = N/2
    max_k = num_sectors // 2

    # Collect all distinct nodal diameters across all RPM points
    all_nds = set()
    for row in results:
        for r in row:
            all_nds.add(r.harmonic_index)
    all_nds = sorted(all_nds)
    n_harmonics = len(all_nds)

    # Use a colormap with enough distinct colors
    if n_harmonics <= 10:
        cmap = plt.cm.tab10
    elif n_harmonics <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo

    rpms = np.array([results[i][0].rpm for i in range(n_rpm)])

    for h_color_idx, nd in enumerate(all_nds):
        color = (cmap(h_color_idx / max(n_harmonics - 1, 1))
                 if n_harmonics > 20 else cmap(h_color_idx % cmap.N))

        # Track modes across RPMs using MAC
        perm = _track_modes_for_harmonic(results, nd)

        # Find number of modes from first valid harmonic result
        first_r = None
        for row in results:
            first_r = _find_harmonic(row, nd)
            if first_r is not None:
                break
        if first_r is None:
            continue
        n_modes = len(first_r.frequencies)

        # Check if Coriolis splitting is present (whirl_direction != 0)
        has_coriolis = False
        for row in results:
            r = _find_harmonic(row, nd)
            if r is not None and np.any(np.asarray(r.whirl_direction) != 0):
                has_coriolis = True
                break

        # Determine if this harmonic needs FW/BW splitting
        needs_split = (nd > 0) if has_coriolis else (0 < nd < max_k)

        for track_id in range(n_modes):
            track_rpms = []
            track_rot_freqs = []
            track_whirl = []

            for i in range(n_rpm):
                if i >= len(perm):
                    break
                mode_idx = perm[i][track_id] if track_id < len(perm[i]) else -1
                if mode_idx < 0:
                    track_rpms.append(rpms[i])
                    track_rot_freqs.append(np.nan)
                    track_whirl.append(0)
                    continue

                r = _find_harmonic(results[i], nd)
                if r is None or mode_idx >= len(r.frequencies):
                    track_rpms.append(rpms[i])
                    track_rot_freqs.append(np.nan)
                    track_whirl.append(0)
                    continue

                track_rpms.append(rpms[i])
                track_rot_freqs.append(r.frequencies[mode_idx])
                whirl_arr = np.asarray(r.whirl_direction)
                track_whirl.append(
                    int(whirl_arr[mode_idx]) if mode_idx < len(whirl_arr) else 0
                )

            track_rpms = np.array(track_rpms)
            track_rot_freqs = np.array(track_rot_freqs)
            track_whirl = np.array(track_whirl)
            valid = ~np.isnan(track_rot_freqs)
            if not np.any(valid):
                continue

            if has_coriolis and needs_split:
                # Coriolis modes: each tracked mode is already FW or BW.
                # Convert to stationary frame and plot single line per mode.
                k_omega = nd * track_rpms / 60.0
                stat_freqs = np.abs(track_rot_freqs + track_whirl * k_omega)

                # Determine dominant whirl for this track
                valid_whirl = track_whirl[valid]
                is_fw = np.sum(valid_whirl == 1) >= np.sum(valid_whirl == -1)
                ls = "-" if is_fw else "--"
                lw = 1.2 if is_fw else 1.0
                ms = 3 if is_fw else 2

                label = f"ND={nd}" if track_id == 0 else None
                ax.plot(track_rpms[valid], stat_freqs[valid], ls,
                        color=color, linewidth=lw, label=label,
                        marker=".", markersize=ms)
            elif needs_split:
                # Kinematic splitting: compute stationary-frame FW and BW
                k_omega = nd * track_rpms / 60.0
                fw_freqs = track_rot_freqs + k_omega
                bw_freqs = np.abs(track_rot_freqs - k_omega)

                label_fw = f"ND={nd}" if track_id == 0 else None
                ax.plot(track_rpms[valid], fw_freqs[valid], "-",
                        color=color, linewidth=1.2, label=label_fw,
                        marker=".", markersize=3)
                ax.plot(track_rpms[valid], bw_freqs[valid], "--",
                        color=color, linewidth=1.0,
                        marker=".", markersize=2)
            else:
                # k=0: standing waves, plot single line
                label = f"ND={nd}" if track_id == 0 else None
                ax.plot(track_rpms[valid], track_rot_freqs[valid], "-",
                        color=color, linewidth=1.2, label=label,
                        marker=".", markersize=3)

            # D11: Confidence bands
            if confidence_bands is not None:
                upper = confidence_bands.get("upper", {})
                lower = confidence_bands.get("lower", {})
                key = (nd, track_id)
                if key in upper and key in lower:
                    u = np.array(upper[key])
                    l = np.array(lower[key])
                    if len(u) == len(track_rpms):
                        ax.fill_between(
                            track_rpms[valid], l[valid], u[valid],
                            alpha=0.15, color=color,
                        )

    # D11: Crossing markers — detect where mode tracks cross EO lines
    all_track_data: list[tuple[np.ndarray, np.ndarray]] = []
    if crossing_markers and engine_orders:
        # Collect all valid track data for crossing detection
        for nd in all_nds:
            perm = _track_modes_for_harmonic(results, nd)
            first_r = None
            for row in results:
                first_r = _find_harmonic(row, nd)
                if first_r is not None:
                    break
            if first_r is None:
                continue
            n_modes = len(first_r.frequencies)
            for track_id in range(n_modes):
                tr, tf = [], []
                for i in range(n_rpm):
                    if i >= len(perm):
                        break
                    mode_idx = perm[i][track_id] if track_id < len(perm[i]) else -1
                    r = _find_harmonic(results[i], nd)
                    if mode_idx >= 0 and r is not None and mode_idx < len(r.frequencies):
                        tr.append(rpms[i])
                        tf.append(r.frequencies[mode_idx])
                if len(tr) > 1:
                    all_track_data.append((np.array(tr), np.array(tf)))

        for tr_rpms, tr_freqs in all_track_data:
            for eo in engine_orders:
                eo_freqs = eo * tr_rpms / 60.0
                diff = tr_freqs - eo_freqs
                for j in range(len(diff) - 1):
                    if diff[j] * diff[j + 1] < 0:
                        # Linear interpolation
                        frac = abs(diff[j]) / (abs(diff[j]) + abs(diff[j + 1]))
                        cross_rpm = tr_rpms[j] + frac * (tr_rpms[j + 1] - tr_rpms[j])
                        cross_freq = eo * cross_rpm / 60.0
                        ax.plot(cross_rpm, cross_freq, "rx", markersize=10,
                                markeredgewidth=2, zorder=5)

    # Engine order lines
    if engine_orders:
        for eo in engine_orders:
            eo_freq = eo * rpms / 60.0
            ax.plot(rpms, eo_freq, "k:", linewidth=0.8, alpha=0.4)
            y_pos = eo_freq[-1]
            if max_freq and y_pos > max_freq:
                # Find where EO line intersects max_freq
                cross = np.where(eo_freq <= max_freq)[0]
                if len(cross) > 0:
                    idx = cross[-1]
                    y_pos = eo_freq[idx]
                    ax.text(rpms[idx], y_pos, f" EO={eo}",
                            fontsize=8, va="bottom", alpha=0.6)
            else:
                ax.text(rpms[-1], y_pos, f" EO={eo}",
                        fontsize=8, va="center", alpha=0.6)

    ax.set_xlabel("RPM", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    title = "Campbell Diagram"
    if condition_label:
        title += f"  ({condition_label})"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(rpms[0], rpms[-1])
    if max_freq:
        ax.set_ylim(0, max_freq)
    else:
        ax.set_ylim(bottom=0)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            unique_handles.append(handle)
            unique_labels.append(lbl)
    # Add whirl direction legend entries
    unique_handles.extend([
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, label="FW"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="BW"),
    ])
    unique_labels.extend(["FW", "BW"])

    ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=7,
              ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _unique_nd(pts: list[tuple]) -> list[tuple]:
    """Keep one ``(nd, freq)`` per nodal diameter (first occurrence wins)."""
    seen: set[int] = set()
    out: list[tuple] = []
    for item in pts:
        nd = item[0]
        if nd not in seen:
            seen.add(nd)
            out.append(item)
    return out


def plot_zzenf(
    results_at_rpm: list[ModalResult],
    num_sectors: int,
    max_freq: float | None = None,
    figsize: tuple[float, float] = (12, 7),
    condition_label: str = "",
    connect_families: bool = True,
    mode_ids: list | None = None,
    mesh: "Mesh | None" = None,
    eo_lines: bool = False,
):
    """Plot ZZENF (zig-zag) interference diagram.

    Parameters
    ----------
    results_at_rpm : list of ModalResult (one per harmonic index) at a single RPM
    num_sectors : number of sectors in the full annulus
    max_freq : max frequency for y-axis (``None`` = auto)
    figsize : figure size
    condition_label : optional label appended to the plot title
    connect_families : if ``True``, draw lines connecting modes of the same
        modal family across nodal diameters
    mode_ids : optional pre-computed mode identifications, one list of
        ``ModeIdentification`` per ``ModalResult``.  Used to group modes
        by ``family_label`` (e.g. ``"1B"``, ``"2B"``).
    mesh : optional ``Mesh``.  If *mode_ids* is not provided and a mesh is
        given, ``identify_modes`` is called automatically to obtain
        family labels.
    eo_lines : if ``True``, overlay the engine-order zig-zag excitation
        lines.  Engine order *e* excites nodal diameter ``e mod N``
        (folded into ``0 .. N/2``) at frequency ``e * RPM / 60``.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import defaultdict

    N = num_sectors
    half_N = N // 2

    has_coriolis = any(
        np.any(np.asarray(r.whirl_direction) != 0) for r in results_at_rpm
    )

    # -- Compute mode_ids if mesh is available but mode_ids not provided --
    if mode_ids is None and mesh is not None and connect_families:
        from turbomodal._core import identify_modes as _id_modes
        mode_ids = [_id_modes(r, mesh) for r in results_at_rpm]

    # -- Build family data: families[key] -> [(nd_folded, freq, whirl)] --
    families: dict = defaultdict(list)

    for r_idx, result in enumerate(results_at_rpm):
        nd = result.harmonic_index
        nd_folded = nd if nd <= half_N else N - nd
        is_nondegen = (nd == 0) or (N % 2 == 0 and nd == half_N)
        whirl_arr = np.asarray(result.whirl_direction)
        n_modes = len(result.frequencies)

        if mode_ids is not None:
            ids = mode_ids[r_idx]
            for i in range(min(n_modes, len(ids))):
                freq = float(result.frequencies[i])
                w = int(whirl_arr[i]) if i < len(whirl_arr) else 0
                families[ids[i].family_label].append((nd_folded, freq, w))
        else:
            # Fallback: at degenerate NDs the solver duplicates eigenvalues
            # in pairs so mode_index // 2 recovers the family index.
            for i in range(n_modes):
                freq = float(result.frequencies[i])
                w = int(whirl_arr[i]) if i < len(whirl_arr) else 0
                fkey = i // 2 if not is_nondegen else i
                families[fkey].append((nd_folded, freq, w))

    # -- Sort families by their lowest frequency --
    family_keys = sorted(
        families.keys(),
        key=lambda k: min(f for _, f, _ in families[k]),
    )
    n_families = len(family_keys)

    # -- Colour palette --
    cmap = plt.colormaps["tab10"]
    fam_colors = [cmap(i % 10) for i in range(n_families)]

    # -- Draw --
    fig, ax = plt.subplots(figsize=figsize)

    for fi, fkey in enumerate(family_keys):
        data = families[fkey]
        col = fam_colors[fi]
        fam_label = str(fkey) if mode_ids is not None else f"Family {fi + 1}"

        if connect_families:
            if has_coriolis:
                # Separate FW, BW, and standing (non-degenerate) points
                fw_pts = [(nd, f) for nd, f, w in data if w == 1]
                bw_pts = [(nd, f) for nd, f, w in data if w == -1]
                st_pts = [(nd, f) for nd, f, w in data if w == 0]

                # FW/BW lines share standing anchor points at k=0 and k=N/2
                fw_line = _unique_nd(sorted(st_pts + fw_pts))
                bw_line = _unique_nd(sorted(st_pts + bw_pts))

                if len(fw_line) > 1:
                    nds, fs = zip(*fw_line)
                    ax.plot(nds, fs, "-", color=col, linewidth=1.5, zorder=2)
                if len(bw_line) > 1:
                    nds, fs = zip(*bw_line)
                    ax.plot(nds, fs, "--", color=col, linewidth=1.5, zorder=2)

                # Markers
                for nd, f, w in data:
                    mk = {1: "^", -1: "v"}.get(w, "o")
                    ax.plot(
                        nd, f, mk, color=col, markersize=6,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                    )
            else:
                # No Coriolis — single line per family.
                # Degenerate NDs produce duplicate (nd, freq) pairs; remove them.
                pts = _unique_nd(sorted(set((nd, f) for nd, f, _ in data)))
                if len(pts) > 1:
                    nds, fs = zip(*pts)
                    ax.plot(
                        nds, fs, "-o", color=col, linewidth=1.5,
                        markersize=6, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3,
                    )
                elif pts:
                    ax.plot(
                        pts[0][0], pts[0][1], "o", color=col, markersize=6,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                    )

            # Annotate family label at the rightmost ND
            if mode_ids is not None:
                rightmost = max(data, key=lambda p: (p[0], p[1]))
                ax.annotate(
                    fam_label, (rightmost[0], rightmost[1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color=col, fontweight="bold", va="center",
                )
        else:
            # No family connection — improved individual markers
            for nd, f, w in data:
                if has_coriolis:
                    mk = {1: "^", -1: "v"}.get(w, "o")
                else:
                    mk = "o"
                ax.plot(
                    nd, f, mk, color=col, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                )

    # -- Axes and styling --
    ax.set_xlabel("Nodal Diameter", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)

    rpm = results_at_rpm[0].rpm if results_at_rpm else 0
    title = "ZZENF Diagram"
    if rpm > 0:
        title += f" @ {rpm:.0f} RPM"
    if condition_label:
        title += f"  ({condition_label})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.set_xticks(range(half_N + 1))
    right_pad = 1.5 if (connect_families and mode_ids is not None) else 0.3
    ax.set_xlim(-0.3, half_N + right_pad)
    if max_freq:
        ax.set_ylim(0, max_freq)
    else:
        all_freqs = [f for pts in families.values() for _, f, _ in pts]
        if all_freqs:
            ax.set_ylim(0, max(all_freqs) * 1.08)

    # -- Engine-order zig-zag lines --
    if eo_lines and rpm > 0:
        f1 = rpm / 60.0  # frequency per engine order
        y_top = ax.get_ylim()[1]
        max_eo = int(y_top / f1) + 1
        eo_nds = []
        eo_freqs = []
        for e in range(max_eo + 1):
            nd_raw = e % N
            nd_fold = nd_raw if nd_raw <= half_N else N - nd_raw
            eo_nds.append(nd_fold)
            eo_freqs.append(e * f1)
        ax.plot(
            eo_nds, eo_freqs, "-", color="gray", linewidth=0.8,
            alpha=0.45, zorder=1,
        )
        # Label a few EO numbers at the zig-zag peaks/troughs
        for e in range(max_eo + 1):
            nd_raw = e % N
            nd_fold = nd_raw if nd_raw <= half_N else N - nd_raw
            freq_e = e * f1
            if freq_e > y_top:
                break
            # Annotate at turning points (ND=0 or ND=N/2)
            if nd_fold == 0 or nd_fold == half_N:
                ax.annotate(
                    str(e), (nd_fold, freq_e),
                    fontsize=6, color="gray", alpha=0.7,
                    textcoords="offset points",
                    xytext=(-8 if nd_fold == half_N else 4, 2),
                    va="bottom",
                )

    ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # -- Legend --
    handles = []
    if has_coriolis:
        handles.extend([
            Line2D([0], [0], marker="^", color="gray", linestyle="-",
                   linewidth=1.5, markersize=7, label="Forward (FW)"),
            Line2D([0], [0], marker="v", color="gray", linestyle="--",
                   linewidth=1.5, markersize=7, label="Backward (BW)"),
            Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=7, label="Standing (k=0, N/2)"),
        ])
    if connect_families and n_families <= 12:
        for fi, fkey in enumerate(family_keys):
            lbl = str(fkey) if mode_ids is not None else f"Family {fi + 1}"
            handles.append(
                Line2D([0], [0], color=fam_colors[fi], linewidth=2, label=lbl),
            )
    if eo_lines and rpm > 0:
        handles.append(
            Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.45,
                   label="EO zig-zag"),
        )
    if not handles:
        handles.append(
            Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=7, label="Natural frequency"),
        )
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    fig.tight_layout()
    return fig


def plot_sensor_contribution(
    shap_values: np.ndarray,
    sensor_names: list[str],
    mode_names: list[str],
    features_per_sensor: int,
    figsize: tuple[float, float] = (10, 8),
):
    """Plot sensor contribution heatmap from SHAP values (D8).

    Parameters
    ----------
    shap_values : (n_samples, n_features, n_outputs) SHAP values.
    sensor_names : list of sensor names.
    mode_names : list of mode/output names.
    features_per_sensor : number of features per sensor.
    figsize : figure size.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    # Group SHAP values by sensor: mean absolute across samples and features within sensor
    n_sensors = len(sensor_names)
    n_modes = len(mode_names)

    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]

    n_samples, n_features, n_out = shap_values.shape
    n_out = min(n_out, n_modes)

    contribution = np.zeros((n_sensors, n_out))
    for s in range(n_sensors):
        start = s * features_per_sensor
        end = min(start + features_per_sensor, n_features)
        if start < n_features:
            contribution[s] = np.mean(np.abs(shap_values[:, start:end, :n_out]), axis=(0, 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contribution, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_out))
    ax.set_xticklabels(mode_names[:n_out], rotation=45, ha="right")
    ax.set_yticks(range(n_sensors))
    ax.set_yticklabels(sensor_names)
    ax.set_xlabel("Mode / Output")
    ax.set_ylabel("Sensor")
    ax.set_title("Sensor Contribution Heatmap (Mean |SHAP|)")
    fig.colorbar(im, ax=ax, label="Mean |SHAP value|")
    fig.tight_layout()
    return fig


def interactive_plane_selector(mesh: Mesh):
    """Alias for :func:`bc_editor` (backward compatibility).

    See :func:`bc_editor` for the full interactive boundary condition editor.
    Returns a single :class:`~turbomodal.solver.BoundaryCondition` (the first
    accepted BC, or a default if none were accepted).
    """
    bcs = bc_editor(mesh)
    if bcs:
        return bcs[0]
    from turbomodal.solver import BoundaryCondition
    return BoundaryCondition(
        name="unnamed", type="fixed",
        plane_point=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]),
    )


_BC_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
_BC_TYPES = ["fixed", "displacement", "frictionless"]


def bc_editor(mesh: Mesh):
    """Interactive boundary condition editor.

    Opens a 3D viewer where you can position cutting planes to define
    boundary condition node groups.  Surface nodes on the positive side
    of the plane *and within the selection radius* are highlighted in
    real-time.

    Controls
    --------
    T          cycle BC type (FIXED -> DISPLACEMENT -> FRICTIONLESS)
    X / Y / Z  toggle constrained component (DISPLACEMENT only)
    1 / 2 / 3  snap plane normal to +X / +Y / +Z axis
    4 / 5 / 6  rotate normal +90 deg around X / Y / Z
    7 / 8 / 9  rotate normal -90 deg around X / Y / Z
    F          flip normal (180 deg)
    Enter      accept current BC, start a new one
    Backspace  undo last accepted BC
    Q          finish and return all BCs

    Sliders (bottom of viewer):
    Origin X/Y/Z      translate the cutting plane origin
    Rot X/Y/Z         rotate the cutting plane normal via Euler angles
    Selection Radius  limit selection to nodes within this radius of origin

    Parameters
    ----------
    mesh : Mesh object with cyclic boundaries identified

    Returns
    -------
    list[BoundaryCondition]
    """
    import pyvista as pv
    from turbomodal.solver import BoundaryCondition

    grid = _mesh_to_pyvista(mesh)
    surface = grid.extract_surface(algorithm="dataset_surface")
    all_coords = np.asarray(mesh.nodes)

    # --- Precompute surface node IDs (excluding cyclic boundaries) ---
    surf_point_ids = surface.point_data.get("vtkOriginalPointIds", None)
    if surf_point_ids is None:
        surf_point_ids = np.array(
            mesh.select_nodes_by_plane(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1e30]),
                tolerance=1e30,
            )
        )
    else:
        surf_point_ids = np.asarray(surf_point_ids)

    exclude = set(mesh.left_boundary) | set(mesh.right_boundary)
    mask = np.array([nid not in exclude for nid in surf_point_ids])
    surface_node_ids = surf_point_ids[mask]
    surface_coords = all_coords[surface_node_ids]

    # Mesh bounds and center
    bounds = surface.bounds
    center = np.array([(bounds[0] + bounds[1]) / 2,
                       (bounds[2] + bounds[3]) / 2,
                       (bounds[4] + bounds[5]) / 2])
    bbox_min = np.array([bounds[0], bounds[2], bounds[4]])
    bbox_max = np.array([bounds[1], bounds[3], bounds[5]])
    bbox_range = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_range))
    slider_lo = bbox_min - 0.2 * bbox_range
    slider_hi = bbox_max + 0.2 * bbox_range
    max_radius = bbox_diag * 1.5

    # --- Editor state ---
    state = {
        "type_idx": 0,
        "components": [True, True, True],
        "counter": 1,
        "origin": center.copy(),
        "normal": np.array([0.0, 0.0, 1.0]),
        "accepted": [],
        "accepted_ids": [],
        "widget": None,
        "suppress_cb": False,
        "ready": False,
        "selection_radius": max_radius,  # start with full (unbounded) selection
    }

    def _current_type():
        return _BC_TYPES[state["type_idx"]]

    def _select_nodes():
        """Return mask into surface_node_ids for current plane + radius."""
        n = np.asarray(state["normal"], dtype=np.float64)
        n_hat = n / (np.linalg.norm(n) + 1e-30)
        o = np.asarray(state["origin"], dtype=np.float64)
        dists = surface_coords @ n_hat - o @ n_hat
        sel_mask = dists >= -1e-6

        # Bounded selection: filter by distance from origin on the plane
        radius = state["selection_radius"]
        if radius < max_radius * 0.99:
            delta = surface_coords - o
            normal_comp = np.outer(delta @ n_hat, n_hat)
            on_plane = delta - normal_comp
            plane_dist = np.linalg.norm(on_plane, axis=1)
            sel_mask &= plane_dist <= radius

        # Exclude nodes already claimed by accepted BCs
        claimed = set()
        for ids in state["accepted_ids"]:
            claimed.update(ids.tolist())
        if claimed:
            for i, nid in enumerate(surface_node_ids):
                if nid in claimed:
                    sel_mask[i] = False
        return sel_mask

    def _type_display():
        parts = []
        for i, t in enumerate(_BC_TYPES):
            label = {"displacement": "DISP", "frictionless": "FRICT"}.get(t, t.upper())
            parts.append(f"[{label}]" if i == state["type_idx"] else f" {label} ")
        return "  ".join(parts)

    def _constrained_dof_count(n_nodes):
        ct = _current_type()
        if ct == "fixed":
            return n_nodes * 3
        elif ct == "displacement":
            return n_nodes * sum(state["components"])
        return n_nodes  # frictionless: 1 DOF per node

    def _add_hud_text(plotter, text, name, position, font_size, color):
        """Add text with a dark semi-transparent background."""
        actor = plotter.add_text(
            text, name=name, position=position,
            font_size=font_size, color=color,
        )
        if hasattr(actor, "prop"):
            actor.prop.background_color = "black"
            actor.prop.background_opacity = 0.6
        return actor

    # Glyph radii for cross-platform sphere rendering (GPU-based
    # render_points_as_spheres is unreliable on Windows OpenGL).
    r_sel = bbox_diag * 0.004   # current selection
    r_acc = bbox_diag * 0.0035  # accepted BCs
    _sphere_sel = pv.Sphere(radius=r_sel)
    _sphere_acc = pv.Sphere(radius=r_acc)

    def _refresh(plotter):
        """Redraw all dynamic actors and HUD text."""
        if not state["ready"]:
            return
        sel_mask = _select_nodes()
        n_selected = int(sel_mask.sum())

        # Current selection — yellow glyphs
        if n_selected > 0:
            pts = pv.PolyData(surface_coords[sel_mask])
            glyphs = pts.glyph(geom=_sphere_sel, scale=False, orient=False)
            plotter.add_mesh(glyphs, name="current_sel", color="yellow")
        else:
            try:
                plotter.remove_actor("current_sel")
            except (KeyError, ValueError):
                pass

        # Accepted BC node groups — colored glyphs
        for i, ids in enumerate(state["accepted_ids"]):
            color = _BC_COLORS[i % len(_BC_COLORS)]
            if len(ids) > 0:
                pts = pv.PolyData(all_coords[ids])
                glyphs = pts.glyph(geom=_sphere_acc, scale=False, orient=False)
                plotter.add_mesh(glyphs, name=f"accepted_{i}", color=color)

        # Selection radius visualization — translucent disc
        radius = state["selection_radius"]
        if radius < max_radius * 0.99:
            try:
                disk = pv.Disc(
                    center=state["origin"],
                    normal=state["normal"],
                    inner=0.0, outer=radius,
                    r_res=1, c_res=36,
                )
                plotter.add_mesh(
                    disk, name="radius_disk",
                    color="yellow", opacity=0.12,
                    show_edges=True, edge_color="yellow", line_width=1,
                )
            except (KeyError, ValueError):
                pass
        else:
            try:
                plotter.remove_actor("radius_disk")
            except (KeyError, ValueError):
                pass

        # --- HUD: Status panel (upper left) ---
        name = f"bc_{state['counter']}"
        n_dofs = _constrained_dof_count(n_selected)
        n_total = len(surface_node_ids)
        n_claimed = sum(len(ids) for ids in state["accepted_ids"])

        hud = [
            f"Editing: {name}",
            f"Type: {_type_display()}",
        ]
        if _current_type() == "displacement":
            labels = ["ux", "uy", "uz"]
            parts = [f"[{labels[i]}=0]" if state["components"][i] else f" {labels[i]} "
                     for i in range(3)]
            hud.append(f"DOFs: {'  '.join(parts)}")
        hud += [
            f"Selected: {n_selected} nodes  ({n_dofs} DOFs)",
            f"Available: {n_total - n_claimed}/{n_total} surface nodes",
            f"Origin: ({state['origin'][0]:.4f}, {state['origin'][1]:.4f}, {state['origin'][2]:.4f})",
            f"Normal: ({state['normal'][0]:.3f}, {state['normal'][1]:.3f}, {state['normal'][2]:.3f})",
        ]
        if radius < max_radius * 0.99:
            hud.append(f"Radius: {radius:.4f}")
        _add_hud_text(plotter, "\n".join(hud), "status_text",
                      "upper_left", 9, "white")

        # --- Accepted list (upper right) ---
        if state["accepted"]:
            lines = [f"Accepted ({len(state['accepted'])})"]
            for i, bc in enumerate(state["accepted"]):
                nn = len(state["accepted_ids"][i])
                tp = bc.type.upper()
                if bc.type == "displacement":
                    ax = [c for c, v in zip("xyz", bc.constrained_components) if v]
                    tp += f"({''.join(ax)})"
                lines.append(f"  {bc.name}: {tp} {nn}n")
            _add_hud_text(plotter, "\n".join(lines), "accepted_text",
                          "upper_right", 9, "white")
        else:
            _add_hud_text(plotter, "No BCs accepted", "accepted_text",
                          "upper_right", 9, "gray")

        # --- Controls (lower left) — compact ---
        ct = _current_type().upper()
        ctrl = f"[T] {ct}  [F] Flip  [Enter] Accept  [Backspace] Undo  [Q] Finish"
        ctrl += "\n[1/2/3] Snap  [4/5/6] +90  [7/8/9] -90"
        if _current_type() == "displacement":
            cx = "LOCKED" if state["components"][0] else "free"
            cy = "LOCKED" if state["components"][1] else "free"
            cz = "LOCKED" if state["components"][2] else "free"
            ctrl += f"\n[X] ux={cx}  [Y] uy={cy}  [Z] uz={cz}"
        _add_hud_text(plotter, ctrl, "controls_text",
                      "lower_left", 8, "lightgray")

        plotter.render()

    # --- Programmatic plane control helpers ---
    def _set_plane(origin, normal):
        state["origin"] = np.array(origin, dtype=np.float64)
        n = np.array(normal, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len > 1e-12:
            n = n / n_len
        state["normal"] = n
        w = state["widget"]
        if w is not None:
            state["suppress_cb"] = True
            w.SetOrigin(*state["origin"])
            w.SetNormal(*state["normal"])
            w.UpdatePlacement()
            state["suppress_cb"] = False
        _refresh(plotter)

    def _rotate_normal(axis_char, degrees):
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis_char.lower()]
        rad = np.radians(degrees)
        c, s = np.cos(rad), np.sin(rad)
        R = np.eye(3)
        i, j = [(1, 2), (2, 0), (0, 1)][axis_idx]
        R[i, i] = c;  R[i, j] = -s
        R[j, i] = s;  R[j, j] = c
        _set_plane(state["origin"], R @ state["normal"])

    # --- Build plotter ---
    plotter = pv.Plotter(title="BC Editor - turbomodal")
    plotter.add_mesh(surface, color="lightgray", opacity=0.3, show_edges=True,
                     edge_color="gray", line_width=0.5)

    # Plane widget
    def on_plane(normal, origin, widget=None):
        if widget is not None and state["widget"] is None:
            state["widget"] = widget
        if state["suppress_cb"] or not state["ready"]:
            return
        state["normal"] = np.array(normal)
        state["origin"] = np.array(origin)
        _refresh(plotter)

    plotter.add_plane_widget(
        on_plane,
        normal=state["normal"],
        origin=state["origin"],
        normal_rotation=True,
        interaction_event="end",
        pass_widget=True,
    )

    # --- Sliders: right side, vertical column with generous spacing ---
    axis_labels = ["X", "Y", "Z"]

    # Selection Radius (most important — topmost slider)
    def _on_radius(value):
        if state["suppress_cb"] or not state["ready"]:
            return
        state["selection_radius"] = value
        _refresh(plotter)

    plotter.add_slider_widget(
        _on_radius,
        rng=(0.0, float(max_radius)),
        value=float(max_radius),
        title="Radius",
        pointa=(0.72, 0.55), pointb=(0.98, 0.55),
        style="modern",
    )

    # Origin X/Y/Z sliders
    for ax_i in range(3):
        def _make_origin_cb(idx):
            def _cb(value):
                if state["suppress_cb"] or not state["ready"]:
                    return
                state["origin"][idx] = value
                w = state["widget"]
                if w is not None:
                    state["suppress_cb"] = True
                    w.SetOrigin(*state["origin"])
                    w.UpdatePlacement()
                    state["suppress_cb"] = False
                _refresh(plotter)
            return _cb
        y_pos = 0.40 - ax_i * 0.15  # y=0.40, 0.25, 0.10
        plotter.add_slider_widget(
            _make_origin_cb(ax_i),
            rng=(float(slider_lo[ax_i]), float(slider_hi[ax_i])),
            value=float(center[ax_i]),
            title=f"Orig {axis_labels[ax_i]}",
            pointa=(0.72, y_pos), pointb=(0.98, y_pos),
            style="modern",
        )

    # --- Key bindings ---
    def _cycle_type():
        state["type_idx"] = (state["type_idx"] + 1) % len(_BC_TYPES)
        ct = _current_type().upper()
        print(f"  -> BC type: {ct}")
        if _current_type() == "displacement":
            print("     Press X/Y/Z to toggle which DOFs are constrained")
        _refresh(plotter)

    def _toggle_comp(idx):
        if _current_type() != "displacement":
            print("  -> X/Y/Z toggles only apply in DISPLACEMENT mode")
            return
        state["components"][idx] = not state["components"][idx]
        tag = "LOCKED" if state["components"][idx] else "free"
        print(f"  -> u{'xyz'[idx]}: {tag}")
        _refresh(plotter)

    def _accept():
        sel_mask = _select_nodes()
        selected_ids = surface_node_ids[sel_mask]
        if len(selected_ids) == 0:
            print("  -> No nodes selected, nothing to accept")
            return
        name = f"bc_{state['counter']}"
        radius = state["selection_radius"]
        bc = BoundaryCondition(
            name=name,
            type=_current_type(),
            plane_point=state["origin"].copy(),
            plane_normal=state["normal"].copy(),
            constrained_components=tuple(state["components"]),
            node_ids=selected_ids.tolist(),
            selection_radius=radius if radius < max_radius * 0.99 else None,
        )
        state["accepted"].append(bc)
        state["accepted_ids"].append(selected_ids.copy())
        tp = _current_type().upper()
        if _current_type() == "displacement":
            ax = [c for c, v in zip("xyz", state["components"]) if v]
            tp += f" ({''.join(ax)})"
        print(f"  -> Accepted: {name} | {tp} | {len(selected_ids)} nodes")
        state["counter"] += 1
        state["type_idx"] = 0
        state["components"] = [True, True, True]
        _refresh(plotter)

    def _undo():
        if state["accepted"]:
            removed = state["accepted"].pop()
            idx = len(state["accepted_ids"]) - 1
            state["accepted_ids"].pop()
            try:
                plotter.remove_actor(f"accepted_{idx}")
            except (KeyError, ValueError):
                pass
            state["counter"] = max(1, state["counter"] - 1)
            print(f"  -> Undone: {removed.name}")
            _refresh(plotter)
        else:
            print("  -> Nothing to undo")

    plotter.add_key_event("t", _cycle_type)
    plotter.add_key_event("x", lambda: _toggle_comp(0))
    plotter.add_key_event("y", lambda: _toggle_comp(1))
    plotter.add_key_event("z", lambda: _toggle_comp(2))
    plotter.add_key_event("Return", _accept)
    plotter.add_key_event("BackSpace", _undo)

    # Plane orientation keys
    def _snap(nx, ny, nz):
        print(f"  -> Snap normal to ({nx},{ny},{nz})")
        _set_plane(state["origin"], [nx, ny, nz])

    plotter.add_key_event("1", lambda: _snap(1, 0, 0))
    plotter.add_key_event("2", lambda: _snap(0, 1, 0))
    plotter.add_key_event("3", lambda: _snap(0, 0, 1))

    for key, axis, deg in [("4", "x", 90), ("5", "y", 90), ("6", "z", 90),
                            ("7", "x", -90), ("8", "y", -90), ("9", "z", -90)]:
        def _make_rot(a, d):
            def _cb():
                print(f"  -> Rotate normal {d:+d} deg around {a.upper()}")
                _rotate_normal(a, d)
            return _cb
        plotter.add_key_event(key, _make_rot(axis, deg))

    def _flip():
        print("  -> Flip normal (180 deg)")
        _set_plane(state["origin"], -state["normal"])
    plotter.add_key_event("f", _flip)

    # All widgets created — enable callbacks and do initial refresh
    state["ready"] = True
    _refresh(plotter)
    plotter.reset_camera()
    plotter.show()

    # Auto-accept if user closed without explicit accept
    if not state["accepted"]:
        sel_mask = _select_nodes()
        selected_ids = surface_node_ids[sel_mask]
        if len(selected_ids) > 0:
            radius = state["selection_radius"]
            bc = BoundaryCondition(
                name=f"bc_{state['counter']}",
                type=_current_type(),
                plane_point=state["origin"].copy(),
                plane_normal=state["normal"].copy(),
                constrained_components=tuple(state["components"]),
                node_ids=selected_ids.tolist(),
                selection_radius=radius if radius < max_radius * 0.99 else None,
            )
            state["accepted"].append(bc)

    return state["accepted"]


def plot_boundary_conditions(mesh: Mesh, bcs, off_screen: bool = False):
    """Visualize selected node groups for each boundary condition.

    Parameters
    ----------
    mesh : Mesh object
    bcs : list of BoundaryCondition objects
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    grid = _mesh_to_pyvista(mesh)
    surface = grid.extract_surface(algorithm="dataset_surface")
    nodes_arr = np.asarray(mesh.nodes)

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(surface, color="lightgray", opacity=0.3, show_edges=True,
                     edge_color="gray", line_width=0.5)

    # Glyph sphere for cross-platform rendering (GPU-based
    # render_points_as_spheres is unreliable on Windows OpenGL).
    bounds = grid.bounds
    diag = np.sqrt(
        (bounds[1] - bounds[0]) ** 2
        + (bounds[3] - bounds[2]) ** 2
        + (bounds[5] - bounds[4]) ** 2
    )
    sphere_geom = pv.Sphere(radius=diag * 0.004)

    for i, bc in enumerate(bcs):
        color = _BC_COLORS[i % len(_BC_COLORS)]
        if bc.node_ids is not None:
            selected = list(bc.node_ids)
        else:
            selected = mesh.select_nodes_by_plane(
                np.asarray(bc.plane_point, dtype=np.float64),
                np.asarray(bc.plane_normal, dtype=np.float64),
                bc.tolerance,
            )
        if selected:
            pts = nodes_arr[selected]
            cloud = pv.PolyData(pts)
            glyphs = cloud.glyph(geom=sphere_geom, scale=False, orient=False)
            tp = bc.type.upper()
            if bc.type == "displacement":
                labels = "xyz"
                active = [labels[j] for j in range(3) if bc.constrained_components[j]]
                tp += f" ({','.join(active)})"
            plotter.add_mesh(glyphs, color=color,
                             label=f"{bc.name}: {tp}, {len(selected)} nodes")

    plotter.add_legend()
    if not off_screen:
        plotter.show()
    return plotter

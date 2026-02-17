"""Visualization for turbomodal: mesh, mode shapes, Campbell and ZZENF diagrams."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from turbomodal._core import Mesh, ModalResult


# VTK cell type for quadratic tetrahedron (TET10)
_VTK_QUADRATIC_TETRA = 24


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


def _replicate_sectors(
    nodes: np.ndarray,
    cells_per_sector: np.ndarray,
    n_sectors: int,
    celltypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a sector mesh around the Z-axis.

    Parameters
    ----------
    nodes : (N, 3) node coordinates for one sector
    cells_per_sector : (E, K+1) cell array in PyVista format (first col = n_pts)
    n_sectors : number of sectors
    celltypes : (E,) VTK cell type array for one sector

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
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1],
        ])
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

    *value* is in the **declared unit** of the CAD file (gmsh preserves
    native STEP/IGES coordinates).  *unit* is the declared source unit
    and controls formatting:

    * ``"mm"`` → show as millimetres (drop to um for tiny values)
    * ``"cm"`` → show as centimetres
    * ``"m"``  → show as metres (drop to mm for small values)
    * ``"inch"`` → show as inches
    * ``"unknown"`` → auto-scale for readability
    """
    if unit == "mm":
        if abs(value) < 0.1:
            return f"{value * 1000:.1f} um"
        return f"{value:.2f} mm"
    if unit == "cm":
        return f"{value:.3f} cm"
    if unit == "m":
        if abs(value) < 0.01:
            return f"{value * 1000:.3f} mm"
        return f"{value:.4f} m"
    if unit == "inch":
        return f'{value:.4f}"'
    # Unknown: display raw value
    abs_v = abs(value)
    if abs_v < 0.01:
        return f"{value:.6f}"
    if abs_v > 1000:
        return f"{value:.1f}"
    return f"{value:.4f}"


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

    Replicates the single sector mesh around the Z-axis to show the
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
    plotter.add_text(
        f"Full Mesh ({n_sectors} sectors, {mesh.num_elements() * n_sectors} elements)",
        font_size=12,
    )
    plotter.add_axes()
    return plotter


def plot_mesh(
    mesh: Mesh,
    show_boundaries: bool = True,
    show_node_sets: bool = True,
    off_screen: bool = False,
):
    """Plot the sector mesh with boundary highlights.

    Parameters
    ----------
    mesh : Mesh object
    show_boundaries : highlight left/right cyclic boundary nodes
    show_node_sets : highlight hub and other node sets
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    grid = _mesh_to_pyvista(mesh)
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid, color="lightgray", opacity=0.5, show_edges=True,
                     edge_color="gray", line_width=0.5)

    nodes = np.asarray(mesh.nodes)

    if show_boundaries:
        left = mesh.left_boundary
        right = mesh.right_boundary
        if left:
            pts_left = nodes[left]
            plotter.add_points(pts_left, color="blue", point_size=6,
                               render_points_as_spheres=True, label="Left boundary")
        if right:
            pts_right = nodes[right]
            plotter.add_points(pts_right, color="red", point_size=6,
                               render_points_as_spheres=True, label="Right boundary")

    if show_node_sets:
        colors = ["green", "orange", "cyan", "magenta", "yellow"]
        ci = 0
        for ns in mesh.node_sets:
            if ns.name in ("left_boundary", "right_boundary"):
                continue
            if ns.node_ids:
                pts = nodes[ns.node_ids]
                plotter.add_points(pts, color=colors[ci % len(colors)],
                                   point_size=5, render_points_as_spheres=True,
                                   label=ns.name)
                ci += 1

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
):
    """Plot a mode shape on the sector mesh.

    Parameters
    ----------
    mesh : Mesh object
    result : ModalResult from solver
    mode_index : which mode to display (0-based)
    scale : displacement amplification factor
    component : 'magnitude', 'x', 'y', 'z', 'real', 'imag'
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    grid = _mesh_to_pyvista(mesh)
    nodes = np.asarray(mesh.nodes)
    n_nodes = mesh.num_nodes()

    # Extract mode shape (complex, ndof x 1)
    mode = np.asarray(result.mode_shapes[:, mode_index])
    mode_3d = mode.reshape(n_nodes, 3)

    # Compute scalar field
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

    # Deform mesh by real part of mode shape
    disp = np.real(mode_3d) * scale
    deformed_nodes = nodes + disp
    deformed_grid = grid.copy()
    deformed_grid.points = deformed_nodes

    freq = result.frequencies[mode_index]
    nd = result.harmonic_index
    whirl = result.whirl_direction[mode_index]
    whirl_str = {1: "FW", -1: "BW", 0: ""}
    title = f"ND={nd} Mode {mode_index} f={freq:.2f} Hz {whirl_str.get(whirl, '')}"

    plotter = pv.Plotter(off_screen=off_screen)
    # Wireframe of undeformed
    plotter.add_mesh(grid, color="lightgray", style="wireframe",
                     opacity=0.3, line_width=0.5)
    # Deformed with scalars
    deformed_grid.point_data["displacement"] = scalars
    plotter.add_mesh(deformed_grid, scalars="displacement", cmap="turbo",
                     show_edges=False)
    plotter.add_text(title, font_size=12)
    plotter.add_axes()
    return plotter


def plot_full_annulus(
    mesh: Mesh,
    result: ModalResult,
    mode_index: int = 0,
    scale: float = 1.0,
    off_screen: bool = False,
):
    """Reconstruct and plot the full 360-degree mode shape.

    For harmonic index k and sector s, the displacement is:
        u_s = Re(u_sector * exp(i * k * s * 2*pi/N))

    Parameters
    ----------
    mesh : Mesh object (single sector)
    result : ModalResult from solver
    mode_index : which mode to display
    scale : displacement amplification factor
    off_screen : render off-screen (for testing)

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    n_nodes = mesh.num_nodes()
    n_sectors = mesh.num_sectors
    k = result.harmonic_index

    # Extract complex mode shape
    mode = np.asarray(result.mode_shapes[:, mode_index])
    mode_3d = mode.reshape(n_nodes, 3)

    sector_angle = 2 * np.pi / n_sectors

    plotter = pv.Plotter(off_screen=off_screen)

    for s in range(n_sectors):
        theta = s * sector_angle
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotation matrix around Z axis
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1],
        ])

        # Rotate nodes
        rotated_nodes = nodes @ R.T

        # Phase factor for this sector
        phase = np.exp(1j * k * theta)
        # Complex displacement in sector reference frame
        u_complex = mode_3d * phase
        # Take real part and rotate to global frame
        u_real = np.real(u_complex) @ R.T

        deformed = rotated_nodes + scale * u_real

        # Build PyVista grid for this sector
        n_elem = elements.shape[0]
        offset = s * n_nodes  # node index offset for this sector

        cells = np.empty((n_elem, 11), dtype=np.int64)
        cells[:, 0] = 10
        cells[:, 1:] = elements + offset

        celltypes = np.full(n_elem, _VTK_QUADRATIC_TETRA, dtype=np.uint8)

        # Accumulate all sector nodes
        if s == 0:
            all_nodes = deformed.copy()
            all_cells = cells.copy()
            all_celltypes = celltypes.copy()
            all_scalars = np.sqrt(np.sum(u_real ** 2, axis=1))
        else:
            all_nodes = np.vstack([all_nodes, deformed])
            all_cells = np.vstack([all_cells, cells])
            all_celltypes = np.concatenate([all_celltypes, celltypes])
            all_scalars = np.concatenate([
                all_scalars,
                np.sqrt(np.sum(u_real ** 2, axis=1)),
            ])

    grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, all_nodes)
    grid.point_data["displacement"] = all_scalars

    freq = result.frequencies[mode_index]
    title = f"Full Annulus: ND={k} Mode {mode_index} f={freq:.2f} Hz"

    plotter.add_mesh(grid, scalars="displacement", cmap="turbo", show_edges=False)
    plotter.add_text(title, font_size=12)
    plotter.add_axes()
    return plotter


def animate_mode(
    mesh: Mesh,
    result: ModalResult,
    mode_index: int = 0,
    scale: float = 1.0,
    n_frames: int = 60,
    filename: str | None = None,
    off_screen: bool = False,
):
    """Animate mode shape oscillation.

    The oscillation is: u(t) = Re(mode * exp(i * 2*pi*t/T))

    Parameters
    ----------
    mesh : Mesh object
    result : ModalResult from solver
    mode_index : which mode to animate
    scale : displacement amplification factor
    n_frames : number of animation frames
    filename : if given, save as GIF
    off_screen : render off-screen

    Returns
    -------
    pyvista.Plotter
    """
    import pyvista as pv

    grid = _mesh_to_pyvista(mesh)
    nodes = np.asarray(mesh.nodes)
    n_nodes = mesh.num_nodes()

    mode = np.asarray(result.mode_shapes[:, mode_index])
    mode_3d = mode.reshape(n_nodes, 3)

    freq = result.frequencies[mode_index]
    title = f"ND={result.harmonic_index} Mode {mode_index} f={freq:.2f} Hz"

    plotter = pv.Plotter(off_screen=off_screen or filename is not None)
    plotter.add_mesh(grid, color="lightgray", style="wireframe",
                     opacity=0.3, line_width=0.5)

    # Initial deformed state
    deformed_grid = grid.copy()
    disp = np.real(mode_3d) * scale
    deformed_grid.points = nodes + disp
    scalars = np.sqrt(np.sum(disp ** 2, axis=1))
    deformed_grid.point_data["displacement"] = scalars
    actor = plotter.add_mesh(deformed_grid, scalars="displacement", cmap="turbo",
                             show_edges=False, clim=[0, np.max(np.abs(mode_3d)) * scale])
    plotter.add_text(title, font_size=12)
    plotter.add_axes()

    if filename:
        plotter.open_gif(filename)

    for frame in range(n_frames):
        t = frame / n_frames
        phase = np.exp(2j * np.pi * t)
        disp = np.real(mode_3d * phase) * scale
        deformed_grid.points = nodes + disp
        deformed_grid.point_data["displacement"] = np.sqrt(np.sum(disp ** 2, axis=1))
        if filename:
            plotter.write_frame()
        else:
            plotter.render()

    if filename:
        plotter.close()

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
):
    """Plot Campbell diagram from RPM sweep results.

    Uses MAC (Modal Assurance Criterion) to track modes across RPM points,
    ensuring smooth frequency lines even through mode crossings/veerings.

    Parameters
    ----------
    results : output from rpm_sweep() — results[rpm_idx][harmonic_idx]
    engine_orders : engine order lines to overlay (e.g. [1, 2, 24])
    max_freq : maximum frequency for y-axis (None = auto)
    figsize : matplotlib figure size

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

        for track_id in range(n_modes):
            track_rpms = []
            track_freqs = []
            track_whirls = []

            for i in range(n_rpm):
                if i >= len(perm):
                    break
                mode_idx = perm[i][track_id] if track_id < len(perm[i]) else -1
                if mode_idx < 0:
                    track_rpms.append(rpms[i])
                    track_freqs.append(np.nan)
                    track_whirls.append(0)
                    continue

                r = _find_harmonic(results[i], nd)
                if r is None:
                    track_rpms.append(rpms[i])
                    track_freqs.append(np.nan)
                    track_whirls.append(0)
                    continue

                if mode_idx < len(r.frequencies):
                    track_rpms.append(rpms[i])
                    track_freqs.append(r.frequencies[mode_idx])
                    track_whirls.append(r.whirl_direction[mode_idx])
                else:
                    track_rpms.append(rpms[i])
                    track_freqs.append(np.nan)
                    track_whirls.append(0)

            track_rpms = np.array(track_rpms)
            track_freqs = np.array(track_freqs)
            track_whirls = np.array(track_whirls)

            # Determine dominant whirl direction for this track
            valid = ~np.isnan(track_freqs)
            if not np.any(valid):
                continue

            whirl_vals = track_whirls[valid]
            n_fw = np.sum(whirl_vals > 0)
            n_bw = np.sum(whirl_vals < 0)

            if n_fw > n_bw:
                linestyle = "-"
            elif n_bw > n_fw:
                linestyle = "--"
            else:
                linestyle = "-"

            label = f"ND={nd}" if track_id == 0 else None
            ax.plot(track_rpms[valid], track_freqs[valid], linestyle,
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
    ax.set_title("Campbell Diagram", fontsize=13)
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


def plot_zzenf(
    results_at_rpm: list[ModalResult],
    num_sectors: int,
    max_freq: float | None = None,
    figsize: tuple[float, float] = (10, 8),
    confidence_bands: dict | None = None,
    crossing_markers: bool = False,
):
    """Plot ZZENF (zig-zag) interference diagram.

    Parameters
    ----------
    results_at_rpm : list of ModalResult (one per harmonic index) at a single RPM
    num_sectors : number of sectors in the full annulus
    max_freq : max frequency for y-axis
    figsize : figure size

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    N = num_sectors
    half_N = N // 2

    for result in results_at_rpm:
        nd = result.harmonic_index
        # Fold: for nd > N/2, fold back as N - nd
        nd_folded = nd if nd <= half_N else N - nd

        for m_idx in range(len(result.frequencies)):
            freq = result.frequencies[m_idx]
            whirl = result.whirl_direction[m_idx]

            marker = "^" if whirl > 0 else ("v" if whirl < 0 else "o")
            color = "tab:blue" if whirl > 0 else ("tab:red" if whirl < 0 else "tab:gray")
            ax.plot(nd_folded, freq, marker, color=color, markersize=6)

    ax.set_xlabel("Nodal Diameter")
    ax.set_ylabel("Frequency (Hz)")
    rpm = results_at_rpm[0].rpm if results_at_rpm else 0
    ax.set_title(f"ZZENF Diagram @ {rpm:.0f} RPM")
    ax.set_xticks(range(half_N + 1))
    if max_freq:
        ax.set_ylim(0, max_freq)
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="tab:blue", linestyle="None",
               markersize=8, label="Forward whirl"),
        Line2D([0], [0], marker="v", color="tab:red", linestyle="None",
               markersize=8, label="Backward whirl"),
        Line2D([0], [0], marker="o", color="tab:gray", linestyle="None",
               markersize=8, label="Standing wave"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
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

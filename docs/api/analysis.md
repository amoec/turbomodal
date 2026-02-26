# turbomodal Analysis API Reference

This document covers the visualization and analysis functions in `turbomodal.viz`:
mesh plotting, mode shape display, Campbell diagrams, and ZZENF (zig-zag)
interference diagrams.

All visualization functions require PyVista (mesh/mode plots) or Matplotlib
(Campbell/ZZENF diagrams) to be installed.

---

## CAD Geometry Preview

### plot_cad

Visualize CAD geometry before volumetric meshing. Imports the CAD file,
generates a lightweight surface triangulation, and renders it in PyVista.
Optionally shows the full 360-degree assembly and dimension annotations
including recommended mesh size.

```python
def plot_cad(
    filepath: str | Path,
    num_sectors: int,
    show_full_disk: bool = False,
    show_dimensions: bool = True,
    surface_mesh_size: float | None = None,
    off_screen: bool = False,
) -> pyvista.Plotter
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str \| Path` | required | Path to CAD file (.step, .stp, .iges, .igs, .brep) |
| `num_sectors` | `int` | required | Number of sectors in the full annulus |
| `show_full_disk` | `bool` | `False` | Show all sectors assembled into the full 360-degree annulus |
| `show_dimensions` | `bool` | `True` | Add dimension annotations (radii, axial length, mesh size) |
| `surface_mesh_size` | `float \| None` | `None` | Override surface triangulation size (None = auto) |
| `off_screen` | `bool` | `False` | Render off-screen (for testing or batch export) |

**Returns:** `pyvista.Plotter` instance.

When `show_full_disk=False`, the single sector is shown. When
`show_full_disk=True`, all sectors are replicated around the Z-axis to display
the complete annular geometry. Dimension annotations include inner/outer radii,
axial length, and recommended mesh size.

**Example:**

```python
import turbomodal as tm

# Preview single sector with dimensions
tm.plot_cad("blade_sector.step", num_sectors=36).show()

# Preview full disk
tm.plot_cad("blade_sector.step", num_sectors=36, show_full_disk=True).show()
```

---

## Mesh Visualization

### plot_mesh

Plot the sector mesh with boundary and node-set highlights.

```python
def plot_mesh(
    mesh: Mesh,
    show_boundaries: bool = True,
    show_node_sets: bool = True,
    off_screen: bool = False,
) -> pyvista.Plotter
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Mesh object |
| `show_boundaries` | `bool` | `True` | Highlight left/right cyclic boundary nodes |
| `show_node_sets` | `bool` | `True` | Highlight hub and other named node sets |
| `off_screen` | `bool` | `False` | Render off-screen (useful for testing or batch export) |

**Returns:** `pyvista.Plotter` instance.

The mesh is rendered as a semi-transparent TET10 wireframe with:
- Left boundary nodes in blue
- Right boundary nodes in red
- Other node sets in green, orange, cyan, magenta, yellow (cycling)

**Example:**

```python
import turbomodal as tm

mesh = tm.load_mesh("sector.msh", num_sectors=24)
plotter = tm.plot_mesh(mesh)
plotter.show()
```

### plot_full_mesh

Plot the full 360-degree mesh without requiring a `ModalResult`. Replicates
the single sector mesh around the Z-axis to show the complete annular geometry.

```python
def plot_full_mesh(
    mesh: Mesh,
    off_screen: bool = False,
) -> pyvista.Plotter
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Single-sector mesh with `num_sectors` set |
| `off_screen` | `bool` | `False` | Render off-screen |

**Returns:** `pyvista.Plotter` showing the full annulus mesh.

Unlike `plot_mode(..., full_annulus=True)`, this function does not require solved mode shapes --
it simply replicates the undeformed sector geometry around the axis of symmetry.

**Example:**

```python
mesh = tm.load_mesh("sector.msh", num_sectors=24)
tm.plot_full_mesh(mesh).show()
```

---

## Mode Shape Visualization

### plot_mode

Plot, animate, or reconstruct the full-annulus mode shape. This single
function replaces the former `plot_full_annulus` and `animate_mode` helpers.

```python
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
) -> pyvista.Plotter
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Single-sector mesh |
| `result` | `ModalResult` | required | Modal result from solver |
| `mode_index` | `int` | `0` | Which mode to display (0-based) |
| `scale` | `float` | `1.0` | Displacement amplification factor |
| `component` | `str` | `"magnitude"` | Scalar field: `"magnitude"`, `"x"`, `"y"`, `"z"`, `"real"`, `"imag"` (single-sector static only) |
| `off_screen` | `bool` | `False` | Render off-screen |
| `full_annulus` | `bool` | `False` | Reconstruct all sectors and display full 360-degree mode shape |
| `animate` | `bool` | `False` | Animate mode shape oscillation: `u(t) = Re(mode * exp(i*2*pi*t/T))` |
| `n_frames` | `int` | `60` | Number of animation frames (when `animate=True`) |
| `filename` | `str \| None` | `None` | Save animation as GIF (when `animate=True`) |

**Returns:** `pyvista.Plotter`

For full-annulus reconstruction, the displacement at harmonic index k,
sector s is: `u_s = Re(u_sector * exp(i * k * s * 2*pi/N))`

**Examples:**

```python
results = tm.solve(mesh, mat, rpm=10000, num_modes=10)

# Single-sector mode shape
tm.plot_mode(mesh, results[3], mode_index=0, scale=50.0).show()

# Full 360-degree annulus
tm.plot_mode(mesh, results[3], mode_index=0, scale=50.0, full_annulus=True).show()

# Animated oscillation
tm.plot_mode(mesh, results[3], mode_index=0, scale=50.0, animate=True)

# Save animation as GIF
tm.plot_mode(mesh, results[3], mode_index=0, scale=50.0,
             animate=True, filename="nd3_mode0.gif")
```

---

## Campbell Diagram

### plot_campbell

Plot a Campbell diagram from RPM sweep results with MAC-based mode tracking.

```python
def plot_campbell(
    results: list[list[ModalResult]],
    engine_orders: Sequence[int] | None = None,
    max_freq: float | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list[list[ModalResult]]` | required | Output from `rpm_sweep()`: `results[rpm_idx][harmonic_idx]` |
| `engine_orders` | `Sequence[int] \| None` | `None` | Engine order lines to overlay (e.g. `[1, 2, 24]`) |
| `max_freq` | `float \| None` | `None` | Maximum frequency for y-axis (None = auto) |
| `figsize` | `tuple[float, float]` | `(12, 8)` | Matplotlib figure size |
| `confidence_bands` | `dict \| None` | `None` | Dict with `"upper"` and `"lower"` keys mapping `(nd, track_id)` to frequency arrays for uncertainty bands |
| `crossing_markers` | `bool` | `False` | Overlay markers at engine-order crossing points |

**Returns:** `matplotlib.figure.Figure`

**Mode tracking:** The function uses MAC (Modal Assurance Criterion) to track
modes across RPM points, ensuring smooth frequency lines even through mode
crossings and veerings. A greedy assignment is used to match modes between
consecutive RPM points based on the highest MAC value.

**Visual conventions:**
- Each nodal diameter gets a distinct color (tab10/tab20/turbo colormap).
- Solid lines = forward whirl (FW), dashed lines = backward whirl (BW).
- Engine order lines are drawn as dotted black lines with `EO=<n>` labels.
- Legend includes both ND labels and FW/BW indicators.

**Example:**

```python
import numpy as np
import turbomodal as tm

rpms = np.linspace(0, 15000, 30)
sweep = tm.rpm_sweep(mesh, mat, rpms, num_modes=10, verbose=1)
fig = tm.plot_campbell(sweep, engine_orders=[1, 2, 24, 48], max_freq=5000)
fig.savefig("campbell.png", dpi=150)
```

**Expected output:** A frequency-vs-RPM plot with colored mode family lines,
engine order crossing lines, and a legend.

---

## ZZENF (Zig-Zag) Diagram

### plot_zzenf

Plot a ZZENF interference diagram at a single RPM.

```python
def plot_zzenf(
    results_at_rpm: list[ModalResult],
    num_sectors: int,
    max_freq: float | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_at_rpm` | `list[ModalResult]` | required | One ModalResult per harmonic index at a single RPM |
| `num_sectors` | `int` | required | Number of sectors in the full annulus |
| `max_freq` | `float \| None` | `None` | Max frequency for y-axis |
| `figsize` | `tuple[float, float]` | `(10, 8)` | Figure size |
| `confidence_bands` | `dict \| None` | `None` | Confidence band data for uncertainty visualization |
| `crossing_markers` | `bool` | `False` | Mark engine-order crossings |

**Returns:** `matplotlib.figure.Figure`

The diagram plots frequency vs. nodal diameter with mode aliasing folded
at N/2. Nodal diameters greater than N/2 are folded back as N - ND.

**Visual conventions:**
- Up triangles (blue) = forward whirl
- Down triangles (red) = backward whirl
- Circles (gray) = standing wave
- X-axis ticks from 0 to N/2

**Example:**

```python
# Single RPM solution
results_10k = tm.solve(mesh, mat, rpm=10000, num_modes=10)
fig = tm.plot_zzenf(results_10k, num_sectors=24, max_freq=5000)
fig.savefig("zzenf_10000rpm.png", dpi=150)
```

**Expected output:** A scatter plot of frequency vs. nodal diameter with
triangular markers indicating whirl direction. Title reads
`ZZENF Diagram @ 10000 RPM`.

### _removed_func

Plot a heatmap of per-sensor SHAP_REMOVEDcontributions across mode families.

```python
def _removed_func(
    internal analysis_values: np.ndarray,
    sensor_names: list[str],
    mode_names: list[str],
    figsize: tuple[float, float] = (10, 6),
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `internal analysis_values` | `ndarray` | required | SHAP_REMOVEDvalues from `| `sensor_names` | `list[str]` | required | Sensor labels for y-axis |
| `mode_names` | `list[str]` | required | Mode family labels for x-axis |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Figure size |

**Returns:** `matplotlib.figure.Figure` with a color-mapped heatmap
showing which sensors contribute most to identifying each mode family.

---

## Boundary Condition Editor

### bc_editor

Interactive 3D boundary condition editor. Opens a PyVista viewer where you
position cutting planes to define boundary condition node groups. Surface nodes
on the positive side of the plane **and within the selection radius** are
highlighted in real-time.

```python
def bc_editor(
    mesh: Mesh,
) -> list[BoundaryCondition]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Mesh object with cyclic boundaries identified |

**Returns:** `list[BoundaryCondition]` — one entry per accepted BC, each with
`node_ids` pre-populated from the visual selection.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `T` | Cycle BC type: FIXED → DISPLACEMENT → FRICTIONLESS |
| `X` / `Y` / `Z` | Toggle constrained DOF component (DISPLACEMENT mode only) |
| `1` / `2` / `3` | Snap plane normal to +X / +Y / +Z axis |
| `4` / `5` / `6` | Rotate normal +90° around X / Y / Z |
| `7` / `8` / `9` | Rotate normal −90° around X / Y / Z |
| `F` | Flip normal (180°) |
| `Enter` | Accept current BC, start a new one |
| `Backspace` | Undo last accepted BC |
| `Q` | Finish and return all accepted BCs |

**Sliders (right side of viewer):**

| Slider | Description |
|--------|-------------|
| Selection Radius | Limit selection to nodes within this radius of origin |
| Origin X / Y / Z | Translate the cutting plane origin |

The plane can also be dragged interactively with the mouse via the built-in
PyVista plane widget. The HUD displays the current BC type, number of selected
nodes, and total accepted BCs.

**Example:**

```python
import turbomodal as tm

mesh = tm.load_mesh("sector.msh", num_sectors=24)

# Open interactive editor
bcs = tm.bc_editor(mesh)

# Each returned BC has node_ids pre-populated
print(bcs[0].node_ids[:5])  # e.g. [42, 43, 44, 107, 108]

# Pass directly to solver
results = tm.solve(mesh, mat, rpm=10000, boundary_conditions=bcs)
```

### plot_boundary_conditions

Visualize the selected node groups for a list of boundary conditions on the
mesh. Each BC is shown as a colored point cloud with a legend entry.

```python
def plot_boundary_conditions(
    mesh: Mesh,
    bcs: list[BoundaryCondition],
    off_screen: bool = False,
) -> pyvista.Plotter
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Mesh object |
| `bcs` | `list[BoundaryCondition]` | required | Boundary conditions to visualize |
| `off_screen` | `bool` | `False` | Render off-screen (for testing or batch export) |

**Returns:** `pyvista.Plotter` with the mesh shown as a translucent wireframe
and each BC's selected nodes rendered as colored spheres. A legend maps colors
to BC names, types, and node counts.

**Example:**

```python
import turbomodal as tm

mesh = tm.load_mesh("sector.msh", num_sectors=24)
bcs = tm.bc_editor(mesh)

# Review BCs before solving
tm.plot_boundary_conditions(mesh, bcs)
```

---

## Internal Utilities

### MAC (Modal Assurance Criterion)

The internal function `_mac(phi_a, phi_b)` computes the MAC value between two
complex mode-shape vectors:

    MAC = |phi_a^H * phi_b|^2 / (|phi_a|^2 * |phi_b|^2)

This is used by `plot_campbell` for cross-RPM mode tracking but is not part of
the public API.

### Mode Tracking

The internal function `_track_modes_for_harmonic(results, nd)` uses MAC-based
greedy assignment to track modes of a given nodal diameter across RPM sweep
points. It returns a permutation list `perm[rpm_idx][track_id] = mode_idx`.

---

## See also

- [Core API](core.md) -- `solve()`, `rpm_sweep()`, `campbell_data()`
- [Signals API](signals.md) -- Virtual sensor arrays
- [Data API](data.md) -- Parametric sweep with HDF5 export
- [ML API](ml.md) -- Machine learning pipeline
- [Optimization API](optimization.md) -- Sensor placement optimization

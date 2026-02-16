# CAD Geometry Visualization Plan

## Context

Currently `load_cad()` imports CAD geometry, meshes it, and returns a `Mesh` object all in one shot. Users have no way to preview the geometry, check dimensions, or see the recommended mesh size before committing to a potentially slow volumetric mesh. There's also no way to view the full annulus mesh without first solving for mode shapes.

This plan adds pre-mesh CAD visualization with dimensional annotations, and post-mesh full-disk viewing without requiring a modal solve.

## New Public API (4 additions)

### 1. `CadInfo` dataclass — `io.py`

```python
@dataclass
class CadInfo:
    filepath: str
    num_sectors: int
    sector_angle_deg: float
    bounding_box: dict          # xmin/ymin/zmin/xmax/ymax/zmax
    inner_radius: float
    outer_radius: float
    axial_length: float
    radial_span: float
    volume: float               # single sector volume
    surface_area: float
    characteristic_length: float
    recommended_mesh_size: float
    num_surfaces: int
    num_volumes: int
```

### 2. `inspect_cad(filepath, num_sectors, verbosity=0) -> CadInfo` — `io.py`

Lightweight: imports CAD via gmsh OCC, queries geometry metadata, no meshing at all. Uses `gmsh.model.getBoundingBox()`, `gmsh.model.occ.getMass()`, and samples parametric surface points for inner/outer radius.

### 3. `plot_cad(filepath, num_sectors, show_full_disk=False, show_dimensions=True, surface_mesh_size=None, off_screen=False) -> pv.Plotter` — `viz.py`

Generates a fast 2D surface triangulation via `gmsh.model.mesh.generate(2)` (milliseconds, not minutes), renders in PyVista. Optionally replicates sectors for full 360-degree view with alternating colors. Annotates with dimensions (radii, axial length, volume, recommended mesh size) when `show_dimensions=True`.

### 4. `plot_full_mesh(mesh, off_screen=False) -> pv.Plotter` — `viz.py`

Replicates the TET10 volumetric mesh around Z-axis for full annulus view. Same rotation math as existing `plot_full_annulus()` but without modal displacement — pure geometry.

## Internal Helpers (2 additions)

### `_extract_surface_tessellation(filepath, num_sectors, surface_mesh_size=None, verbosity=0)` — `io.py`

Returns `(nodes, triangles, CadInfo)`. Steps:
1. `gmsh.initialize()` / `gmsh.model.occ.importShapes()` / synchronize
2. Query bounding box, volume, surface area, surface count
3. Compute inner/outer radius from surface node radial distances
4. Compute `recommended_mesh_size = characteristic_length / 20`
5. `gmsh.model.mesh.generate(2)` — fast surface-only triangulation
6. Extract nodes `(N,3)` and triangles `(M,3)` from gmsh
7. `gmsh.finalize()` and return

### `_replicate_sectors(nodes, cells, n_sectors, celltypes)` — `viz.py`

Shared Z-axis rotation loop extracted from the existing `plot_full_annulus()` pattern. Returns `(all_nodes, all_cells, all_celltypes)`.

```python
for s in range(n_sectors):
    theta = s * 2 * pi / n_sectors
    R = rotation_matrix_z(theta)
    rotated_nodes = nodes @ R.T
    offset_cells = cells.copy()
    offset_cells[:, 1:] += s * n_nodes
```

## Recommended Mesh Size Heuristic

```python
radial_span = outer_radius - inner_radius
axial_length = zmax - zmin
characteristic_length = min(radial_span, axial_length)
if characteristic_length < 1e-10:
    characteristic_length = max(radial_span, axial_length)
recommended_mesh_size = characteristic_length / 20
```

Division by 20 gives 3-5 elements through blade/disk thickness — standard turbomachinery FEA starting point.

## Dimension Annotations

When `show_dimensions=True`, `plot_cad()` adds:
- Text overlay (upper-left): filename, sector count/angle, inner/outer radius, axial length, volume, recommended mesh size
- Display in mm if values suggest SI meters (< 1.0 range)
- Bounding box outline via `plotter.add_bounding_box()`
- Axes widget via `plotter.add_axes()`

## Files to Modify

| File | Changes |
|------|---------|
| `python/turbomodal/io.py` | Add `CadInfo`, `inspect_cad()`, `_extract_surface_tessellation()` |
| `python/turbomodal/viz.py` | Add `plot_cad()`, `plot_full_mesh()`, `_replicate_sectors()` |
| `python/turbomodal/__init__.py` | Export `inspect_cad`, `CadInfo`, `plot_cad`, `plot_full_mesh` |
| `python/tests/test_viz.py` | Add `TestPlotCad`, `TestPlotFullMesh` |
| `python/tests/test_io.py` | Add `TestInspectCad`, `TestExtractSurfaceTessellation` |
| `python/tests/conftest.py` | Add `test_step_path` fixture pointing to `test_data/test_sector.step` |

## Implementation Order

1. `CadInfo` dataclass in `io.py`
2. `inspect_cad()` in `io.py`
3. `_extract_surface_tessellation()` in `io.py`
4. `_replicate_sectors()` helper in `viz.py`
5. `plot_full_mesh()` in `viz.py`
6. `plot_cad()` in `viz.py` (imports from `io.py` inside function body to avoid circular imports)
7. `__init__.py` exports
8. Tests

## Verification

1. `pytest python/tests/test_io.py -v -k "inspect_cad or surface_tessellation"` — metadata and surface extraction
2. `pytest python/tests/test_viz.py -v -k "plot_cad or plot_full_mesh"` — visualization functions
3. `pytest python/tests/ -v --tb=short` — full suite, no regressions (171+ tests pass)
4. Manual: `python -c "from turbomodal import plot_cad; plot_cad('tests/test_data/test_sector.step', 24)"` — visual check
5. Manual: `python -c "from turbomodal import plot_cad; plot_cad('tests/test_data/test_sector.step', 24, show_full_disk=True)"` — full disk check

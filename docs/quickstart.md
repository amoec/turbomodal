# Quick Start Tutorial

## Overview

End-to-end walkthrough: load a mesh, define boundary conditions, solve
cyclic symmetry FEA, perform an RPM sweep, and visualize results with a
Campbell diagram.

---

## Step 1: Inspect and Load Geometry

### 1a: Preview CAD geometry (optional)

Before meshing, inspect the CAD file to check dimensions and get a
recommended mesh size:

```python
import turbomodal as tm

# Inspect dimensions without meshing
info = tm.inspect_cad("blade_sector.step", num_sectors=36)
print(f"Inner radius: {info.inner_radius*1000:.1f} mm")
print(f"Outer radius: {info.outer_radius*1000:.1f} mm")
print(f"Axial length: {info.axial_length*1000:.1f} mm")
print(f"Recommended mesh size: {info.recommended_mesh_size*1000:.2f} mm")

# Visualize the sector geometry
tm.plot_cad("blade_sector.step", num_sectors=36).show()

# Or preview the full disk (all sectors assembled)
tm.plot_cad("blade_sector.step", num_sectors=36, show_full_disk=True).show()
```

### 1b: Load and mesh

```python
# From a mesh file (.msh, .bdf/.nas, .inp, .vtk/.vtu, .cgns, .med, .xdmf)
mesh = tm.load_mesh("blade_sector.msh", num_sectors=36)

# Or from CAD geometry (.step, .iges, .brep)
mesh = tm.load_cad(
    "blade_sector.step",
    num_sectors=36,
    mesh_size=0.005,
    order=2,
    auto_detect_boundaries=True,
)

print(f"Nodes: {mesh.num_nodes()}, Elements: {mesh.num_elements()}")
tm.plot_mesh(mesh, show_boundaries=True).show()

# View the full annulus mesh (no solve needed)
tm.plot_full_mesh(mesh).show()
```

Turbomodal requires quadratic tetrahedra (TET10). The `load_cad` function
uses gmsh with the OpenCASCADE kernel and automatically detects cyclic
boundary surfaces (left/right cuts and hub).

---

## Step 2: Define Boundary Conditions

By default, `solve()` applies a fixed hub constraint. For custom boundary
conditions (e.g. partial constraints, frictionless surfaces, displacement
locks on specific DOFs), use the interactive BC editor or define them
programmatically.

### 2a: Interactive BC editor

```python
bcs = tm.bc_editor(mesh)
```

This opens a 3D viewer where you position a cutting plane to select surface
nodes.  The editor provides:

- **Real-time node highlighting** — yellow spheres show selected nodes as
  you move the plane
- **Selection radius** — a slider to limit selection to a bounded circular
  region instead of the full half-space
- **BC type cycling** — press `T` to cycle through FIXED, DISPLACEMENT,
  and FRICTIONLESS
- **DOF toggles** — in DISPLACEMENT mode, press `X`/`Y`/`Z` to toggle
  which displacement components are constrained
- **Plane orientation** — press `1`/`2`/`3` to snap the normal to an axis,
  `4`-`9` for ±90° rotations, `F` to flip
- **Multiple BCs** — press `Enter` to accept and start the next BC,
  `Backspace` to undo

Returns a list of `BoundaryCondition` objects ready for `solve()`.

### 2b: Programmatic BCs

```python
import numpy as np

bcs = [
    tm.BoundaryCondition(
        name="hub_fixed",
        type="fixed",
        plane_point=np.array([0.0, 0.0, -0.005]),
        plane_normal=np.array([0.0, 0.0, -1.0]),
    ),
    tm.BoundaryCondition(
        name="tip_axial",
        type="displacement",
        plane_point=np.array([0.0, 0.0, 0.01]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        constrained_components=(False, False, True),  # only uz locked
    ),
]
```

### 2c: Visualize BCs before solving

```python
tm.plot_boundary_conditions(mesh, bcs)
```

---

## Step 3: Solve Cyclic Symmetry FEA

```python
mat = tm.Material(E=200e9, nu=0.3, rho=7800)

# Default hub constraint
results = tm.solve(mesh, mat, rpm=3000, num_modes=10, verbose=1)

# Or with custom boundary conditions from Step 2
results = tm.solve(mesh, mat, rpm=3000, num_modes=10,
                   boundary_conditions=bcs, verbose=1)

for r in results:
    freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:3])
    print(f"  ND={r.harmonic_index}: [{freqs}] Hz")
```

`solve` returns a list of `ModalResult` objects (one per harmonic index,
ND 0 through N/2). Each provides `frequencies`, `mode_shapes` (complex,
shape n_dof x n_modes), `whirl_direction` (+1 FW, -1 BW, 0 standing),
and `harmonic_index`.

When `boundary_conditions` is provided, it overrides the default hub
constraint. Each BC selects surface nodes on the positive side of a
cutting plane and applies the specified constraint type. If a BC was
created with `bc_editor`, the pre-selected node IDs are used directly.

For fluid coupling, pass a `FluidConfig`:

```python
fluid = tm.FluidConfig()
fluid.fluid_type = tm.FluidType.LIQUID
fluid.density = 1000.0
results = tm.solve(mesh, mat, rpm=3000, num_modes=10, fluid=fluid)
```

Visualize mode shapes:

```python
# Single sector
tm.plot_mode(mesh, results[2], mode_index=0, scale=0.001).show()

# Full 360-degree annulus
tm.plot_mode(mesh, results[2], mode_index=0, scale=0.001, full_annulus=True).show()

# Animated oscillation
tm.plot_mode(mesh, results[2], mode_index=0, scale=0.001, animate=True)
```

---

## Step 4: RPM Sweep and Campbell Diagram

```python
import numpy as np

rpm_values = np.linspace(0, 15000, 20)
sweep_results = tm.rpm_sweep(mesh, mat, rpm_values, num_modes=10, verbose=1)

campbell = tm.campbell_data(sweep_results)
# campbell['frequencies'] shape: (N_rpm, N_harmonics, N_modes)

style = tm.DiagramStyle(eo_linewidth=2.0, family_marker_size=8)
fig = tm.plot_campbell(sweep_results, engine_orders=[1, 2, 36],
                       stator_vanes=44, style=style)
fig = tm.plot_zzenf(sweep_results[-1], num_sectors=36,
                    engine_orders=[1, 2, 36], stator_vanes=44,
                    crossing_markers=True, style=style)

# Compare against ground truth (rows=NDs, cols=modes)
gt = np.array([[...], [...], ...])  # shape (N/2+1, n_modes)
diag = tm.diagnose_frequencies(sweep_results, gt, num_sectors=36)
print(diag["summary"])
```

---

## Next Steps

- **API Reference** -- See `docs/api/` for detailed class and function
  documentation.
- **Examples** -- `examples/python_example.py` demonstrates mesh loading,
  solving, and visualization.
- **Tests** -- Run `pytest python/tests/ -v` for the Python test suite.
  Run `pytest -m validation` for analytical benchmark tests
  (Leissa plate, Kwak added mass, FMM, SDOF FRF).
- **C++ Tests** -- Build with `BUILD_TESTS=ON` and run `ctest` for 15
  test suites (~25 seconds). Add `BUILD_VALIDATION_TESTS=ON` for slow
  validation tests including Leissa, Kwak, and Coriolis benchmarks.

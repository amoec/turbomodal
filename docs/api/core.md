# turbomodal Core API Reference

This document covers the core modules of turbomodal: mesh I/O (`turbomodal.io`),
solver interface (`turbomodal.solver`), and visualization (`turbomodal.viz`).
The underlying C++ classes (`Material`, `Mesh`, `ModalResult`, etc.) are exposed
via pybind11 through `turbomodal._core`.

---

## C++ Bound Classes

The following classes are imported from the compiled `turbomodal._core` extension
module and re-exported at the top level of the `turbomodal` package.

### Material

Isotropic material properties for FEA.

```python
turbomodal.Material()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `E` | `float` | Young's modulus (Pa) |
| `nu` | `float` | Poisson's ratio |
| `rho` | `float` | Density (kg/m^3) |

**Methods:**

- `at_temperature(temperature: float) -> Material` -- Returns a new `Material`
  with stiffness adjusted for the given temperature (Kelvin). Uses a linear
  temperature-dependent modulus when `E_slope` is set.

### NodeSet

Named group of mesh node indices.

```python
turbomodal.NodeSet()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name (e.g. `"left_boundary"`, `"hub"`) |
| `node_ids` | `list[int]` | Sorted list of 0-based node indices |

### Mesh

Quadratic tetrahedral (TET10) finite element mesh for one cyclic sector.

```python
turbomodal.Mesh()
```

**Properties / Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `num_sectors` | `int` | Number of sectors in the full annulus |
| `nodes` | `ndarray (n_nodes, 3)` | Node coordinates (float64) |
| `elements` | `ndarray (n_elem, 10)` | TET10 connectivity (int32) |
| `node_sets` | `list[NodeSet]` | Named node groups |
| `left_boundary` | `list[int]` | Node IDs on the left cyclic boundary |
| `right_boundary` | `list[int]` | Node IDs on the right cyclic boundary |

**Methods:**

- `num_nodes() -> int` -- Total number of nodes in the sector mesh.
- `num_elements() -> int` -- Total number of TET10 elements.
- `load_from_arrays(coords, connectivity, node_sets, num_sectors)` -- Build mesh
  from NumPy arrays.
  - `coords`: `ndarray (n_nodes, 3)` float64
  - `connectivity`: `ndarray (n_elem, 10)` int32
  - `node_sets`: `list[NodeSet]`
  - `num_sectors`: `int`
- `load_from_gmsh(filepath: str)` -- Load a gmsh `.msh` file via the native C++
  reader.
- `identify_cyclic_boundaries()` -- Automatically detect left and right cyclic
  boundary surfaces.
- `match_boundary_nodes()` -- Pair left and right boundary nodes for cyclic
  constraint assembly.

### SolverConfig

Configuration for the cyclic symmetry eigensolver.

```python
turbomodal.SolverConfig()
```

### SolverStatus

Enumeration of solver exit statuses.

### ModalResult

Container for modal analysis results at a single harmonic index.

```python
turbomodal.ModalResult
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `harmonic_index` | `int` | Nodal diameter (0 to N/2) |
| `rpm` | `float` | Rotational speed at which this was solved |
| `frequencies` | `ndarray (n_modes,)` | Natural frequencies in Hz |
| `mode_shapes` | `ndarray (n_dof, n_modes)` | Complex mode shape matrix |
| `whirl_direction` | `ndarray (n_modes,)` | +1 = forward, -1 = backward, 0 = standing |

### FluidConfig / FluidType

Configuration for fluid-structure interaction coupling.

```python
turbomodal.FluidConfig()
```

### CyclicSymmetrySolver

The main solver class.

```python
turbomodal.CyclicSymmetrySolver(mesh, material, fluid)
```

**Parameters:**

- `mesh` : `Mesh` -- Sector mesh with identified cyclic boundaries.
- `material` : `Material` -- Material properties.
- `fluid` : `FluidConfig` -- Fluid coupling (use default `FluidConfig()` for dry
  analysis).

**Methods:**

- `solve_at_rpm(rpm: float, num_modes: int) -> list[ModalResult]` -- Solve the
  eigenvalue problem at the given RPM. Returns one `ModalResult` per harmonic
  index (0 to N/2).

### Additional C++ Classes

- `DampingConfig` / `DampingType` -- Damping models (structural, Rayleigh, modal).
- `ForcedResponseConfig` / `ForcedResponseResult` / `ForcedResponseSolver` --
  Harmonic forced response analysis.
- `ExcitationType` -- Excitation type enumeration.
- `MistuningConfig` / `MistuningResult` / `FMMSolver` -- Fundamental Mistuning
  Model for blade-to-blade frequency deviations.
- `AddedMassModel` / `ModalSolver` -- Added mass and general modal solver.
- `ModeIdentification` / `GroundTruthLabel` -- Mode classification utilities.
- `identify_nodal_circles(...)`, `classify_mode_family(...)`,
  `identify_modes(...)` -- C++ mode identification functions.

---

## turbomodal.io -- Mesh Import

### load_cad

Load a CAD file, mesh it with gmsh, and return a turbomodal `Mesh`.

```python
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
) -> Mesh
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str \| Path` | required | Path to CAD file (.step, .stp, .iges, .igs, .brep, .stl) |
| `num_sectors` | `int` | required | Number of sectors in the full annulus |
| `mesh_size` | `float \| None` | `None` | Characteristic mesh element size (None = auto) |
| `order` | `int` | `2` | Element order (2 = quadratic TET10) |
| `left_boundary_name` | `str` | `"left_boundary"` | Physical group name for left cyclic boundary |
| `right_boundary_name` | `str` | `"right_boundary"` | Physical group name for right cyclic boundary |
| `hub_name` | `str \| None` | `"hub"` | Physical group name for hub constraint (None = no hub) |
| `auto_detect_boundaries` | `bool` | `True` | Auto-identify cyclic boundary surfaces |
| `verbosity` | `int` | `0` | gmsh verbosity (0=silent, 5=debug) |

**Returns:** `Mesh` -- Ready for cyclic symmetry analysis.

**Raises:** `FileNotFoundError` if path does not exist; `ValueError` for unsupported formats.

**Example:**

```python
import turbomodal as tm

mesh = tm.load_cad("blade_sector.step", num_sectors=24, mesh_size=0.005)
print(f"Nodes: {mesh.num_nodes()}, Elements: {mesh.num_elements()}")
```

### load_mesh

Load a pre-meshed file (NASTRAN, Abaqus, VTK, gmsh MSH, etc.).

```python
def load_mesh(
    filepath: str | Path,
    num_sectors: int,
    file_format: str | None = None,
) -> Mesh
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str \| Path` | required | Path to mesh file |
| `num_sectors` | `int` | required | Number of sectors in the full annulus |
| `file_format` | `str \| None` | `None` | Override format detection (e.g. `"nastran"`, `"abaqus"`) |

**Supported formats:**

- `.msh` -- gmsh MSH 2.x (native C++ loader)
- `.bdf` / `.nas` -- NASTRAN
- `.inp` -- Abaqus
- `.vtk` / `.vtu` -- VTK
- `.cgns` -- CGNS
- `.med` -- Salome MED
- `.xdmf` -- XDMF

**Returns:** `Mesh`

**Raises:** `FileNotFoundError`, `RuntimeError` if no TET10 elements found.

**Example:**

```python
mesh = tm.load_mesh("sector.bdf", num_sectors=36)
```

---

## turbomodal.solver -- Solver Interface

### solve

Solve cyclic symmetry modal analysis at a single RPM.

```python
def solve(
    mesh: Mesh,
    material: Material,
    rpm: float = 0.0,
    num_modes: int = 20,
    fluid: FluidConfig | None = None,
    config: SolverConfig | None = None,
    verbose: int = 0,
) -> list[ModalResult]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Mesh with cyclic boundaries identified |
| `material` | `Material` | required | Material properties |
| `rpm` | `float` | `0.0` | Rotation speed in RPM (0 = stationary) |
| `num_modes` | `int` | `20` | Number of modes per harmonic index |
| `fluid` | `FluidConfig \| None` | `None` | Fluid coupling config (None = dry) |
| `config` | `SolverConfig \| None` | `None` | Solver config (None = defaults) |
| `verbose` | `int` | `0` | 0=silent, 1=progress, 2=detailed |

**Returns:** `list[ModalResult]` -- One per harmonic index (0 to N/2).

**Example:**

```python
mat = tm.Material()
mat.E = 200e9
mat.nu = 0.3
mat.rho = 7800.0

results = tm.solve(mesh, mat, rpm=10000, num_modes=10, verbose=1)
for r in results:
    print(f"ND={r.harmonic_index}: {r.frequencies[:3]} Hz")
```

### rpm_sweep

Solve modal analysis over a range of RPM values.

```python
def rpm_sweep(
    mesh: Mesh,
    material: Material,
    rpm_values: np.ndarray | Sequence[float],
    num_modes: int = 20,
    fluid: FluidConfig | None = None,
    verbose: int = 0,
) -> list[list[ModalResult]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rpm_values` | `ndarray \| Sequence[float]` | required | Array of RPM values |
| *(other parameters same as `solve`)* | | | |

**Returns:** `list[list[ModalResult]]` -- `results[rpm_idx][harmonic_idx]`

### campbell_data

Extract Campbell diagram data from RPM sweep results.

```python
def campbell_data(
    results: list[list[ModalResult]],
) -> dict[str, np.ndarray]
```

**Returns:** Dictionary with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"rpm"` | `(N,)` | RPM values |
| `"frequencies"` | `(N, H, M)` | Frequencies in Hz |
| `"harmonic_index"` | `(H,)` | Harmonic indices |
| `"whirl_direction"` | `(N, H, M)` | +1=FW, -1=BW, 0=standing |

Where N = RPM points, H = harmonics, M = modes per harmonic.

**Example:**

```python
rpms = np.linspace(0, 15000, 20)
sweep = tm.rpm_sweep(mesh, mat, rpms, num_modes=10, verbose=1)
cd = tm.campbell_data(sweep)
print(cd["frequencies"].shape)  # (20, H, 10)
```

---

## See also

- [Signals API](signals.md) -- Sensor array and signal generation
- [Data API](data.md) -- Dataset export and parametric sweeps
- [Analysis API](analysis.md) -- Campbell and ZZENF visualization
- [ML API](ml.md) -- Machine learning pipeline
- [Optimization API](optimization.md) -- Sensor placement optimization

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

### AddedMassModel

Static methods for computing Kwak's Added Virtual Mass Incremental (AVMI)
factors for fluid-coupled vibration of annular plates.

**Methods:**

- `AddedMassModel.kwak_avmi(nodal_diameter, rho_fluid, rho_structure, thickness, radius) -> float`
  -- Compute the AVMI factor for a given nodal diameter.
- `AddedMassModel.frequency_ratio(nodal_diameter, rho_fluid, rho_structure, thickness, radius) -> float`
  -- Compute the wet-to-dry frequency ratio `f_wet / f_dry`.

**Example:**

```python
import turbomodal as tm

ratio = tm.AddedMassModel.frequency_ratio(
    nodal_diameter=3, rho_fluid=1000.0, rho_structure=7800.0,
    thickness=0.01, radius=0.15,
)
print(f"Frequency ratio (wet/dry) at ND=3: {ratio:.4f}")
```

### DampingConfig / DampingType

Configuration for structural damping in forced response analysis.

```python
turbomodal.DampingConfig()
```

**DampingType** enumeration:

| Value | Description |
|-------|-------------|
| `DampingType.NONE` | No damping |
| `DampingType.MODAL` | Per-mode damping ratios |
| `DampingType.RAYLEIGH` | Mass- and stiffness-proportional (alpha, beta) |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | `DampingType` | Damping model to use |
| `modal_damping_ratios` | `ndarray` | Per-mode damping ratios (for `MODAL` type) |
| `rayleigh_alpha` | `float` | Mass-proportional coefficient (for `RAYLEIGH` type) |
| `rayleigh_beta` | `float` | Stiffness-proportional coefficient (for `RAYLEIGH` type) |
| `aero_damping_ratios` | `ndarray` | Aerodynamic damping ratios (additive) |

**Methods:**

- `effective_damping(mode_index: int, omega_r: float) -> float` -- Compute the
  effective damping ratio for a mode, combining structural and aerodynamic
  contributions.

**Example:**

```python
import turbomodal as tm

damping = tm.DampingConfig()
damping.type = tm.DampingType.RAYLEIGH
damping.rayleigh_alpha = 0.5   # mass-proportional
damping.rayleigh_beta = 1e-6   # stiffness-proportional
```

### ForcedResponseConfig / ExcitationType

Configuration for harmonic forced response analysis.

```python
turbomodal.ForcedResponseConfig()
```

**ExcitationType** enumeration:

| Value | Description |
|-------|-------------|
| `ExcitationType.UNIFORM_PRESSURE` | Uniform pressure load on all blade surfaces |
| `ExcitationType.POINT_FORCE` | Concentrated force at a single node |
| `ExcitationType.SPATIAL_DISTRIBUTION` | Arbitrary spatial force distribution |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `engine_order` | `int` | Engine order of the excitation |
| `force_amplitude` | `float` | Force amplitude (N or Pa) |
| `excitation_type` | `ExcitationType` | Type of excitation |
| `force_node_id` | `int` | Node ID for `POINT_FORCE` excitation |
| `force_direction` | `ndarray (3,)` | Direction of point force |
| `force_vector` | `ndarray` | Full force vector for `SPATIAL_DISTRIBUTION` |
| `freq_min` | `float` | Minimum sweep frequency (Hz) |
| `freq_max` | `float` | Maximum sweep frequency (Hz) |
| `num_freq_points` | `int` | Number of points in the frequency sweep |

### ForcedResponseResult

Container for forced response analysis results.

```python
turbomodal.ForcedResponseResult()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `engine_order` | `int` | Engine order of the excitation |
| `rpm` | `float` | Rotational speed (RPM) |
| `natural_frequencies` | `ndarray` | Natural frequencies of participating modes (Hz) |
| `modal_forces` | `ndarray (complex)` | Modal force coefficients |
| `modal_damping_ratios` | `ndarray` | Effective damping ratio per mode |
| `participation_factors` | `ndarray` | Modal participation factors |
| `effective_modal_mass` | `ndarray` | Effective modal mass per mode |
| `sweep_frequencies` | `ndarray` | Frequency sweep points (Hz) |
| `modal_amplitudes` | `ndarray (complex)` | Complex amplitude at each sweep frequency |
| `max_response_amplitude` | `float` | Peak response amplitude |
| `resonance_frequencies` | `ndarray` | Detected resonance frequencies (Hz) |

### ForcedResponseSolver

Solver for modal-superposition-based harmonic forced response.

```python
turbomodal.ForcedResponseSolver(mesh, damping=DampingConfig())
```

**Parameters:**

- `mesh` : `Mesh` -- Sector mesh.
- `damping` : `DampingConfig` -- Damping configuration (default: no damping).

**Methods:**

- `solve(modal_results, rpm, config) -> ForcedResponseResult` -- Compute the
  forced response from modal results.
- `compute_modal_forces(mode_shapes, force_vector) -> ndarray` -- Project a
  physical force vector onto the modal basis.
- `modal_frf(omega, omega_r, Q_r, zeta_r) -> complex` -- Single-DOF modal
  frequency response function.
- `compute_participation_factors(mode_shapes, M, direction) -> ndarray` --
  Modal participation factors along a direction.
- `compute_effective_modal_mass(mode_shapes, M, direction) -> ndarray` --
  Effective modal mass per mode.
- `build_eo_excitation(engine_order, amplitude, type, ...) -> ndarray` --
  Build an engine-order excitation vector.

**Example:**

```python
import turbomodal as tm

damping = tm.DampingConfig()
damping.type = tm.DampingType.MODAL
damping.modal_damping_ratios = np.array([0.01] * 10)

fr_solver = tm.ForcedResponseSolver(mesh, damping)

fr_config = tm.ForcedResponseConfig()
fr_config.engine_order = 24
fr_config.force_amplitude = 100.0
fr_config.freq_min = 100.0
fr_config.freq_max = 5000.0
fr_config.num_freq_points = 500

fr_result = fr_solver.solve(results, rpm=10000, config=fr_config)
print(f"Peak amplitude: {fr_result.max_response_amplitude:.6f}")
print(f"Resonance at: {fr_result.resonance_frequencies} Hz")
```

### MistuningConfig / MistuningResult

Configuration and results for the Fundamental Mistuning Model (FMM).

```python
turbomodal.MistuningConfig()
```

**MistuningConfig attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `blade_frequency_deviations` | `ndarray (N,)` | Per-blade frequency deviation from tuned (fractional) |
| `tuned_frequencies` | `ndarray` | Tuned system natural frequencies (Hz) |
| `mode_family_index` | `int` | Index of the mode family to mistune |

**MistuningResult attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `frequencies` | `ndarray` | Mistuned natural frequencies (Hz) |
| `blade_amplitudes` | `ndarray (complex)` | Per-blade complex amplitudes |
| `amplitude_magnification` | `ndarray` | Per-blade amplitude magnification factor |
| `peak_magnification` | `float` | Maximum amplitude magnification across all blades |
| `localization_ipr` | `float` | Inverse Participation Ratio (localization metric) |

### FMMSolver

Static solver for the Fundamental Mistuning Model.

**Methods:**

- `FMMSolver.solve(num_sectors, tuned_frequencies, blade_frequency_deviations) -> MistuningResult`
  -- Solve the FMM eigenvalue problem.
- `FMMSolver.random_mistuning(num_sectors, sigma, seed=42) -> ndarray` --
  Generate random blade frequency deviations with standard deviation `sigma`.

**Example:**

```python
import turbomodal as tm

# Generate random 2% mistuning for 36 blades
deviations = tm.FMMSolver.random_mistuning(num_sectors=36, sigma=0.02)

# Solve FMM
result = tm.FMMSolver.solve(
    num_sectors=36,
    tuned_frequencies=tuned_freqs,
    blade_frequency_deviations=deviations,
)
print(f"Peak magnification: {result.peak_magnification:.3f}")
print(f"Localization IPR: {result.localization_ipr:.3f}")
```

### ModeIdentification

Result container for C++ mode classification.

```python
turbomodal.ModeIdentification()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `nodal_diameter` | `int` | Identified nodal diameter |
| `nodal_circle` | `int` | Identified nodal circle count |
| `whirl_direction` | `int` | +1=FW, -1=BW, 0=standing |
| `frequency` | `float` | Natural frequency (Hz) |
| `wave_velocity` | `float` | Circumferential wave velocity (m/s) |
| `participation_factor` | `float` | Modal participation factor |
| `family_label` | `str` | Mode family label (`"B"`, `"T"`, or `"A"`) |

### GroundTruthLabel

Extended label container for supervised ML training data.

```python
turbomodal.GroundTruthLabel()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `rpm` | `float` | Operating RPM |
| `temperature` | `float` | Operating temperature (K) |
| `nodal_diameter` | `int` | Nodal diameter |
| `nodal_circle` | `int` | Nodal circle count |
| `whirl_direction` | `int` | +1=FW, -1=BW, 0=standing |
| `frequency` | `float` | Natural frequency (Hz) |
| `wave_velocity` | `float` | Circumferential wave velocity (m/s) |
| `family_label` | `str` | Mode family (`"B"`, `"T"`, `"A"`) |
| `participation_factor` | `float` | Modal participation factor |
| `effective_modal_mass` | `float` | Effective modal mass |
| `amplitude` | `float` | Vibration amplitude |
| `damping_ratio` | `float` | Effective damping ratio |
| `amplitude_magnification` | `float` | Mistuning magnification factor |
| `is_localized` | `bool` | Whether mode is spatially localized |
| `condition_id` | `int` | Index into parametric sweep conditions |
| `mode_index` | `int` | Mode index within the harmonic |

### Mode Identification Functions

Free functions for C++ mode classification.

- `identify_nodal_circles(mode_shape, mesh) -> int` -- Count nodal circles
  from the displacement field of a single mode shape vector.
- `classify_mode_family(mode_shape, mesh) -> str` -- Classify mode as
  bending (`"B"`), torsion (`"T"`), or axial (`"A"`).
- `identify_modes(result, mesh, characteristic_radius=0.0) -> list[ModeIdentification]`
  -- Identify all modes in a `ModalResult`, returning one `ModeIdentification`
  per mode.

**Example:**

```python
import turbomodal as tm

results = tm.solve(mesh, mat, rpm=10000, num_modes=5)
modes = tm.identify_modes(results[3], mesh, characteristic_radius=0.15)
for m in modes:
    print(f"ND={m.nodal_diameter} NC={m.nodal_circle} {m.family_label} "
          f"f={m.frequency:.1f} Hz")
```

---

## turbomodal.io -- Mesh Import

### inspect_cad

Inspect a CAD file and return geometry metadata without meshing. This is a
lightweight operation that imports the CAD geometry, computes bounding box,
dimensions, and a recommended mesh size, but does **not** generate any mesh.

```python
def inspect_cad(
    filepath: str | Path,
    num_sectors: int,
    verbosity: int = 0,
) -> CadInfo
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str \| Path` | required | Path to CAD file (.step, .stp, .iges, .igs, .brep) |
| `num_sectors` | `int` | required | Number of sectors in the full annulus |
| `verbosity` | `int` | `0` | gmsh verbosity level (0=silent, 5=debug) |

**Returns:** `CadInfo` -- Geometry metadata dataclass.

**Raises:** `FileNotFoundError` if path does not exist; `ValueError` for unsupported formats.

**Example:**

```python
import turbomodal as tm

info = tm.inspect_cad("blade_sector.step", num_sectors=36)
print(f"Inner radius: {info.inner_radius*1000:.1f} mm")
print(f"Outer radius: {info.outer_radius*1000:.1f} mm")
print(f"Axial length: {info.axial_length*1000:.1f} mm")
print(f"Recommended mesh size: {info.recommended_mesh_size*1000:.2f} mm")
```

### CadInfo

Dataclass returned by `inspect_cad()` containing geometry metadata.

```python
@dataclass
class CadInfo:
    filepath: str
    num_sectors: int
    sector_angle_deg: float
    bounding_box: dict          # {"xmin", "ymin", "zmin", "xmax", "ymax", "zmax"}
    inner_radius: float         # metres
    outer_radius: float         # metres
    axial_length: float         # metres
    radial_span: float          # metres
    volume: float               # m^3
    surface_area: float         # m^2
    characteristic_length: float  # metres
    recommended_mesh_size: float  # metres (characteristic_length / 20)
    num_surfaces: int
    num_volumes: int
```

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

### BoundaryCondition

Specification for a boundary condition applied via a cutting plane.

```python
@dataclass
class BoundaryCondition:
    name: str
    type: str                    # "fixed", "displacement", "frictionless"
    plane_point: np.ndarray      # (3,) point on the cutting plane
    plane_normal: np.ndarray     # (3,) outward normal
    constrained_components: tuple[bool, bool, bool] = (True, True, True)
    tolerance: float = 1e-6
    node_ids: list[int] | None = None
    selection_radius: float | None = None
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Descriptive name for this constraint group |
| `type` | `str` | required | `"fixed"`, `"displacement"`, or `"frictionless"` |
| `plane_point` | `ndarray (3,)` | required | Point on the cutting plane |
| `plane_normal` | `ndarray (3,)` | required | Outward normal -- surface nodes on the positive side are selected |
| `constrained_components` | `tuple[bool,bool,bool]` | `(True,True,True)` | For `"displacement"` type, which DOF components are locked (x,y,z). Ignored for other types. |
| `tolerance` | `float` | `1e-6` | Distance tolerance for plane selection |
| `node_ids` | `list[int] \| None` | `None` | Pre-selected node IDs (from `bc_editor`). When provided, the solver uses these directly instead of calling `select_nodes_by_plane`. |
| `selection_radius` | `float \| None` | `None` | Radius used during interactive selection (stored for reproducibility). No effect at solve time. |

**BC types:**

- **`"fixed"`** -- All 3 translational DOFs are constrained (ux=uy=uz=0).
- **`"displacement"`** -- Only the DOFs indicated by `constrained_components`
  are locked. E.g. `(False, False, True)` locks only uz.
- **`"frictionless"`** -- The DOF component aligned with the local surface
  normal is constrained (zero normal displacement, free tangential sliding).

**Example:**

```python
import numpy as np
import turbomodal as tm

# Fixed hub
bc_hub = tm.BoundaryCondition(
    name="hub", type="fixed",
    plane_point=np.array([0, 0, -0.005]),
    plane_normal=np.array([0, 0, -1.0]),
)

# Axial-only displacement lock at the tip
bc_tip = tm.BoundaryCondition(
    name="tip_uz", type="displacement",
    plane_point=np.array([0, 0, 0.01]),
    plane_normal=np.array([0, 0, 1.0]),
    constrained_components=(False, False, True),
)
```

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
    harmonic_indices: list[int] | None = None,
    hub_constraint: str = "fixed",
    boundary_conditions: list[BoundaryCondition] | None = None,
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
| `harmonic_indices` | `list[int] \| None` | `None` | Specific harmonic indices to solve (None = all 0..N/2) |
| `hub_constraint` | `str` | `"fixed"` | `"fixed"` or `"free"`. Ignored when `boundary_conditions` is provided. |
| `boundary_conditions` | `list[BoundaryCondition] \| None` | `None` | Custom BCs. Overrides `hub_constraint` when provided. |

**Returns:** `list[ModalResult]` -- One per harmonic index (0 to N/2).

When `boundary_conditions` is provided, each BC selects surface nodes
on the positive side of a cutting plane and applies the specified
constraint type. If a BC was created with `bc_editor`, the pre-selected
`node_ids` are used directly (no plane selection at solve time).

**Example:**

```python
mat = tm.Material(E=200e9, nu=0.3, rho=7800.0)

# Default hub constraint
results = tm.solve(mesh, mat, rpm=10000, num_modes=10, verbose=1)

# Custom boundary conditions
bcs = tm.bc_editor(mesh)  # or define programmatically
results = tm.solve(mesh, mat, rpm=10000, num_modes=10,
                   boundary_conditions=bcs, verbose=1)
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
    boundary_conditions: list[BoundaryCondition] | None = None,
) -> list[list[ModalResult]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rpm_values` | `ndarray \| Sequence[float]` | required | Array of RPM values |
| `boundary_conditions` | `list[BoundaryCondition] \| None` | `None` | Custom BCs (same as `solve`) |
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

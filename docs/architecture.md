# Architecture

## Overview

turbomodal is a finite element solver and visualisation toolkit for
turbomachinery modal analysis. Data flows from CAD/mesh import through a
cyclic-symmetry eigenvalue solver into post-processing and visualisation.

| Layer          | Concern                      | Language | Key Modules                          |
|----------------|------------------------------|----------|--------------------------------------|
| FEA Core       | Solver and Geometry          | C++      | `_core`, `io`, `solver`              |
| Visualisation  | Post-processing and Display  | Python   | `viz`, `bc_editor`, `mac`, `_utils`  |

## System Architecture

```
                     +---------------------+
                     |  CAD / Mesh File    |
                     | (.step, .msh, .bdf) |
                     +---------+-----------+
                               |
                               v
                  +------------+----------------+
                  |    Geometry Import (Python)  |
                  |                              |
                  |  inspect_cad() -> CadInfo    |
                  |  load_cad() / load_mesh()    |
                  +------------+----------------+
                               |
                               v
                  +------------+----------------+
                  |    FEA Solver (C++)          |
                  |                              |
                  |  CyclicSymmetrySolver        |
                  |    .solve_at_rpm()           |
                  +------------+----------------+
                               |
                               v
                  +------------+----------------+
                  |    ModalResult               |
                  |      .frequencies            |
                  |      .mode_shapes            |
                  |      .harmonic_index          |
                  |      .whirl_direction         |
                  +------------+----------------+
                               |
                               v
                  +------------+----------------+
                  |    Visualisation (Python)    |
                  |                              |
                  |  plot_cad() (CAD preview)    |
                  |  plot_full_mesh()            |
                  |  plot_campbell / plot_zzenf   |
                  |  diagnose_frequencies()      |
                  |  bc_editor (constraints)     |
                  |  mac (MAC matrix)            |
                  +-----------------------------+
```

## FEA and Geometry

### C++ Backend

The core finite element solver is written in C++17 and compiled as a static
library `turbomodal_core`. It is exposed to Python through pybind11 bindings
compiled as the `turbomodal._core` extension module.

Source files in `src/`:

| File                    | Responsibility                                     |
|-------------------------|----------------------------------------------------|
| `material.cpp`          | Isotropic material with temperature dependence      |
| `element.cpp`           | TET10 element stiffness and mass matrices           |
| `mesh.cpp`              | Mesh I/O (ZZENM format, gmsh), boundary detection   |
| `assembler.cpp`         | Global sparse matrix assembly                      |
| `rotating_effects.cpp`  | Spin-softening and centrifugal stiffening           |
| `modal_solver.cpp`      | Standard eigenvalue solver (Spectra)                |
| `cyclic_solver.cpp`     | Cyclic symmetry reduction and harmonic-index loop   |
| `added_mass.cpp`        | Fluid-structure added mass (Kwak model)             |
| `damping.cpp`           | Proportional and modal damping                      |
| `forced_response.cpp`   | Engine-order excitation forced response             |
| `mistuning.cpp`         | Fundamental Mistuning Model (FMM)                   |
| `mode_identification.cpp` | Nodal circle counting, mode family classification |

### CyclicSymmetrySolver

The central solver class. It exploits the rotational periodicity of a bladed
disk with `N` sectors by solving only a single-sector mesh. For each harmonic
index `k` in `[0, N/2]`, cyclic boundary conditions couple left and right
boundary DOFs through the complex phase factor `exp(j * 2*pi*k/N)`.

```
CyclicSymmetrySolver(mesh, material, fluid)
  .solve_at_rpm(rpm, num_modes_per_harmonic)
      -> list[ModalResult]   (one per harmonic index)
```

Each `ModalResult` contains:

- `frequencies` -- natural frequencies in Hz
- `mode_shapes` -- complex eigenvectors, shape `(n_dof, n_modes)`
- `harmonic_index` -- the nodal diameter count
- `whirl_direction` -- `+1` forward, `-1` backward, `0` standing wave
- `rpm` -- the rotational speed at which the solve was performed

### TET10 Element

All meshes use 10-node quadratic tetrahedra. The `TET10Element` class
computes 30x30 element stiffness and consistent mass matrices via 4-point
Gauss quadrature. Quadratic shape functions provide second-order convergence
for displacement and first-order convergence for stress.

### Mesh I/O

Three Python-side entry points handle geometry import:

- `inspect_cad(filepath, num_sectors)` -- lightweight CAD inspection that
  imports STEP/IGES/BREP geometry via gmsh, computes bounding-box dimensions,
  inner/outer radii, axial length, and a recommended mesh size, without
  generating any volumetric mesh. Returns a `CadInfo` dataclass.
- `load_cad(filepath, num_sectors, ...)` -- imports STEP/IGES/BREP/STL
  geometry via gmsh, automatically detects cyclic boundary surfaces, and
  generates a TET10 mesh.
- `load_mesh(filepath, num_sectors, ...)` -- imports pre-meshed files via
  meshio (NASTRAN, Abaqus, VTK, CGNS, Salome MED, XDMF) or gmsh MSH format
  via the native C++ loader.

### External Dependencies (C++)

- **Eigen 3.4.0** -- dense and sparse linear algebra. Downloaded
  automatically by CMake via `FetchContent` to avoid version conflicts with
  system Eigen.
- **Spectra** -- header-only eigenvalue solver for sparse matrices. Included
  as a git submodule at `external/spectra/`.

### Mode Identification (C++)

The `mode_identification.cpp` module provides:

- `identify_nodal_circles(mode_shape, mesh)` -- counts radial zero-crossings
  of the dominant displacement component.
- `classify_mode_family(mode_shape, mesh)` -- returns `"B"` (bending),
  `"T"` (torsion), or `"A"` (axial) based on the dominant DOF component.
- `identify_modes(result, mesh)` -- produces a vector of `ModeIdentification`
  structs with nodal diameter, nodal circle, whirl direction, frequency,
  wave velocity, participation factor, and family label.

## Visualisation

### viz.py

Provides plotting functions built on PyVista and Matplotlib:

- `plot_cad(filepath, num_sectors, ...)` -- pre-mesh CAD geometry preview
  using a lightweight surface triangulation. Supports single-sector and
  full-disk views with dimension annotations.
- `plot_full_mesh(mesh)` -- renders the full 360-degree mesh by replicating
  the single sector, without requiring a solved `ModalResult`.
- `plot_campbell` and `plot_zzenf` -- interference diagram plotting with
  `confidence_bands`, `crossing_markers`, `stator_vanes` (NPF line overlay),
  and `style` (`DiagramStyle` dataclass) to control all visual properties.
  Both functions share a unified parameter set.
- `diagnose_frequencies(results, ground_truth, num_sectors)` -- compares
  solver output against a 2-D ground truth matrix and produces error
  heatmaps, per-ND bar charts, and parity scatter plots.

### bc_editor.py

Interactive boundary condition editor for defining displacement constraints
on mesh surfaces.

### mac.py

Computes the Modal Assurance Criterion (MAC) matrix for comparing mode shape
similarity between two sets of eigenvectors.

## C++ / Python Boundary

The boundary between C++ and Python is the pybind11 module `turbomodal._core`,
built from `src/python_bindings.cpp`. It exposes the following C++ types and
functions to Python:

**Classes:**
`Material`, `NodeSet`, `Mesh`, `GlobalAssembler`, `SolverConfig`,
`SolverStatus`, `ModalResult`, `FluidConfig`, `FluidType`,
`CyclicSymmetrySolver`, `AddedMassModel`, `ModalSolver`, `DampingConfig`,
`DampingType`, `ForcedResponseConfig`, `ForcedResponseResult`,
`ForcedResponseSolver`, `ExcitationType`, `MistuningConfig`,
`MistuningResult`, `FMMSolver`, `ModeIdentification`, `GroundTruthLabel`.

**Free functions:**
`identify_nodal_circles()`, `classify_mode_family()`, `identify_modes()`.

Data exchange between C++ and Python uses Eigen-to-NumPy automatic conversion
provided by pybind11. Sparse matrices are passed as `scipy.sparse.csr_matrix`
where appropriate. Complex mode shapes are handled as `numpy.complex128`
arrays.

The Python layer adds:

- Geometry import and inspection (`io.py`) via gmsh and meshio, including
  lightweight CAD inspection (`inspect_cad`, `CadInfo`).
- High-level solver wrappers (`solver.py`) with progress bars and
  Campbell diagram extraction.
- Visualisation (`viz.py`) via PyVista and Matplotlib, including pre-mesh
  CAD preview (`plot_cad`) and full annulus mesh display (`plot_full_mesh`).
- Boundary condition editing (`bc_editor.py`) for interactive constraint
  definition.
- Modal Assurance Criterion computation (`mac.py`).

## Design Decisions

### Why Cyclic Symmetry

A bladed disk with `N` identical sectors has `N`-fold rotational symmetry.
Exploiting this reduces the eigenvalue problem from the full annulus
(`N * n_dof_sector` DOFs) to a single sector (`n_dof_sector` DOFs) solved
`N/2 + 1` times -- one per harmonic index. For a typical 36-blade disk this
is an 18x reduction in problem size, making parametric sweeps over thousands
of operating conditions tractable.

### Deferred Imports Pattern

Optional dependencies (PyVista, gmsh, Matplotlib) are imported inside
function bodies rather than at module level. This allows users to
`import turbomodal` and use the core solver without installing heavy
visualisation packages. An `ImportError` is raised only when a specific
function that requires the missing dependency is called.

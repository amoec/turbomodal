# Roadmap to Official Release
>
> **Current Build**: `v0.1.0` &nbsp;&middot;&nbsp; **Target Release**: `v1.0.0` &nbsp;&middot;&nbsp; **Last Updated**: 2026-03-01

---

## Milestone Overview

| Milestone | Description | Status |
|:---------:|-------------|:------:|
| **M1** | Core solver and visualization | Done |
| **M2** | Test coverage and CI | Done |
| **M3** | Documentation and examples | Done |
| **M4** | Animation and visualization polish | Done |
| **M5** | Packaging and distribution | In Progress |
| **M6** | Validation against commercial solvers | Not Started |
| **M7** | Performance benchmarking at scale | Not Started |
| **M8** | v1.0.0 release | Blocked on M5-M7 |

---

## Subsystem Status

### Subsystem A: FEA & Geometry (C++)

| Feature | Files | Status |
|---------|-------|:------:|
| TET10 quadratic elements (Keast Rule 6 mass quadrature) | `element.cpp/hpp` | :white_check_mark: |
| Sparse global K, M, G assembly | `assembler.cpp/hpp` | :white_check_mark: |
| Cyclic symmetry reduction | `cyclic_solver.cpp/hpp` | :white_check_mark: |
| Shift-invert Lanczos eigenvalue solver (Spectra) | `modal_solver.cpp/hpp` | :white_check_mark: |
| Complex Hermitian GEP + Lancaster QEP | `modal_solver.cpp/hpp` | :white_check_mark: |
| Centrifugal stiffening, spin softening, Coriolis | `rotating_effects.cpp/hpp` | :white_check_mark: |
| Kwak AVMI added mass model | `added_mass.cpp/hpp` | :white_check_mark: |
| Potential flow BEM (submerged disk) | `potential_flow.cpp/hpp` | :white_check_mark: |
| Rayleigh, modal, and aerodynamic damping | `damping.cpp/hpp` | :white_check_mark: |
| Harmonic forced response (modal superposition) | `forced_response.cpp/hpp` | :white_check_mark: |
| Fundamental Mistuning Model (FMM) | `mistuning.cpp/hpp` | :white_check_mark: |
| Mode identification (nodal circles, whirl) | `mode_identification.cpp/hpp` | :white_check_mark: |
| Static condensation | `static_condensation.cpp/hpp` | :white_check_mark: |
| Gmsh MSH native mesh I/O | `mesh.cpp/hpp` | :white_check_mark: |
| NASTRAN, Abaqus, VTK, CGNS, MED, XDMF (via meshio) | `io.py` | :white_check_mark: |
| STEP / IGES / BREP CAD import (gmsh OCC) | `io.py` | :white_check_mark: |
| Automatic cyclic boundary detection + node matching | `mesh.cpp/hpp` | :white_check_mark: |
| pybind11 bindings for all C++ classes | `python_bindings.cpp` | :white_check_mark: |
| Higher-order elements (TET20, HEX27) |: | :black_square_button: Future |
| Non-linear contact (friction, gaps) |: | :black_square_button: Future |
| Craig-Bampton model order reduction |: | :black_square_button: Future |

### Visualization

| Feature | Files | Status |
|---------|-------|:------:|
| Sector mesh + full annulus mesh plotting | `viz.py` | :white_check_mark: |
| Mode shape (static, animated, GIF export) | `viz.py` | :white_check_mark: |
| Full annulus mode animation | `viz.py` | :white_check_mark: |
| Campbell diagram (MAC tracking, EO lines, NPF, DiagramStyle) | `viz.py` | :white_check_mark: |
| ZZENF interference diagram (EO zig-zag, NPF, crossings, DiagramStyle) | `viz.py` | :white_check_mark: |
| Frequency diagnostics vs ground truth (`diagnose_frequencies`) | `viz.py` | :white_check_mark: |
| CAD geometry preview | `viz.py` | :white_check_mark: |
| Interactive BC editor | `viz.py` | :white_check_mark: |

---

## Testing

| Suite | Count | Platform | Status |
|-------|:-----:|----------|:------:|
| C++ unit tests | 15 suites | Linux, macOS, Windows | :white_check_mark: |
| Python tests | 120+ tests across 8 files | Linux, macOS, Windows | :white_check_mark: |
| Analytical validation (Leissa, Kwak, Coriolis, FMM, SDOF) | 5 benchmarks | All | :white_check_mark: |
| CI matrix (Python 3.10, 3.12 &times; 3 OS) | 6 jobs | GitHub Actions | :white_check_mark: |
| Wheel builds (cibuildwheel) | 3 OS | GitHub Actions | :white_check_mark: |

---

## Release Checklist

### v0.1.0: Initial Build :white_check_mark:

- [x] Solver and visualization implemented and functional
- [x] C++ solver with pybind11 bindings
- [x] 15 C++ test suites passing on all platforms
- [x] 120+ Python tests passing
- [x] CI pipeline (build + test) on Linux, macOS, Windows
- [x] Wheel builds via cibuildwheel
- [x] README, CHANGELOG, CONTRIBUTING
- [x] API documentation for all modules
- [x] Quick start tutorial and examples
- [x] Animated mode shape visualization (GIF + interactive)

### v1.0.0: Official Release :construction:

**Packaging & Distribution**

- [ ] Publish to PyPI (`pip install turbomodal`)
- [ ] Verify `pip install` from PyPI on clean environments (Linux, macOS, Windows)
- [ ] Tag `v1.0.0` on `main` and trigger release workflow
- [ ] GitHub Release with auto-generated notes and wheel artifacts

**Validation & Benchmarking**

- [ ] Benchmark against ANSYS/NASTRAN on reference bladed disk geometry
- [ ] Performance profile on 500k+ DOF meshes
- [ ] Document validation results in `docs/validation.md`

**Documentation**

- [ ] Troubleshooting guide (common solver failures, numerical conditioning)
- [ ] Expand `docs/validation.md` with benchmark tables and plots

**Code Hardening**

- [ ] Audit public API for consistent naming and signatures
- [ ] Review and lock public API surface (`__all__` exports)
- [ ] Add `py.typed` marker and verify type stubs against all public classes
- [ ] Fuzz test mesh I/O with malformed inputs

---

## Future Directions (post v1.0.0)

These are not blocking the official release but are tracked for future development.

| Feature | Priority | Complexity |
|---------|:--------:|:----------:|
| Higher-order elements (TET20, HEX27) | Medium | High |
| Adaptive mesh refinement (AMR) | Low | High |
| Craig-Bampton component mode synthesis | Medium | High |
| Non-linear contact (friction dampers, shroud gaps) | Medium | High |
| Transient time-domain dynamics | Low | High |
| GPU acceleration (CUDA sparse solvers) | Low | High |
| Distributed parallel solve (MPI) | Low | High |
| Real-time inference server (ONNX export) | Medium | Medium |
| Web-based 3D visualization (PyVista Trame) | Low | Medium |
| Additional element types (wedge, hex, shell) | Medium | Medium |

---

<sub>This roadmap is maintained manually. Version is derived from git tags via setuptools-scm.</sub>

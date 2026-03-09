# Roadmap to Official Release
>
> **Current Build**: `v0.1.0` &nbsp;&middot;&nbsp; **Target Release**: `v1.0.0` &nbsp;&middot;&nbsp; **Last Updated**: 2026-03-01

---

## Milestone Overview

| Milestone | Description | Status |
|:---------:|-------------|:------:|
| **M1** | Core solver and subsystems | Done |
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

### Subsystem B: Signals & Noise (Python)

| Feature | Files | Status |
|---------|-------|:------:|
| VirtualSensorArray (BTT, strain, accelerometer, displacement) | `sensors.py` | :white_check_mark: |
| Signal synthesis from modal superposition | `signal_gen.py` | :white_check_mark: |
| Full-annulus virtual probe model (circumferential phase, rotating/stationary) | `signal_gen.py` | :white_check_mark: |
| BTT blade passage gating and discrete arrival output | `signal_gen.py`, `sensors.py` | :white_check_mark: |
| Time vector control (custom array, t_start/t_end, damping) | `signal_gen.py` | :white_check_mark: |
| Sensor circumferential angles and blade tip profile from mesh | `sensors.py` | :white_check_mark: |
| Gaussian noise, harmonic interference | `noise.py` | :white_check_mark: |
| Sensor drift (linear, random-walk) | `noise.py` | :white_check_mark: |
| Bandwidth limiting (Butterworth LP) | `noise.py` | :white_check_mark: |
| ADC quantisation, signal dropout | `noise.py` | :white_check_mark: |
| HDF5 dataset export / import | `dataset.py` | :white_check_mark: |
| Latin Hypercube parametric sweeps | `parametric.py` | :white_check_mark: |

### Subsystem C: ML Pipeline (Python)

| Feature | Files | Status |
|---------|-------|:------:|
| Spectrogram, mel, order tracking features | `ml/features.py` | :white_check_mark: |
| Traveling wave decomposition (TWD) | `ml/features.py` | :white_check_mark: |
| Cross-spectral density (CSD) | `ml/features.py` | :white_check_mark: |
| Physics-informed features (freq ratios, temp scaling) | `ml/features.py` | :white_check_mark: |
| Tier 1: Linear (Logistic + Ridge) | `ml/models.py` | :white_check_mark: |
| Tier 2: Tree (XGBoost / LightGBM / RF) | `ml/models.py` | :white_check_mark: |
| Tier 3: SVM (SVC + SVR) | `ml/models.py` | :white_check_mark: |
| Tier 4: Shallow NN (2-layer PyTorch) | `ml/models.py` | :white_check_mark: |
| Tier 5: 1-D CNN + ResNet variant | `ml/models.py` | :white_check_mark: |
| Tier 6: Conv+BiLSTM + Transformer variant | `ml/models.py` | :white_check_mark: |
| Complexity ladder (auto tier selection) | `ml/pipeline.py` | :white_check_mark: |
| Optuna Bayesian HPO | `ml/pipeline.py` | :white_check_mark: |
| GroupKFold cross-validation | `ml/pipeline.py` | :white_check_mark: |
| MC Dropout, Deep Ensembles, heteroscedastic UQ | `ml/pipeline.py` | :white_check_mark: |
| MLflow experiment tracking (with no-op fallback) | `ml/pipeline.py` | :white_check_mark: |
| Multi-task learning (ND, whirl, amplitude, velocity) | `ml/pipeline.py` | :white_check_mark: |

### Subsystem D: Optimisation & Explainability (Python)

| Feature | Files | Status |
|---------|-------|:------:|
| Fisher Information Matrix sensor ranking | `optimization/sensor_placement.py` | :white_check_mark: |
| Greedy forward selection | `optimization/sensor_placement.py` | :white_check_mark: |
| Bayesian refinement (Optuna) | `optimization/sensor_placement.py` | :white_check_mark: |
| Minimize-sensors mode | `optimization/sensor_placement.py` | :white_check_mark: |
| SHAP values (TreeSHAP + KernelSHAP) | `optimization/explainability.py` | :white_check_mark: |
| Grad-CAM attribution | `optimization/explainability.py` | :white_check_mark: |
| 6 physics consistency rules | `optimization/explainability.py` | :white_check_mark: |
| Confidence calibration (Platt, isotonic, temp, conformal) | `optimization/explainability.py` | :white_check_mark: |
| Per-prediction explanation cards | `optimization/explainability.py` | :white_check_mark: |
| Model selection reports | `optimization/explainability.py` | :white_check_mark: |

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
| Sensor contribution heatmap | `viz.py` | :white_check_mark: |

---

## Testing

| Suite | Count | Platform | Status |
|-------|:-----:|----------|:------:|
| C++ unit tests | 15 suites | Linux, macOS, Windows | :white_check_mark: |
| Python tests | 210+ tests across 13 files | Linux, macOS, Windows | :white_check_mark: |
| Analytical validation (Leissa, Kwak, Coriolis, FMM, SDOF) | 5 benchmarks | All | :white_check_mark: |
| CI matrix (Python 3.10, 3.12 &times; 3 OS) | 6 jobs | GitHub Actions | :white_check_mark: |
| Wheel builds (cibuildwheel) | 3 OS | GitHub Actions | :white_check_mark: |

---

## Release Checklist

### v0.1.0: Initial Build :white_check_mark:

- [x] All four subsystems implemented and functional
- [x] C++ solver with pybind11 bindings
- [x] 15 C++ test suites passing on all platforms
- [x] 210+ Python tests passing
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
- [ ] Validate ML pipeline on real BTT sensor data (non-synthetic)
- [ ] Performance profile on 500k+ DOF meshes
- [ ] Memory profile for large parametric sweeps (1000+ conditions)
- [ ] Document validation results in `docs/validation.md`

**Documentation**

- [ ] Troubleshooting guide (common solver failures, numerical conditioning)
- [ ] HDF5 dataset schema specification
- [ ] ML tuning guide (feature selection, tier comparison, UQ interpretation)
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

# Validation Criteria

## Overview

This document defines the physics consistency rules and testing strategy that
govern turbomodal's modal identification solver. Predictions are evaluated
against these criteria to ensure physical plausibility.

---

## Physics Consistency Checks

The solver validates predictions against physical constraints. These checks
serve as a safety net for detecting physically impossible outputs.

### Check 1: Positive Frequency

```
frequency > 0
```

Natural frequencies of a structural system are always positive. A zero or
negative predicted frequency indicates a model failure or a degenerate input.
Applied to every sample.

### Check 2: Valid Nodal Diameter Range

```
0 <= ND <= num_sectors // 2
```

For a bladed disk with `N` sectors, the maximum independent nodal diameter is
`N/2` (integer division). Predictions outside this range are physically
meaningless because they would alias to a lower harmonic.

### Check 3: Valid Whirl Direction

```
whirl_direction in {-1, 0, 1}
```

Only three whirl states are physically meaningful: backward (-1), standing
wave (0), and forward (+1). Any other value indicates a model or encoding
error.

### Check 4: Whirl Ordering

```
For each ND group where rpm > 0:
    min(forward whirl frequencies) >= max(backward whirl frequencies)
```

In the rotating frame, Coriolis effects split the forward and backward
traveling wave frequencies. For a given nodal diameter, the forward whirl
frequency should always be greater than or equal to the backward whirl
frequency. This check is only applied when `rpm > 0` (stationary analysis has
no whirl splitting).

The check groups predictions by nodal diameter, finds the minimum forward
frequency and maximum backward frequency within each group, and flags
violations where forward < backward.

### Check 5: Wave Velocity Consistency

```
When ND > 0 and frequency > 0:
    v_expected = 2 * pi * f * R / ND
    |v_predicted - v_expected| / v_expected < 0.50
```

The circumferential wave velocity must be consistent with the predicted
frequency and nodal diameter. The tolerance is 50% relative error, which
accounts for the fact that the predicted velocity and predicted ND/frequency
may all contain independent errors.

### Output

The consistency check returns a dict with four keys:

| Key                | Shape    | Description                                      |
|--------------------|----------|--------------------------------------------------|
| `is_consistent`    | `(N,)`   | Boolean, True if all applicable checks pass      |
| `violations`       | `(N,)`   | List of string descriptions per sample           |
| `consistency_score`| `(N,)`   | Float in [0, 1], fraction of checks passed       |
| `anomaly_flag`     | `(N,)`   | Boolean, True when `consistency_score < 0.8`     |

---

## Test Suite

### Overview

The project has 15 C++ test suites and 8 Python test files providing
comprehensive coverage of the solver and Python utilities.

### C++ Test Suites

Built with `BUILD_TESTS=ON` (unit tests) and optionally
`BUILD_VALIDATION_TESTS=ON` (slow validation and integration tests).
Run via `ctest --output-on-failure` from the build directory.

| Suite                    | Module              | Tests | Description                                      |
|--------------------------|---------------------|-------|--------------------------------------------------|
| `MaterialTests`          | material            | ~5    | Material properties, temperature dependence      |
| `ElementTests`           | element             | ~8    | TET10 stiffness/mass, shape functions            |
| `MeshTests`              | mesh                | ~6    | Mesh loading, cyclic boundary detection          |
| `AssemblerTests`         | assembler           | ~5    | Global K/M assembly                              |
| `SolverTests`            | modal_solver        | ~6    | Eigenvalue solver                                |
| `CyclicTests`            | cyclic_solver       | ~8    | Cyclic symmetry transformation                   |
| `AddedMassTests`         | added_mass          | ~4    | Kwak AVMI formula                                |
| `PotentialFlowTests`     | potential_flow      | ~5    | Meridional mesh, potential flow BEM              |
| `RotatingTests`          | rotating_effects    | ~6    | Centrifugal stiffening, spin softening           |
| `DampingTests`           | damping             | 14    | Rayleigh, modal, aerodynamic, effective damping  |
| `ForcedResponseTests`    | forced_response     | 18    | Modal FRF, modal forces, EO aliasing, participation factors |
| `MistuningTests`         | mistuning           | 18    | FMM solver, random mistuning, Whitehead bound    |
| `ModeIdentificationTests`| mode_identification | 13    | Family classification, nodal circles, whirl      |
| `ValidationTests`*       | (cross-module)      | 10    | Leissa plate, Kwak added mass, Rayleigh quotient, mass conservation, FMM, Coriolis splitting |
| `IntegrationTests`*      | (cross-module)      | 3     | End-to-end: load-solve-identify, forced response pipeline, mistuning pipeline |

\* Requires `BUILD_VALIDATION_TESTS=ON`. These tests take approximately
10 minutes due to mesh loading and multiple solves.

### Python Test Files

Located under `python/tests/`:

| File                       | Description                                      |
|----------------------------|--------------------------------------------------|
| `test_python_bindings.py`  | C++ binding tests: Material, Mesh, Solver        |
| `test_io.py`               | Mesh import (Gmsh `.msh`), CAD loading           |
| `test_viz.py`              | Visualization: mesh plots, mode shapes, Campbell |
| `test_solver.py`           | High-level `solve()`, `rpm_sweep()`, `campbell_data()` |
| `test_mac.py`              | Modal Assurance Criterion computation            |
| `test_utils.py`            | Utility functions                                |
| `test_validation_python.py`| Leissa plate theory, Kwak frequency ratios, FMM tuned identity, Campbell ordering, SDOF FRF analytical |
| `test_integration.py`      | End-to-end: mesh loading, solve, mistuning, forced response |
| `conftest.py`              | Session-scoped fixtures: mesh paths, solved wedge |

### Running Tests

Run all Python tests:

```bash
pytest python/tests/ -v
```

Run only validation tests (marked with `@pytest.mark.validation`):

```bash
pytest python/tests/ -v -m validation
```

Run with coverage reporting:

```bash
pytest python/tests/ -v --cov=turbomodal --cov-report=term-missing
```

Run C++ unit tests (15 suites, ~25 seconds):

```bash
cd build && ctest --output-on-failure
```

Run C++ validation tests (slow, ~10 minutes):

```bash
cmake .. -DBUILD_TESTS=ON -DBUILD_VALIDATION_TESTS=ON -DBUILD_PYTHON=OFF
cmake --build .
ctest --output-on-failure
```

---

## Continuous Integration

### Required Dependencies

The test suite requires at minimum:

- `numpy`
- `scipy`
- `scikit-learn`
- `pytest`

### CI Configuration Notes

1. **Seed determinism**: most tests use explicit RNG seeds
   (`np.random.default_rng(seed)`) for reproducibility.

2. **Test data**: FEA-dependent tests (solver, integration, validation)
   require mesh files in `tests/test_data/` and will skip if not found.

3. **Timeouts**: Python tests complete in under 5 seconds. C++ validation
   tests take ~10 minutes total (Leissa flat disk, Kwak added mass, and
   Coriolis splitting each require full cyclic symmetry solves).

4. **Fixture dependencies**: `test_io.py`, `test_viz.py`, and
   `test_python_bindings.py` require the compiled C++ extension module
   (`turbomodal._core`) and mesh fixture files. These tests will be skipped
   if the C++ extension is not built.

5. **Session-scoped fixtures**: `conftest.py` provides session-scoped
   `wedge_mesh_path` and `leissa_mesh_path` fixtures to avoid redundant
   mesh loading. Module-scoped test fixtures (e.g. `wedge_mesh` in
   `test_solver.py`) depend on these.

6. **Pytest markers**: `@pytest.mark.validation` gates slow Python
   validation tests (Leissa, Kwak, FMM, Campbell, SDOF FRF).
   `@pytest.mark.slow` is available for other long-running tests. Register
   markers in `conftest.py` via `pytest_configure`.
